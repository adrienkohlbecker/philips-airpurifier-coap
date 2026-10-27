"""Coordinator behaviour: command failures, observation lifecycle, liveness, status at setup."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, create_autospec, patch

import pytest
from aioairctrl import CoAPClient
from aiocoap.error import LibraryShutdown
from homeassistant.const import CONF_HOST, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.philips_airpurifier_coap import async_remove_entry, async_setup_entry
from custom_components.philips_airpurifier_coap import coordinator as coordinator_module
from custom_components.philips_airpurifier_coap.const import CONF_DEVICE_ID, CONF_MODEL, CONF_STATUS, DOMAIN
from custom_components.philips_airpurifier_coap.coordinator import CommandFailed, Coordinator
from custom_components.philips_airpurifier_coap.philips import (
    PhilipsAC085011C,
    PhilipsAC4236,
    PhilipsGenericFanBase,
)
from custom_components.philips_airpurifier_coap.sensor import PhilipsSensor

HOST = "192.0.2.1"
CREATE = "custom_components.philips_airpurifier_coap.coordinator.CoAPClient.create"


def fake_client(set_result=True):
    """A CoAP client, checked against the real signatures, whose observation
    yields one status and then waits until the client shuts down."""
    client = create_autospec(CoAPClient, instance=True)
    client.set_control_values.return_value = set_result
    client.get_status.return_value = ({"D03102": 1}, 60)
    client.ping.return_value = True
    client.observing = True
    stopped = asyncio.Event()

    async def shutdown():
        stopped.set()

    async def observe_status():
        yield {"D03102": 1}
        await stopped.wait()

    client.shutdown.side_effect = shutdown
    client.observe_status = observe_status
    return client


@pytest.fixture
async def coordinator(hass: HomeAssistant):
    coordinator = Coordinator(hass, fake_client(), HOST, {"D03102": 1})
    yield coordinator
    await coordinator.shutdown()


async def test_command_passes_confirm_through(coordinator: Coordinator) -> None:
    await coordinator.async_set_control_values({"D0310C": 0}, confirm=True)
    coordinator.client.set_control_values.assert_awaited_once_with(data={"D0310C": 0}, confirm=True)


@pytest.mark.parametrize("outcome", [False, TimeoutError("no reply")])
async def test_failed_command_raises_and_reconnects(coordinator: Coordinator, outcome) -> None:
    if isinstance(outcome, Exception):
        coordinator.client.set_control_values.side_effect = outcome
    else:
        coordinator.client.set_control_values.return_value = outcome
    with (
        patch.object(coordinator, "reconnect", AsyncMock()) as reconnect,
        pytest.raises(CommandFailed),
    ):
        await coordinator.async_set_control_value("D0310C", 0)
    reconnect.assert_awaited_once()


async def test_confirmed_command_fails_without_an_observation(coordinator: Coordinator) -> None:
    coordinator.client.observing = False
    with (
        patch.object(coordinator_module, "READY_TIMEOUT", 0.01),
        patch.object(coordinator, "reconnect", AsyncMock()) as reconnect,
        pytest.raises(CommandFailed),
    ):
        await coordinator.async_set_control_value("D0310C", 0, confirm=True)
    coordinator.client.set_control_values.assert_not_called()
    reconnect.assert_awaited_once()


async def test_command_waits_for_a_reconnect_in_progress(coordinator: Coordinator) -> None:
    new_client = fake_client()
    created = asyncio.Event()

    async def slow_create(host):
        await created.wait()
        return new_client

    with patch(CREATE, side_effect=slow_create):
        await coordinator.reconnect()
        command = asyncio.create_task(coordinator.async_set_control_value("D03102", 0))
        await asyncio.sleep(0)
        assert not command.done()
        created.set()
        await command
    new_client.set_control_values.assert_awaited_once()


async def test_failure_during_a_reconnect_does_not_restart_it(coordinator: Coordinator) -> None:
    created = asyncio.Event()

    async def slow_create(host):
        await created.wait()
        return fake_client()

    with patch(CREATE, side_effect=slow_create) as create:
        await coordinator.reconnect()
        running = coordinator._reconnect_task
        await coordinator._reconnect_unless_running()
        assert coordinator._reconnect_task is running
        created.set()
        await running
    create.assert_called_once()


@pytest.mark.parametrize("failure", [None, ValueError("bad payload")])
async def test_an_ended_observation_is_left_to_the_liveness_check(coordinator: Coordinator, failure) -> None:
    """Reconnecting from the observation itself would hammer a device that ends them."""

    async def observe_status():
        yield {"D03102": 0}
        if failure:
            raise failure

    coordinator.client.observe_status = observe_status
    with patch.object(coordinator, "reconnect", AsyncMock()) as reconnect:
        await coordinator._async_observe_status()
    assert coordinator.status == {"D03102": 0}
    reconnect.assert_not_called()


async def test_liveness_reconnects_an_ended_observation(coordinator: Coordinator) -> None:
    with patch.object(coordinator, "reconnect", AsyncMock()) as reconnect:
        await coordinator._keep_observing()
    reconnect.assert_awaited_once()


async def test_liveness_renews_a_silent_observation_on_the_same_client(coordinator: Coordinator) -> None:
    coordinator._start_observing()
    observing = coordinator._task
    coordinator._last_notified -= coordinator_module.OBSERVE_REFRESH + 1
    with patch.object(coordinator, "reconnect", AsyncMock()) as reconnect:
        await coordinator._keep_observing()
    reconnect.assert_not_called()
    assert coordinator._task is not observing
    await asyncio.sleep(0)
    assert observing is not None
    assert observing.cancelled()


async def test_liveness_leaves_a_quiet_observation_alone(coordinator: Coordinator) -> None:
    coordinator._start_observing()
    observing = coordinator._task
    with patch.object(coordinator, "reconnect", AsyncMock()) as reconnect:
        await coordinator._keep_observing()
    reconnect.assert_not_called()
    assert coordinator._task is observing


async def run_liveness(coordinator: Coordinator, pings: list[bool]) -> list[bool]:
    """Run the liveness loop over the scripted pings; return its reachability changes."""
    assert coordinator._liveness_task is not None
    coordinator._liveness_task.cancel()
    seen: list[bool] = []
    answered = asyncio.Event()
    script = iter(pings)

    async def ping():
        try:
            return next(script)
        except StopIteration:
            answered.set()
            await asyncio.Event().wait()

    coordinator.client.ping.side_effect = ping

    def record() -> None:
        if coordinator.reachable != (seen[-1] if seen else True):
            seen.append(coordinator.reachable)

    coordinator._listeners.append(record)
    with patch.object(coordinator_module, "PING_INTERVAL", 0):
        task = asyncio.create_task(coordinator._liveness())
        await asyncio.wait_for(answered.wait(), 1)
        task.cancel()
    return seen


async def test_a_device_missing_pings_is_unavailable_until_it_answers(coordinator: Coordinator) -> None:
    coordinator._start_observing()
    with patch.object(coordinator, "reconnect", AsyncMock()) as reconnect:
        seen = await run_liveness(coordinator, [False, True, False, False, False, False, True])
    # Two misses in a row are tolerated; the third makes it unreachable, once.
    assert seen == [False, True]
    assert coordinator.reachable
    reconnect.assert_awaited_once()


async def test_reconnect_stops_observing_before_shutdown(coordinator: Coordinator) -> None:
    """The shutdown ends the old observation, which must not start another reconnect."""
    old_client = coordinator.client
    coordinator._start_observing()
    await asyncio.sleep(0)
    observing = coordinator._task
    assert observing is not None

    new_client = fake_client()
    with (
        patch(CREATE, AsyncMock(return_value=new_client)),
        patch.object(coordinator, "reconnect", wraps=coordinator.reconnect) as reconnect,
    ):
        await coordinator._reconnect()
        await asyncio.gather(observing, return_exceptions=True)
    assert observing.cancelled()
    old_client.shutdown.assert_awaited_once()
    assert coordinator.client is new_client
    reconnect.assert_not_called()


async def test_nothing_reconnects_after_shutdown(hass: HomeAssistant) -> None:
    """A command failing because of the shutdown must not revive the coordinator."""
    client = fake_client()
    coordinator = Coordinator(hass, client, HOST, {"D03102": 1})
    sent = asyncio.Event()
    released = asyncio.Event()

    async def set_control_values(data, confirm=False):
        sent.set()
        await released.wait()
        raise LibraryShutdown

    client.set_control_values.side_effect = set_control_values
    with patch(CREATE, AsyncMock(return_value=fake_client())) as create:
        command = asyncio.create_task(coordinator.async_set_control_value("D03102", 0))
        await sent.wait()
        await coordinator.shutdown()
        released.set()
        with pytest.raises(CommandFailed):
            await command
        await coordinator.reconnect()
        coordinator._start_observing()
    create.assert_not_called()
    assert coordinator._task is None
    assert coordinator._reconnect_task is None


def fan_of(fan_class, status: dict, set_control_values: AsyncMock):
    """A fan entity of fan_class on a coordinator stub holding status."""
    fan = object.__new__(fan_class)
    fan.coordinator = SimpleNamespace(status=status, async_set_control_values=set_control_values)  # type: ignore[assignment]
    fan._available_preset_modes = {"auto": {"D03102": 1, "D0310C": 0}}
    fan._handle_coordinator_update = lambda: None  # type: ignore[method-assign]
    return fan


@pytest.mark.parametrize(
    ("fan_class", "confirm"),
    [(PhilipsAC085011C, True), (PhilipsAC4236, True), (PhilipsGenericFanBase, False)],
)
async def test_fan_preset_confirms_only_on_opted_in_models(fan_class, confirm) -> None:
    send = AsyncMock()
    fan = fan_of(fan_class, {"D03102": 1, "D0310C": 17}, send)
    await fan.async_set_preset_mode("auto")
    send.assert_awaited_once_with({"D03102": 1, "D0310C": 0}, confirm=confirm)


async def test_unconfirmed_preset_is_shown_at_once() -> None:
    fan = fan_of(PhilipsGenericFanBase, {"D03102": 1, "D0310C": 17}, AsyncMock())
    await fan.async_set_preset_mode("auto")
    assert fan.coordinator.status == {"D03102": 1, "D0310C": 0}


async def test_confirmed_preset_keeps_a_newer_report() -> None:
    """The device reported the preset, then left it, before the command returned."""
    status = {"D03102": 1, "D0310C": 17}
    fan = fan_of(PhilipsAC085011C, status, AsyncMock())

    async def send(data, confirm):
        fan.coordinator.status = {"D03102": 1, "D0310C": 17}

    fan.coordinator.async_set_control_values = send
    await fan.async_set_preset_mode("auto")
    assert fan.coordinator.status == {"D03102": 1, "D0310C": 17}


async def test_failed_preset_is_not_shown() -> None:
    fan = fan_of(PhilipsGenericFanBase, {"D03102": 1, "D0310C": 17}, AsyncMock(side_effect=CommandFailed(HOST, {})))
    with pytest.raises(CommandFailed):
        await fan.async_set_preset_mode("auto")
    assert fan.coordinator.status == {"D03102": 1, "D0310C": 17}


async def test_sensor_without_its_key_is_unavailable() -> None:
    sensor = object.__new__(PhilipsSensor)
    sensor.coordinator = SimpleNamespace(status={"D03102": 1}, reachable=True)  # type: ignore[assignment]
    sensor.kind = "D03221"
    assert not sensor.available
    sensor.coordinator.status["D03221"] = 3
    assert sensor.available


async def test_sensor_without_its_key_keeps_its_default_icon() -> None:
    """HA reads the icon of an unavailable entity too."""
    sensor = object.__new__(PhilipsSensor)
    sensor.coordinator = SimpleNamespace(status={"D03102": 1}, reachable=True)  # type: ignore[assignment]
    sensor.kind = "D03221"
    sensor._icon_map = {0: "mdi:low", 10: "mdi:high"}
    sensor._norm_icon = "mdi:low"
    assert sensor.icon == "mdi:low"


async def test_a_failing_entity_does_not_end_the_observation(coordinator: Coordinator) -> None:
    seen = []

    def broken():
        raise KeyError("D03221")

    async def observe_status():
        yield {"D03102": 0}
        yield {"D03102": 1}

    coordinator._listeners = [broken, lambda: seen.append(coordinator.status)]
    coordinator.client.observe_status = observe_status
    await coordinator._async_observe_status()
    assert seen == [{"D03102": 0}, {"D03102": 1}]


async def test_liveness_leaves_a_command_in_flight_alone(coordinator: Coordinator) -> None:
    with patch.object(coordinator, "reconnect", AsyncMock()) as reconnect:
        async with coordinator._command_lock:
            await run_liveness(coordinator, [True])
        reconnect.assert_not_called()
        await run_liveness(coordinator, [True])
    reconnect.assert_awaited_once()


async def test_a_notified_status_is_saved(coordinator: Coordinator) -> None:
    store = Mock()
    coordinator._store = store

    async def observe_status():
        yield {"D03102": 0}

    coordinator.client.observe_status = observe_status
    await coordinator._async_observe_status()
    store.async_delay_save.assert_called_once()
    assert store.async_delay_save.call_args.args[0]() == {"D03102": 0}


async def test_an_unreachable_device_is_unavailable() -> None:
    fan = object.__new__(PhilipsGenericFanBase)
    fan.coordinator = SimpleNamespace(status={"D03102": 1}, reachable=False)  # type: ignore[assignment]
    assert not fan.available
    fan.coordinator.reachable = True
    assert fan.available


async def test_a_stalled_command_fails_within_the_command_timeout(coordinator: Coordinator) -> None:
    async def stall(data, confirm=False):
        await asyncio.sleep(10)

    coordinator.client.set_control_values.side_effect = stall
    with (
        patch.object(coordinator_module, "COMMAND_TIMEOUT", 0.01),
        patch.object(coordinator, "reconnect", AsyncMock()) as reconnect,
        pytest.raises(CommandFailed),
    ):
        await asyncio.wait_for(coordinator.async_set_control_value("D03102", 0), 1)
    reconnect.assert_awaited_once()


def add_entry(hass: HomeAssistant, data_extra: dict) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={CONF_HOST: HOST, CONF_MODEL: "AC0850/11", CONF_NAME: "Bedroom", CONF_DEVICE_ID: "device", **data_extra},
    )
    entry.add_to_hass(hass)
    return entry


async def setup(hass: HomeAssistant, entry: MockConfigEntry, client) -> None:
    with (
        patch("custom_components.philips_airpurifier_coap.CoAPClient.create", AsyncMock(return_value=client)),
        patch(
            "custom_components.philips_airpurifier_coap.async_get_mac_address_from_host",
            AsyncMock(return_value=None),
        ),
        patch.object(hass.config_entries, "async_forward_entry_setups", AsyncMock()),
    ):
        assert await async_setup_entry(hass, entry)


async def test_setup_starts_from_the_stored_status_without_asking_the_device(hass: HomeAssistant) -> None:
    """The device answers a status request only once its status changes."""
    entry = add_entry(hass, {CONF_STATUS: {"D03102": 1, "D0310C": 0, "D03221": 3}})
    client = fake_client()
    await setup(hass, entry, client)
    data = hass.data[DOMAIN][entry.entry_id]
    try:
        client.get_status.assert_not_called()
        assert data.coordinator.status == {"D03102": 1, "D0310C": 0, "D03221": 3}
    finally:
        await data.coordinator.shutdown()


async def test_setup_starts_from_the_saved_status_and_keeps_stored_keys(hass: HomeAssistant, hass_storage) -> None:
    entry = add_entry(hass, {CONF_STATUS: {"D03102": 1, "D0310C": 0, "D03221": 3}})
    hass_storage[f"{DOMAIN}.{entry.entry_id}.status"] = {
        "version": 1,
        "key": f"{DOMAIN}.{entry.entry_id}.status",
        "data": {"D03102": 0, "D0310C": 17},
    }
    await setup(hass, entry, fake_client())
    data = hass.data[DOMAIN][entry.entry_id]
    try:
        assert data.coordinator.status == {"D03102": 0, "D0310C": 17}
        # saved values, plus the stored key the device omitted, so its entity still exists
        assert data.latest_status == {"D03102": 0, "D0310C": 17, "D03221": 3}
        assert entry.data[CONF_STATUS] == {"D03102": 1, "D0310C": 0, "D03221": 3}
    finally:
        await data.coordinator.shutdown()


async def test_setup_stores_the_status_of_an_old_entry(hass: HomeAssistant) -> None:
    entry = add_entry(hass, {})
    await setup(hass, entry, fake_client())
    data = hass.data[DOMAIN][entry.entry_id]
    try:
        assert entry.data[CONF_STATUS] == {"D03102": 1}
    finally:
        await data.coordinator.shutdown()


async def test_setup_of_an_old_entry_without_a_status_is_retried(hass: HomeAssistant) -> None:
    entry = add_entry(hass, {})
    client = fake_client()
    client.get_status.side_effect = TimeoutError("no reply")
    with (
        patch("custom_components.philips_airpurifier_coap.CoAPClient.create", AsyncMock(return_value=client)),
        patch(
            "custom_components.philips_airpurifier_coap.async_get_mac_address_from_host",
            AsyncMock(return_value=None),
        ),
        pytest.raises(ConfigEntryNotReady),
    ):
        await async_setup_entry(hass, entry)
    client.shutdown.assert_awaited_once()


async def test_removing_an_entry_deletes_its_saved_status(hass: HomeAssistant, hass_storage) -> None:
    entry = add_entry(hass, {CONF_STATUS: {"D03102": 1}})
    key = f"{DOMAIN}.{entry.entry_id}.status"
    hass_storage[key] = {"version": 1, "key": key, "data": {"D03102": 0}}
    await async_remove_entry(hass, entry)
    assert key not in hass_storage
