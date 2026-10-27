"""Coordinator behaviour: command failures, observation lifecycle, fresh status at setup."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, create_autospec, patch

import pytest
from aioairctrl import CoAPClient
from aiocoap.error import LibraryShutdown
from homeassistant.const import CONF_HOST, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.philips_airpurifier_coap import async_setup_entry
from custom_components.philips_airpurifier_coap import coordinator as coordinator_module
from custom_components.philips_airpurifier_coap.const import CONF_DEVICE_ID, CONF_MODEL, CONF_STATUS, DOMAIN
from custom_components.philips_airpurifier_coap.coordinator import CommandFailed, Coordinator
from custom_components.philips_airpurifier_coap.philips import (
    PhilipsAC085011C,
    PhilipsAC4236,
    PhilipsGenericFanBase,
)

HOST = "192.0.2.1"
CREATE = "custom_components.philips_airpurifier_coap.coordinator.CoAPClient.create"


def fake_client(set_result=True):
    """A CoAP client, checked against the real signatures, whose observation
    yields one status and then waits until the client shuts down."""
    client = create_autospec(CoAPClient, instance=True)
    client.set_control_values.return_value = set_result
    client.get_status.return_value = ({"D03102": 1}, 60)
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
    coordinator._observing.set()
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
async def test_long_observation_that_ends_reconnects(coordinator: Coordinator, failure) -> None:
    async def observe_status():
        yield {"D03102": 0}
        if failure:
            raise failure

    coordinator.client.observe_status = observe_status
    with (
        patch.object(coordinator_module, "MIN_OBSERVATION_SECONDS", 0),
        patch.object(coordinator, "reconnect", AsyncMock()) as reconnect,
    ):
        await coordinator._async_observe_status()
    assert coordinator.status == {"D03102": 0}
    assert not coordinator._observing.is_set()
    reconnect.assert_awaited_once()


async def test_short_observation_is_left_to_the_watchdog(coordinator: Coordinator) -> None:
    """A device that refuses observations must not be reconnected in a loop."""

    async def observe_status():
        yield {"D03102": 0}

    coordinator.client.observe_status = observe_status
    with patch.object(coordinator, "reconnect", AsyncMock()) as reconnect:
        await coordinator._async_observe_status()
    reconnect.assert_not_called()


async def test_reconnect_stops_observing_before_shutdown(coordinator: Coordinator) -> None:
    """The shutdown ends the old observation, which must not start another reconnect."""
    old_client = coordinator.client
    coordinator._start_observing()
    await asyncio.sleep(0)
    observing = coordinator._task
    assert observing is not None

    new_client = fake_client()
    with (
        patch.object(coordinator_module, "MIN_OBSERVATION_SECONDS", 0),
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


@pytest.mark.parametrize(
    ("fan_class", "confirm"),
    [(PhilipsAC085011C, True), (PhilipsAC4236, True), (PhilipsGenericFanBase, False)],
)
async def test_fan_preset_confirms_only_on_opted_in_models(fan_class, confirm) -> None:
    coordinator = SimpleNamespace(async_set_control_values=AsyncMock())
    fan = SimpleNamespace(
        coordinator=coordinator,
        CONFIRM_COMMANDS=fan_class.CONFIRM_COMMANDS,
        _available_preset_modes={"auto": {"D03102": 1, "D0310C": 0}},
        _device_status={"D0310C": 17},
        _handle_coordinator_update=lambda: None,
    )
    await PhilipsGenericFanBase.async_set_preset_mode(fan, "auto")  # type: ignore[arg-type]
    coordinator.async_set_control_values.assert_awaited_once_with({"D03102": 1, "D0310C": 0}, confirm=confirm)


async def test_failed_preset_is_not_written_optimistically() -> None:
    coordinator = SimpleNamespace(async_set_control_values=AsyncMock(side_effect=CommandFailed(HOST, {})))
    fan = SimpleNamespace(
        coordinator=coordinator,
        CONFIRM_COMMANDS=True,
        _available_preset_modes={"auto": {"D03102": 1, "D0310C": 0}},
        _device_status={"D0310C": 17},
        _handle_coordinator_update=lambda: None,
    )
    with pytest.raises(CommandFailed):
        await PhilipsGenericFanBase.async_set_preset_mode(fan, "auto")  # type: ignore[arg-type]
    assert fan._device_status == {"D0310C": 17}


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


async def test_setup_shows_the_live_status_and_keeps_stored_keys(hass: HomeAssistant) -> None:
    entry = add_entry(hass, {CONF_STATUS: {"D03102": 1, "D0310C": 0, "D03221": 3}})
    client = fake_client()
    client.get_status.return_value = ({"D03102": 0, "D0310C": 17}, 60)
    await setup(hass, entry, client)
    data = hass.data[DOMAIN][entry.entry_id]
    try:
        assert data.coordinator.status == {"D03102": 0, "D0310C": 17}
        # live values, plus the stored key the device omitted, so its entity still exists
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


async def test_setup_without_a_live_status_is_retried(hass: HomeAssistant) -> None:
    entry = add_entry(hass, {CONF_STATUS: {"D03102": 1}})
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


async def test_tiny_max_age_keeps_the_watchdog_period(coordinator: Coordinator) -> None:
    coordinator.client.get_status.return_value = ({"D03102": 1}, 0)
    await coordinator.async_first_refresh()
    assert coordinator._timeout == coordinator_module.MIN_TIMEOUT
