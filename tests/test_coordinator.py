"""Coordinator behaviour: command failures, dead observations, fresh status at setup."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from homeassistant.const import CONF_HOST, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError
from pytest_homeassistant_custom_component.common import MockConfigEntry

from custom_components.philips_airpurifier_coap import async_setup_entry
from custom_components.philips_airpurifier_coap.const import CONF_DEVICE_ID, CONF_MODEL, CONF_STATUS, DOMAIN
from custom_components.philips_airpurifier_coap.coordinator import Coordinator

HOST = "192.0.2.1"


def fake_client(statuses=(), set_result=True):
    """A CoAP client whose observation yields statuses, then ends."""
    client = MagicMock()
    client.shutdown = AsyncMock()
    client.set_control_values = AsyncMock(return_value=set_result)
    client.get_status = AsyncMock(return_value=({"D03102": 1}, 60))

    async def observe_status():
        for status in statuses:
            yield status

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
    coordinator.client.set_control_values = AsyncMock(
        side_effect=outcome if isinstance(outcome, Exception) else None, return_value=outcome
    )
    with (
        patch.object(coordinator, "reconnect", AsyncMock()) as reconnect,
        pytest.raises(HomeAssistantError),
    ):
        await coordinator.async_set_control_value("D0310C", 0)
    reconnect.assert_awaited_once()


@pytest.mark.parametrize("failure", [None, ValueError("bad payload")])
async def test_ended_observation_reconnects(coordinator: Coordinator, failure) -> None:
    async def observe_status():
        yield {"D03102": 0}
        if failure:
            raise failure

    coordinator.client.observe_status = observe_status
    with patch.object(coordinator, "reconnect", AsyncMock()) as reconnect:
        await coordinator._async_observe_status()
    assert coordinator.status == {"D03102": 0}
    reconnect.assert_awaited_once()


async def test_reconnect_stops_observing_before_shutdown(hass: HomeAssistant, coordinator: Coordinator) -> None:
    """A shutdown that ends the old observation must not start a second reconnect."""
    old_client = coordinator.client
    blocked = asyncio.Event()

    async def observe_status():
        yield {"D03102": 1}
        await blocked.wait()  # never set: the observation waits for notifications
        yield {}

    old_client.observe_status = observe_status
    coordinator._start_observing()
    await asyncio.sleep(0)
    observing = coordinator._task
    assert observing is not None

    new_client = fake_client()
    new_client.observe_status = observe_status
    with (
        patch(
            "custom_components.philips_airpurifier_coap.coordinator.CoAPClient.create",
            AsyncMock(return_value=new_client),
        ),
        patch.object(coordinator, "reconnect", wraps=coordinator.reconnect) as reconnect,
    ):
        await coordinator._reconnect()
        await asyncio.gather(observing, return_exceptions=True)
    assert observing.cancelled()
    old_client.shutdown.assert_awaited_once()
    assert coordinator.client is new_client
    reconnect.assert_not_called()


def entry_with_stale_status(hass: HomeAssistant) -> MockConfigEntry:
    entry = MockConfigEntry(
        domain=DOMAIN,
        data={
            CONF_HOST: HOST,
            CONF_MODEL: "AC0850/11",
            CONF_NAME: "Bedroom",
            CONF_DEVICE_ID: "device",
            CONF_STATUS: {"D03102": 1, "D0310C": 0},
        },
    )
    entry.add_to_hass(hass)
    return entry


async def test_setup_shows_the_live_status_not_the_stored_one(hass: HomeAssistant) -> None:
    entry = entry_with_stale_status(hass)
    client = fake_client()
    client.get_status = AsyncMock(return_value=({"D03102": 0, "D0310C": 17}, 60))
    with (
        patch("custom_components.philips_airpurifier_coap.CoAPClient.create", AsyncMock(return_value=client)),
        patch(
            "custom_components.philips_airpurifier_coap.async_get_mac_address_from_host", AsyncMock(return_value=None)
        ),
        patch.object(hass.config_entries, "async_forward_entry_setups", AsyncMock()),
    ):
        assert await async_setup_entry(hass, entry)
    data = hass.data[DOMAIN][entry.entry_id]
    try:
        assert data.coordinator.status == {"D03102": 0, "D0310C": 17}
        assert data.latest_status == {"D03102": 0, "D0310C": 17}
    finally:
        await data.coordinator.shutdown()


async def test_setup_without_a_live_status_is_retried(hass: HomeAssistant) -> None:
    entry = entry_with_stale_status(hass)
    client = fake_client()
    client.get_status = AsyncMock(side_effect=TimeoutError("no reply"))
    with (
        patch("custom_components.philips_airpurifier_coap.CoAPClient.create", AsyncMock(return_value=client)),
        patch(
            "custom_components.philips_airpurifier_coap.async_get_mac_address_from_host", AsyncMock(return_value=None)
        ),
        pytest.raises(ConfigEntryNotReady),
    ):
        await async_setup_entry(hass, entry)
    client.shutdown.assert_awaited_once()
