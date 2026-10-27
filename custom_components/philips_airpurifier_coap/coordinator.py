"""Module containing the Coordinator class to manage data requests from the Philips API."""

import asyncio
import contextlib
import logging
from asyncio.tasks import Task
from collections.abc import Callable
from typing import Any

from aioairctrl import CoAPClient
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.exceptions import ConfigEntryNotReady, HomeAssistantError

from .timer import Timer

_LOGGER = logging.getLogger(__name__)

MISSED_PACKAGE_COUNT = 3


class CommandFailed(HomeAssistantError):
    """A device did not take a command."""

    def __init__(self, host: str, data: dict[str, Any]) -> None:
        """Describe which command the device at host did not take."""
        super().__init__(f"{host} did not apply {data}")


class Coordinator:
    """Class to coordinate the data requests from the Philips API."""

    def __init__(self, hass: HomeAssistant, client: CoAPClient, host: str, status: dict[str, Any] | None) -> None:
        """Initialize the Coordinator.

        :param hass: HomeAssistant instance.
        :param client: CoAPClient instance.
        :param host: Host address of the device.
        :param status: Initial status of the device.
        """

        self.hass = hass
        self.client = client
        self.host = host
        self.status = status

        self._listeners: list[CALLBACK_TYPE] = []
        self._task: Task | None = None

        self._reconnect_task: Task | None = None
        self._timeout: int = 60

        self._timer_disconnected = Timer(
            timeout=self._timeout * MISSED_PACKAGE_COUNT,
            callback=self.reconnect,
            autostart=True,
        )
        self._timer_disconnected.setAutoRestart(True)

    async def shutdown(self):
        """Shutdown the API connection."""

        # Before the client: an observation ended by the shutdown would reconnect.
        if self._task is not None:
            self._task.cancel()
            self._task = None

        if self._reconnect_task is not None:
            self._reconnect_task.cancel()

        if self._timer_disconnected is not None:
            self._timer_disconnected.cancel()

        if self.client is not None:
            await self.client.shutdown()

    async def reconnect(self):
        """Reconnect to the API connection."""

        try:
            if self._reconnect_task is not None:
                self._reconnect_task.cancel()
                self._reconnect_task = None

            self._reconnect_task = asyncio.create_task(self._reconnect())

        except Exception:
            _LOGGER.exception("Exception on starting reconnect!")

    async def _reconnect(self):
        """Reconnect to the API connection."""

        try:
            # Stop observing first: an observation ended by the shutdown would
            # otherwise start another reconnect.
            if self._task:
                self._task.cancel()
                self._task = None
            with contextlib.suppress(Exception):
                await self.client.shutdown()
            self.client = await CoAPClient.create(self.host)
            self._start_observing()

        except asyncio.CancelledError:
            # Silently drop this exception, because we are responsible for it.
            # Reconnect took to long
            pass

        except Exception as ex:
            _LOGGER.warning("Reconnecting to %s failed: %r", self.host, ex)

    async def async_first_refresh(self) -> None:
        """Refresh the data for the first time."""

        try:
            self.status, timeout = await self.client.get_status()
            self._timeout = timeout
            if self._timer_disconnected is not None:
                self._timer_disconnected.setTimeout(timeout * MISSED_PACKAGE_COUNT)
            _LOGGER.debug("finished first refresh for host %s", self.host)

        except Exception as ex:
            _LOGGER.exception("Config not ready, first refresh failed for host %s", self.host)
            raise ConfigEntryNotReady from ex

    @callback
    def async_add_listener(self, update_callback: CALLBACK_TYPE) -> Callable[[], None]:
        """Listen for data updates."""

        start_observing = not self._listeners
        self._listeners.append(update_callback)

        if start_observing:
            self._start_observing()

        @callback
        def remove_listener() -> None:
            """Remove update listener."""
            self.async_remove_listener(update_callback)

        return remove_listener

    @callback
    def async_remove_listener(self, update_callback) -> None:
        """Remove data update."""

        self._listeners.remove(update_callback)

        if not self._listeners and self._task:
            self._task.cancel()
            self._task = None

    async def _async_observe_status(self) -> None:
        """Observe the status of the device, reconnecting once the observation ends."""

        try:
            async for status in self.client.observe_status():
                self.status = status
                self._timer_disconnected.reset()
                for update_callback in self._listeners:
                    update_callback()
        except Exception as ex:
            _LOGGER.warning("Observing %s failed, reconnecting: %r", self.host, ex)
        else:
            _LOGGER.warning("Observation of %s ended, reconnecting", self.host)
        # Without this the entities would freeze until the silence watchdog fires.
        await self.reconnect()

    async def async_set_control_value(self, key: str, value: Any, confirm: bool = False) -> None:
        """Set one control value; see async_set_control_values."""
        await self.async_set_control_values({key: value}, confirm=confirm)

    async def async_set_control_values(self, data: dict[str, Any], confirm: bool = False) -> None:
        """Send a command, raising HomeAssistantError unless the device took it.

        With confirm, the device must also report the new values (see
        aioairctrl's set_control_values), since some firmware accepts a
        command and keeps its old mode. A failure also reconnects: the device
        stopped answering, or the observation went deaf.
        """
        try:
            applied = await self.client.set_control_values(data=data, confirm=confirm)
        except Exception as ex:
            await self.reconnect()
            raise CommandFailed(self.host, data) from ex
        if not applied:
            await self.reconnect()
            raise CommandFailed(self.host, data)

    def _start_observing(self) -> None:
        """Schedule state observation."""

        if self._task:
            self._task.cancel()
            self._task = None

        self._task = asyncio.create_task(self._async_observe_status())
        self._timer_disconnected.reset()
