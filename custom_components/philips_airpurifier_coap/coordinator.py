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

from .const import DOMAIN
from .timer import Timer

_LOGGER = logging.getLogger(__name__)

MISSED_PACKAGE_COUNT = 3
# The shortest silence watchdog period: a device announcing a tiny Max-Age
# would otherwise make the watchdog reconnect back to back.
MIN_TIMEOUT = 60
# An observation that ends sooner than this reconnects through the watchdog
# instead of at once, so a device that refuses observations is not hammered.
MIN_OBSERVATION_SECONDS = 60
# How long a command waits for a reconnect in progress, or a confirmed command
# for the observation that confirms it.
READY_TIMEOUT = 20
# How long a command may take once sent, resends and confirmation included,
# so a deaf device fails a service call instead of stalling it for minutes.
COMMAND_TIMEOUT = 30


class CommandFailed(HomeAssistantError):
    """A device did not take a command."""

    def __init__(self, host: str, data: dict[str, Any]) -> None:
        """Describe which command the device at host did not take."""
        super().__init__(
            translation_domain=DOMAIN,
            translation_key="command_failed",
            translation_placeholders={"host": host, "command": str(data)},
        )


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
        self._timeout: int = MIN_TIMEOUT
        # Set once shut down: nothing may reconnect or observe after that.
        self._closed = False
        # Set while an observation delivers notifications, which confirmed
        # commands need.
        self._observing = asyncio.Event()
        self._command_lock = asyncio.Lock()

        self._timer_disconnected = Timer(
            timeout=self._timeout * MISSED_PACKAGE_COUNT,
            callback=self._watchdog,
            autostart=True,
        )
        self._timer_disconnected.setAutoRestart(True)

    async def shutdown(self):
        """Shutdown the API connection."""

        self._closed = True
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
        """Reconnect to the API connection, replacing a reconnect in progress."""

        if self._closed:
            return
        try:
            if self._reconnect_task is not None:
                self._reconnect_task.cancel()
                self._reconnect_task = None

            self._reconnect_task = asyncio.create_task(self._reconnect())

        except Exception:
            _LOGGER.exception("Exception on starting reconnect!")

    async def _watchdog(self) -> None:
        """Reconnect after a silence, unless a command is in flight.

        A command proves the connection itself and reconnects when it fails;
        replacing the client under it would fail it instead.
        """
        if not self._command_lock.locked():
            await self._reconnect_unless_running()

    async def _reconnect_unless_running(self) -> None:
        """Reconnect unless a reconnect is already under way."""
        if self._reconnect_task is None or self._reconnect_task.done():
            await self.reconnect()

    async def _reconnect(self):
        """Reconnect to the API connection."""

        try:
            # Stop observing first: an observation ended by the shutdown would
            # otherwise start another reconnect.
            if self._task:
                self._task.cancel()
                self._task = None
            self._observing.clear()
            with contextlib.suppress(Exception):
                await self.client.shutdown()
            self.client = await CoAPClient.create(self.host)
            if self._closed:
                await self.client.shutdown()
                return
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
        except Exception as ex:
            raise ConfigEntryNotReady(
                translation_domain=DOMAIN,
                translation_key="first_refresh_failed",
                translation_placeholders={"host": self.host, "error": repr(ex)},
            ) from ex
        self._timeout = max(timeout, MIN_TIMEOUT)
        if self._timer_disconnected is not None:
            self._timer_disconnected.setTimeout(self._timeout * MISSED_PACKAGE_COUNT)
        _LOGGER.debug("finished first refresh for host %s", self.host)

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
        """Observe the status of the device, reconnecting once the observation ends.

        A long-lived observation that ends reconnects at once, since the
        entities would otherwise freeze until the silence watchdog fires. One
        that ends soon after starting is left to the watchdog, so a device
        that keeps refusing observations is retried at its pace, not in a loop.
        """
        started = self.hass.loop.time()
        try:
            async for status in self.client.observe_status():
                self.status = status
                self._observing.set()
                self._timer_disconnected.reset()
                for update_callback in list(self._listeners):
                    try:
                        update_callback()
                    except Exception:
                        # One broken entity must not end the observation for all.
                        _LOGGER.exception("Updating an entity of %s failed", self.host)
        except Exception as ex:
            reason = repr(ex)
        else:
            reason = "it ended"
        finally:
            self._observing.clear()
        if self.hass.loop.time() - started < MIN_OBSERVATION_SECONDS:
            _LOGGER.debug("Observing %s stopped early (%s), leaving it to the watchdog", self.host, reason)
            return
        _LOGGER.warning("Observing %s stopped (%s), reconnecting", self.host, reason)
        await self._reconnect_unless_running()

    async def async_set_control_value(self, key: str, value: Any, confirm: bool = False) -> None:
        """Set one control value; see async_set_control_values."""
        await self.async_set_control_values({key: value}, confirm=confirm)

    async def async_set_control_values(self, data: dict[str, Any], confirm: bool = False) -> None:
        """Send a command, raising CommandFailed unless the device took it.

        With confirm, the device must also report the new values (see
        aioairctrl's set_control_values), since some firmware accepts a
        command and keeps its old mode; the command first waits for a live
        observation to confirm it. Commands wait for a reconnect in progress
        and run one at a time, within READY_TIMEOUT plus COMMAND_TIMEOUT. A
        failure reconnects: the device stopped answering, or the observation
        went deaf.
        """
        async with self._command_lock:
            try:
                async with asyncio.timeout(READY_TIMEOUT):
                    while self._reconnect_task is not None and not self._reconnect_task.done():
                        await asyncio.shield(self._reconnect_task)
                    if confirm:
                        await self._observing.wait()
            except TimeoutError as ex:
                await self._reconnect_unless_running()
                raise CommandFailed(self.host, data) from ex
            try:
                async with asyncio.timeout(COMMAND_TIMEOUT):
                    applied = await self.client.set_control_values(data=data, confirm=confirm)
            except Exception as ex:
                await self._reconnect_unless_running()
                raise CommandFailed(self.host, data) from ex
            if not applied:
                await self._reconnect_unless_running()
                raise CommandFailed(self.host, data)

    def _start_observing(self) -> None:
        """Schedule state observation."""

        if self._closed:
            return
        if self._task:
            self._task.cancel()
            self._task = None

        self._task = asyncio.create_task(self._async_observe_status())
        self._timer_disconnected.reset()
