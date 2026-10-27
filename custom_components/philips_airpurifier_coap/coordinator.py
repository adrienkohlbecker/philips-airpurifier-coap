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
from homeassistant.helpers.storage import Store

from .const import DOMAIN

_LOGGER = logging.getLogger(__name__)

# The device notifies only when its status changes, so silence says nothing
# about it; a ping every PING_INTERVAL seconds does. A purifier in Wi-Fi power
# save misses the odd ping, so it counts as gone after PING_MISSES in a row.
PING_INTERVAL = 30
PING_MISSES = 3
# A notification-less observation is registered again after this many seconds,
# in case its registration got lost; the device keeps one per address, so the
# renewal replaces it.
OBSERVE_REFRESH = 900
# How long a notified status waits before it is saved, batching bursts.
STATUS_SAVE_DELAY = 60
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

    def __init__(
        self,
        hass: HomeAssistant,
        client: CoAPClient,
        host: str,
        status: dict[str, Any] | None,
        store: Store | None = None,
    ) -> None:
        """Initialize the Coordinator.

        :param hass: HomeAssistant instance.
        :param client: CoAPClient instance.
        :param host: Host address of the device.
        :param status: Initial status of the device.
        :param store: Where each notified status is saved, for the next start.
        """

        self.hass = hass
        self.client = client
        self.host = host
        self.status = status
        # Whether the device answers pings; entities are unavailable otherwise.
        self.reachable = True

        self._store = store
        self._listeners: list[CALLBACK_TYPE] = []
        self._task: Task | None = None

        self._reconnect_task: Task | None = None
        # Set once shut down: nothing may reconnect or observe after that.
        self._closed = False
        self._command_lock = asyncio.Lock()
        self._last_notified = hass.loop.time()
        # Set once the device notified this run, so self.status is live.
        self._notified = False
        self._liveness_task: Task | None = hass.async_create_background_task(
            self._liveness(), f"philips_airpurifier_coap liveness {host}"
        )

    async def shutdown(self):
        """Shutdown the API connection."""

        self._closed = True
        if self._task is not None:
            self._task.cancel()
            self._task = None

        if self._reconnect_task is not None:
            self._reconnect_task.cancel()

        if self._liveness_task is not None:
            self._liveness_task.cancel()
            self._liveness_task = None

        if self.client is not None:
            await self.client.shutdown()

        # Replaces a pending delayed save, which would otherwise land after a
        # reload read the file, or recreate it after the entry was removed.
        if self._store is not None and self._notified:
            await self._store.async_save(self.status)

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

    async def _liveness(self) -> None:
        """Ping the device every PING_INTERVAL, and keep its observation going.

        A device missing PING_MISSES pings in a row turns unreachable; one
        answering again reconnects at once, since it may have restarted and
        lost its key and observation, failing a command in flight. While
        reachable, an observation that ended is reconnected, and one silent
        for OBSERVE_REFRESH registered again, unless a command is in flight:
        it proves the connection itself and reconnects when it fails.
        """
        misses = 0
        while True:
            await asyncio.sleep(PING_INTERVAL)
            try:
                if await self.client.ping():
                    misses = 0
                    if not self.reachable:
                        _LOGGER.info("%s answers again, reconnecting", self.host)
                        self._set_reachable(True)
                        await self._reconnect_unless_running()
                    elif not self._command_lock.locked():
                        await self._keep_observing()
                    continue
                misses += 1
                if misses == PING_MISSES:
                    _LOGGER.warning("%s missed %d pings, marking it unavailable", self.host, misses)
                    self._set_reachable(False)
            except Exception:
                # The loop must outlive any one check, or the device would never
                # turn unavailable or be observed again.
                _LOGGER.exception("Checking %s failed", self.host)

    async def _keep_observing(self) -> None:
        """Reconnect an observation that ended, renew one silent for too long."""
        if self._reconnect_task is not None and not self._reconnect_task.done():
            return
        if self._task is None or self._task.done():
            await self.reconnect()
        elif self.hass.loop.time() - self._last_notified > OBSERVE_REFRESH:
            _LOGGER.debug("no notification from %s for %ss, observing again", self.host, OBSERVE_REFRESH)
            self._start_observing()

    def _set_reachable(self, reachable: bool) -> None:
        self.reachable = reachable
        self._notify_listeners()

    def _notify_listeners(self) -> None:
        for update_callback in list(self._listeners):
            try:
                update_callback()
            except Exception:
                # One broken entity must not stop the others from updating.
                _LOGGER.exception("Updating an entity of %s failed", self.host)

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
            with contextlib.suppress(Exception):
                await self.client.shutdown()
            self.client = await CoAPClient.create(self.host)
            if self._notified and self.status is not None:
                self.client.remember_status(self.status)
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
        """Read the status from the device, for an entry that has none stored.

        The device answers only once its status changes, which can take long.
        """

        try:
            self.status, _ = await self.client.get_status()
        except Exception as ex:
            raise ConfigEntryNotReady(
                translation_domain=DOMAIN,
                translation_key="first_refresh_failed",
                translation_placeholders={"host": self.host, "error": repr(ex)},
            ) from ex
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
        """Observe the status of the device, saving each status it notifies.

        An observation that ends is reconnected by the liveness loop, within
        PING_INTERVAL, so a device that keeps ending them is not hammered.
        """
        try:
            async for status in self.client.observe_status():
                self.status = status
                self._notified = True
                self._last_notified = self.hass.loop.time()
                if self._store is not None:
                    self._store.async_delay_save(lambda: self.status, STATUS_SAVE_DELAY)
                self._notify_listeners()
        except Exception as ex:
            _LOGGER.warning("Observing %s stopped (%r), reconnecting soon", self.host, ex)
        else:
            _LOGGER.warning("Observing %s stopped, reconnecting soon", self.host)

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
                    # The observation registers as soon as its task first runs.
                    while confirm and not self.client.observing:
                        await asyncio.sleep(0.05)
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

        self._last_notified = self.hass.loop.time()
        self._task = asyncio.create_task(self._async_observe_status())
