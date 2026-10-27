"""Support for Philips AirPurifier with CoAP."""

from __future__ import annotations

import asyncio
import json
import logging
from functools import partial
from ipaddress import IPv6Address, ip_address
from os import walk
from pathlib import Path

from aioairctrl import CoAPClient
from getmac import get_mac_address
from homeassistant.components.frontend import add_extra_js_url
from homeassistant.components.http import StaticPathConfig
from homeassistant.config_entries import ConfigEntry
from homeassistant.const import CONF_HOST, CONF_NAME
from homeassistant.core import HomeAssistant
from homeassistant.exceptions import ConfigEntryNotReady
from homeassistant.helpers import config_validation as cv
from homeassistant.helpers.http import HomeAssistantView
from homeassistant.helpers.storage import Store

from .config_entry_data import ConfigEntryData
from .const import (
    CONF_DEVICE_ID,
    CONF_MODEL,
    CONF_STATUS,
    DOMAIN,
    ICONLIST_URL,
    ICONS_PATH,
    ICONS_URL,
    LOADER_PATH,
    LOADER_URL,
    PAP,
)
from .coordinator import Coordinator
from .helpers import normalize_connection_mac
from .model import DeviceInformation

_LOGGER = logging.getLogger(__name__)


PLATFORMS = [
    "binary_sensor",
    "climate",
    "fan",
    "humidifier",
    "light",
    "number",
    "select",
    "sensor",
    "switch",
]

CONFIG_SCHEMA = cv.empty_config_schema(DOMAIN)


# icons code thanks to Thomas Loven:
# https://github.com/thomasloven/hass-fontawesome/blob/master/custom_components/fontawesome/__init__.py


class ListingView(HomeAssistantView):
    """Provide a json list of the used icons."""

    requires_auth = False

    def __init__(self, url, iconpath, hass) -> None:
        """Initialize the ListingView with a URL and icon path."""
        self.url = url
        self.iconpath = iconpath
        self.hass: HomeAssistant = hass
        self.name = "Icon Listing"

    async def get(self, request, *args):
        """Call executor to avoid blocking I/O call to get list of used icons."""
        return await self.hass.async_add_executor_job(self.get_icons_list, self.iconpath)

    def get_icons_list(self, iconpath):
        """Handle GET request to provide a JSON list of the used icons."""
        icons = []
        for dirpath, _dirnames, filenames in walk(iconpath):
            icons.extend(
                [
                    {"name": (Path(dirpath[len(self.iconpath) :]) / fn[:-4]).as_posix()}
                    for fn in filenames
                    if fn.endswith(".svg")
                ]
            )
        return json.dumps(icons)


async def async_setup(hass: HomeAssistant, config) -> bool:
    """Set up the icons for the Philips AirPurifier integration."""
    _LOGGER.debug("async_setup called")

    await hass.http.async_register_static_paths([StaticPathConfig(LOADER_URL, hass.config.path(LOADER_PATH), True)])
    add_extra_js_url(hass, LOADER_URL)

    iset = PAP
    iconpath = hass.config.path(ICONS_PATH + "/" + iset)
    await hass.http.async_register_static_paths([StaticPathConfig(ICONS_URL + "/" + iset, iconpath, True)])
    hass.http.register_view(ListingView(ICONLIST_URL + "/" + iset, iconpath, hass))

    return True


async def async_get_mac_address_from_host(hass: HomeAssistant, host: str) -> str | None:
    """Get mac address from host."""
    mac_address: str | None

    # first we try if this is an ip address
    try:
        ip_addr = ip_address(host)
    except ValueError:
        # that didn't work, so try a hostname
        mac_address = await hass.async_add_executor_job(partial(get_mac_address, hostname=host))
    else:
        # it is an ip address, but it could be IPv4 or IPv6
        if ip_addr.version == 4:
            mac_address = await hass.async_add_executor_job(partial(get_mac_address, ip=host))
        else:
            ip_addr = IPv6Address(int(ip_addr))
            mac_address = await hass.async_add_executor_job(partial(get_mac_address, ip6=str(ip_addr)))
    if not mac_address:
        return None

    normalized_mac = normalize_connection_mac(mac_address)
    if normalized_mac is None:
        _LOGGER.debug("Ignoring invalid MAC address '%s' for host %s", mac_address, host)
        return None

    return normalized_mac


async def async_setup_entry(hass: HomeAssistant, entry: ConfigEntry) -> bool:
    """Set up the Philips AirPurifier integration."""

    host = entry.options.get(CONF_HOST, entry.data[CONF_HOST])
    model = entry.data[CONF_MODEL]
    name = entry.data[CONF_NAME]
    device_id = entry.data[CONF_DEVICE_ID]
    mac = await async_get_mac_address_from_host(hass, host)

    _LOGGER.debug("async_setup_entry called for host %s", host)

    try:
        client = await asyncio.wait_for(CoAPClient.create(host), timeout=25)
        _LOGGER.debug("got a valid client for host %s", host)

    except Exception as ex:
        _LOGGER.warning(r"Failed to connect to host %s: %s", host, ex)
        raise ConfigEntryNotReady from ex

    device_information = DeviceInformation(host=host, mac=mac, model=model, name=name, device_id=device_id)

    # The device answers a status request only once its status changes, which
    # can take hours, so the entities start from the last notified status,
    # saved for this, or else the status stored when the entry was created.
    store = _status_store(hass, entry)
    status = await store.async_load() or entry.data.get(CONF_STATUS)
    coordinator = Coordinator(hass, client, host, status, store)
    if status is None:
        # old entries have no status data
        try:
            await coordinator.async_first_refresh()
        except ConfigEntryNotReady:
            await coordinator.shutdown()
            raise
        status = coordinator.status
        hass.config_entries.async_update_entry(entry, data={**entry.data, CONF_STATUS: status})

    # Entities are created for the keys in latest_status. A device can omit
    # keys depending on its state, so the stored keys keep the entity set
    # stable across restarts; entities read their values from the live status.
    latest_status = {**entry.data[CONF_STATUS], **status}

    # store the data in the hass.data
    config_entry_data = ConfigEntryData(
        device_information=device_information,
        coordinator=coordinator,
        latest_status=latest_status,
        client=client,
    )
    hass.data.setdefault(DOMAIN, {})[entry.entry_id] = config_entry_data

    try:
        await hass.config_entries.async_forward_entry_setups(entry, PLATFORMS)
    except BaseException:
        await coordinator.shutdown()
        hass.data[DOMAIN].pop(entry.entry_id)
        raise

    entry.async_on_unload(entry.add_update_listener(_async_update_listener))

    return True


def _status_store(hass: HomeAssistant, entry: ConfigEntry) -> Store:
    """Where the entry's last notified status is saved across restarts."""
    return Store(hass, 1, f"{DOMAIN}.{entry.entry_id}.status")


async def async_remove_entry(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Delete the entry's saved status."""
    await _status_store(hass, entry).async_remove()


async def _async_update_listener(hass: HomeAssistant, entry: ConfigEntry) -> None:
    """Reload the config entry when options change."""
    await hass.config_entries.async_reload(entry.entry_id)


async def async_unload_entry(hass: HomeAssistant, entry: ConfigEntry):
    """Unload a config entry."""

    for p in PLATFORMS:
        await hass.config_entries.async_forward_entry_unload(entry, p)

    config_entry_data: ConfigEntryData = hass.data[DOMAIN][entry.entry_id]
    await config_entry_data.coordinator.shutdown()

    hass.data[DOMAIN].pop(entry.entry_id)

    return True
