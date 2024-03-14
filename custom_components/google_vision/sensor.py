"""
Perform image processing with Google Vision
"""
import logging
import re
from datetime import timedelta

from google.cloud import vision
from google.oauth2 import service_account
from homeassistant.util import slugify

import voluptuous as vol

import homeassistant.util.dt as dt_util
from homeassistant.core import split_entity_id
import homeassistant.helpers.config_validation as cv

from homeassistant.exceptions import HomeAssistantError
from homeassistant.components.camera import Image
from homeassistant.components.sensor import (
    RestoreSensor,
    PLATFORM_SCHEMA,
    DEVICE_CLASSES_SCHEMA,
    STATE_CLASSES_SCHEMA,
    CONF_STATE_CLASS,
)
from homeassistant.const import (
    CONF_NAME,
    CONF_DEVICE_CLASS,
    CONF_UNIT_OF_MEASUREMENT,
    CONF_ENTITY_ID,
    CONF_UNIQUE_ID,
    STATE_UNAVAILABLE,
)
from .const import (
    DOMAIN,
    ATTR_LAST_DETECTION,
)

_LOGGER = logging.getLogger(__name__)

SCAN_INTERVAL = timedelta(days=365)  # Effectively disable scan.

DEFAULT_TIMEOUT = 10

CONF_API_KEY_FILE = "api_key_file"
CONF_SOURCES = "sources"
CONF_EXPECT_DIGITS = "expected_digits"
CONF_DECIMALS = "decimals"
CONF_KEYWORD = "keyword"
CONF_KEYWORD_POS = "keyword_position"
FILE = "file"
OBJECT = "object"

CAMERA_SCHEMA = vol.Schema(
    {
        vol.Required(CONF_ENTITY_ID): cv.entity_id,
        vol.Optional(CONF_NAME): cv.string,
        vol.Optional(CONF_KEYWORD): cv.string,
        vol.Optional(CONF_KEYWORD_POS): cv.string,
        vol.Required(CONF_EXPECT_DIGITS): cv.positive_int,
        vol.Required(CONF_DECIMALS): cv.positive_int,
        vol.Optional(CONF_UNIQUE_ID): cv.string,
        vol.Optional(CONF_DEVICE_CLASS): DEVICE_CLASSES_SCHEMA,
        vol.Optional(CONF_STATE_CLASS): STATE_CLASSES_SCHEMA,
        vol.Optional(CONF_UNIT_OF_MEASUREMENT): cv.string,
    }
)

PLATFORM_SCHEMA = PLATFORM_SCHEMA.extend(
    {
        vol.Required(CONF_API_KEY_FILE): cv.string,
        vol.Required(CONF_SOURCES): vol.All(cv.ensure_list, [CAMERA_SCHEMA]),
    }
)


def setup_platform(hass, config, add_devices, discovery_info=None):
    """Set up platform."""

    entities = []
    for camera in config[CONF_SOURCES]:
        entities.append(
            Gvision(
                config.get(CONF_API_KEY_FILE),
                camera[CONF_ENTITY_ID],
                camera.get(CONF_KEYWORD),
                camera.get(CONF_KEYWORD_POS),
                camera[CONF_EXPECT_DIGITS],
                camera[CONF_DECIMALS],
                camera.get(CONF_NAME),
                camera.get(CONF_UNIQUE_ID),
                camera.get(CONF_DEVICE_CLASS),
                camera.get(CONF_STATE_CLASS),
                camera.get(CONF_UNIT_OF_MEASUREMENT),
            )
        )
    add_devices(entities)


class Vision(object):
    """Interact with Google Vision."""

    def __init__(self, api_key_file):
        credentials = service_account.Credentials.from_service_account_file(
            api_key_file
        )
        scoped_credentials = credentials.with_scopes(
            ["https://www.googleapis.com/auth/cloud-platform"]
        )
        self._client = vision.ImageAnnotatorClient(credentials=scoped_credentials)

    def object_localization(self, image_bytes):
        """Return the list of objects in an image from the imge bytes."""
        return self._client.object_localization(image=vision.Image(content=image_bytes))

    def document_text_detection(self, image_bytes):
        """Return the list of objects in an image from the imge bytes."""
        return self._client.document_text_detection(image=vision.Image(content=image_bytes))
    
class Gvision(RestoreSensor):
    """Perform object recognition with Google Vision."""

    def __init__(
        self, 
        api_key_file, 
        camera_entity, 
        keyword,
        keyword_pos,
        expected_digits,
        decimals,
        name=None,
        unique_id=None,
        device_class=None,
        state_class=None,
        unit_of_measurement=None,
    ):
        """Init with the client."""
        self._api = Vision(api_key_file)
        self._camera_entity = camera_entity
        if name:  
            self._name = name
        else:
            entity_name = split_entity_id(camera_entity)[1]
            self._name = "{} {}".format("google vision", entity_name)
        if unique_id:  
            self._attr_unique_id = unique_id
        else:
            self._attr_unique_id =f'{DOMAIN}_{slugify(self._name)}_sensor'
        
        self._keyword = keyword.lower() if keyword else None
        self._keyword_pos = keyword_pos if keyword_pos else None
        self._expected_digits = expected_digits
        self._decimals = decimals
        self._state = None  # The number of instances of interest
        self._last_detection = None

        if device_class:
            self._attr_device_class = device_class
        if state_class:
            self._attr_state_class = state_class
        if unit_of_measurement:
            self._attr_native_unit_of_measurement = unit_of_measurement

    async def async_added_to_hass(self) -> None:
        """Run when about to be added to hass."""
        await super().async_added_to_hass()

        last_state = await self.async_get_last_state()
        last_sensor_state = await self.async_get_last_sensor_data()
        if (
            not last_state
            or not last_sensor_state
            or last_state.state == STATE_UNAVAILABLE
        ):
            return


        self._state = self._attr_native_value = last_sensor_state.native_value
        if ATTR_LAST_DETECTION in last_state.attributes:
            self._last_detection = self._attr_last_detection = last_state.attributes[ATTR_LAST_DETECTION]
        
    def process_image(self, image):
        """Process an image."""
        self._state = None

        response = self._api.document_text_detection(image)
        objects = response.text_annotations
        # _LOGGER.debug("GVision response: %s", objects)

        if not len(objects) > 0:
            return

        index = None
        keyword_obj = None
        prev_obj = None
        for obj in objects:
            _LOGGER.debug("GVision object: %s", obj.description)
            if self._keyword_pos == "after":
                if obj.description.lower().startswith(self._keyword) and prev_obj:
                    _LOGGER.debug("Found %s: %s", self._keyword_pos, prev_obj.description)
                    value = re.sub('[^0-9]','', prev_obj.description)
                    if (value.isnumeric()):
                        if len(value) == self._expected_digits:
                            index = float(value) / (10.0 ** self._decimals)
                            break
                        elif self._decimals > 0 and len(value) == self._expected_digits - 1:
                            index = float(value) / (10.0 ** (self._decimals - 1))
                            break
                    else:
                        _LOGGER.error("value is not numeric: %s", prev_obj.description)
            elif self._keyword_pos == "before":
                if keyword_obj:
                    _LOGGER.debug("Found %s: %s", self._keyword_pos, obj.description)
                    value = re.sub('[^0-9]','', obj.description)
                    if value.isnumeric():
                        if len(value) == self._expected_digits:
                            index = float(value) / (10.0 ** self._decimals)
                            break
                        elif self._decimals > 0 and len(value) == self._expected_digits - 1:
                            index = float(value) / (10.0 ** (self._decimals - 1))
                            break
                    else:
                        _LOGGER.error("value is not numeric: %s", obj.description)
                if obj.description.lower().startswith(self._keyword):
                    keyword_obj = obj
            else:
                value = obj.description
                if value.isnumeric():
                    if len(value) == self._expected_digits:
                        index = float(value) / (10.0 ** self._decimals)
                        break
                    elif self._decimals > 0 and len(value) == self._expected_digits - 1:
                        index = float(value) / (10.0 ** (self._decimals - 1))
                        break

            prev_obj = obj

        if not index:
            return
        
        self._state = self._attr_native_value = index
        self._last_detection = self._attr_last_detection = dt_util.now().strftime("%Y-%m-%d %H:%M:%S")

    @property
    def camera_entity(self):
        """Return camera entity id from process pictures."""
        return self._camera_entity

    @property
    def state(self):
        """Return the state of the entity."""
        return self._state

    @property
    def name(self):
        """Return the name of the sensor."""
        return self._name

    @property
    def extra_state_attributes(self):
        """Return device specific state attributes."""
        attr = {}
        if self._last_detection:
            attr[
                ATTR_LAST_DETECTION
            ] = self._last_detection
        return attr

    async def async_update(self) -> None:
        """Update image and process it.

        This method is a coroutine.
        """
        camera = self.hass.components.camera

        try:
            image: Image = await camera.async_get_image(
                self.camera_entity, timeout=DEFAULT_TIMEOUT
            )

        except HomeAssistantError as err:
            _LOGGER.error("Error on receive image from entity: %s", err)
            return

        # process image data
        await self.hass.async_add_executor_job(self.process_image, image.content)

