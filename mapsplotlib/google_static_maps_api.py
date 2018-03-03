# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

try:
    from StringIO import StringIO
except ImportError:
    from io import BytesIO as StringIO

import numpy as np
import pandas as pd
import requests
from PIL import Image


TILE_SIZE = 256                                      # Basic Mercator Google Maps tile is 256 x 256
MAX_SIN_LAT = 1. - 1e-5                              # Bound for sinus of latitude
MAX_SIZE = 640                                       # Max size of the map in pixels
SCALE = 2                                            # 1 or 2 (free plan), see Google Static Maps API docs
DEFAULT_ZOOM = 10                                    # Default zoom level, in case it cannot be determined automatically
MAPTYPE = 'roadmap'                                  # Default map type
BASE_URL = 'https://maps.googleapis.com/maps/api/staticmap?'
HTTP_SUCCESS_STATUS = 200

cache = {}                                           # Caching queries to limit API calls / speed them up


class GoogleStaticMapsAPI:
    """

    API calls to the Google Static Maps API
    Associated transformation between geographic coordinate system / pixel location

    See https://developers.google.com/maps/documentation/static-maps/intro for more info.

    """
    @classmethod
    def register_api_key(cls, api_key):
        """Register a Google Static Maps API key to enable queries to Google.
        Create your own Google Static Maps API key on https://console.developers.google.com.

        :param str api_key: the API key

        :return: None
        """
        cls._api_key = api_key

    @classmethod
    def map(
            cls, center=None, zoom=None, size=(MAX_SIZE, MAX_SIZE), scale=SCALE,
            maptype=MAPTYPE, file_format='png32', markers=None):
        """GET query on the Google Static Maps API to retrieve a static image.

        :param object center: (required if markers not present) defines the center of the map, equidistant from edges.
            This parameter takes a location as either
                * a tuple of floats (latitude, longitude)
                * or a string address (e.g. "city hall, new york, ny") identifying a unique location

        :param int zoom: (required if markers not present) defines the zoom level of the map:
            *  1: World
            *  5: Landmass/continent
            * 10: City
            * 15: Streets
            * 20: Buildings

        :param (int, int) size: (required) defines the rectangular dimensions (pixels) of the map image.
            Max size for each dimension is 640 (free account).

        :param int scale: (optional), 1 or 2 (free plan). Affects the number of pixels that are returned.
            scale=2 returns twice as many pixels as scale=1 while retaining the same coverage area and level of detail
            (i.e. the contents of the map don't change).

        :param string maptype: (optional) defines the type of map to construct. Several possible values, including
            * roadmap (default): specifies a standard roadmap image, as is normally shown on the Google Maps.
            * satellite: specifies a satellite image.
            * terrain: specifies a physical relief map image, showing terrain and vegetation.
            * hybrid:  specifies a hybrid of the satellite and roadmap image, showing a transparent layer of
                major streets and place names on the satellite image.

        :param string file_format: image format
            * png8 or png (default) specifies the 8-bit PNG format.
            * png32 specifies the 32-bit PNG format.
            * gif specifies the GIF format.
            * jpg specifies the JPEG compression format.
            * jpg-baseline

        :param {string: object} markers: points to be marked on the map, under the form of a dict with keys
            * 'color': (optional) 24-bit (0xFFFFCC) or predefined from
                {black, brown, green, purple, yellow, blue, gray, orange, red, white}
            * 'size': (optional) {tiny, mid, small}
            * 'label': (optional) specifies a single uppercase alphanumeric character from the set {A-Z, 0-9}.
                Only compatible with <mid> size markers
            * 'coordinates': list of tuples (lat, long) for which the options are common.

        :return: map image
        :rtype: PIL.Image
        """
        if not hasattr(cls, '_api_key'):
            raise KeyError('No Google Static Maps API key registered - refer to the README.')

        # For now, caching only if no markers are given
        should_cache = markers is None

        url = BASE_URL
        if center:
            url += 'center={},{}&'.format(*center) if isinstance(center, tuple) else 'center={}&'.format(center)
        if zoom:
            url += 'zoom={}&'.format(zoom)

        markers = markers if markers else []
        for marker in markers:
            if 'latitude' in marker and 'longitude' in marker:
                url += 'markers='
                for key in ['color', 'size', 'label']:
                    if key in marker:
                        url += '{}:{}%7C'.format(key, marker[key])
                url += '{},{}%7C'.format(marker['latitude'], marker['longitude'])
                url += '&'

        url += 'scale={}&'.format(scale)
        url += 'size={}x{}&'.format(*tuple(min(el, MAX_SIZE) for el in size))
        url += 'maptype={}&'.format(maptype)
        url += 'format={}&'.format(file_format)
        url += 'key={}'.format(cls._api_key)

        if url in cache:
            return cache[url]

        response = requests.get(url)
        # Checking response code, in case of error adding Google API message to the debug of requests exception
        if response.status_code != HTTP_SUCCESS_STATUS:
            print('HTTPError: {} - {}. {}'.format(response.status_code, response.reason, response.content))
        response.raise_for_status()     # This raises an error in case of unexpected status code
        # Processing the image in case of success
        img = Image.open(StringIO((response.content)))
        if should_cache:
            cache[url] = img

        return img

    @classmethod
    def to_pixel(cls, latitude, longitude):
        """Transform a pair lat/long in pixel location on a world map without zoom (absolute location).

        :param float latitude: latitude of point
        :param float longitude: longitude of point

        :return: pixel coordinates
        :rtype: pandas.Series
        """
        siny = np.clip(np.sin(latitude * np.pi / 180), -MAX_SIN_LAT, MAX_SIN_LAT)
        return pd.Series(
            [
                TILE_SIZE * (0.5 + longitude / 360),
                TILE_SIZE * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi)),
            ],
            index=['x_pixel', 'y_pixel'],
        )

    @classmethod
    def to_pixels(cls, latitudes, longitudes):
        """Transform a set of lat/long coordinates in pixel location on a world map without zoom (absolute location).

        :param pandas.Series latitudes: set of latitudes
        :param pandas.Series longitudes: set of longitudes

        :return: pixel coordinates
        :rtype: pandas.DataFrame
        """
        siny = np.clip(np.sin(latitudes * np.pi / 180), -MAX_SIN_LAT, MAX_SIN_LAT)
        return pd.concat(
            [
                TILE_SIZE * (0.5 + longitudes / 360),
                TILE_SIZE * (0.5 - np.log((1 + siny) / (1 - siny)) / (4 * np.pi)),
            ],
            axis=1, keys=['x_pixel', 'y_pixel'],
        )

    @classmethod
    def to_tile_coordinates(cls, latitudes, longitudes, center_lat, center_long, zoom, size, scale):
        """Transform a set of lat/long coordinates into pixel position in a tile. These coordinates depend on
            * the zoom level
            * the tile location on the world map

        :param pandas.Series latitudes: set of latitudes
        :param pandas.Series longitudes: set of longitudes
        :param float center_lat: center of the tile (latitude)
        :param float center_long: center of the tile (longitude)
        :param int zoom: Google maps zoom level
        :param int size: size of the tile
        :param int scale: 1 or 2 (free plan), see Google Static Maps API docs

        :return: pixel coordinates in the tile
        :rtype: pandas.DataFrame
        """
        pixels = cls.to_pixels(latitudes, longitudes)
        return scale * ((pixels - cls.to_pixel(center_lat, center_long)) * 2 ** zoom + size / 2)

    @classmethod
    def get_zoom(cls, latitudes, longitudes, size, scale):
        """Compute level of zoom needed to display all points in a single tile.

        :param pandas.Series latitudes: set of latitudes
        :param pandas.Series longitudes: set of longitudes
        :param int size: size of the tile
        :param int scale: 1 or 2 (free plan), see Google Static Maps API docs

        :return: zoom level
        :rtype: int
        """
        # Extreme pixels
        min_pixel = cls.to_pixel(latitudes.min(), longitudes.min())
        max_pixel = cls.to_pixel(latitudes.max(), longitudes.max())
        # Longitude spans from -180 to +180, latitudes only from -90 to +90
        max_amplitude = ((max_pixel - min_pixel).abs() * pd.Series([2., 1.], index=['x_pixel', 'y_pixel'])).max()
        if max_amplitude == 0 or np.isnan(max_amplitude):
            return DEFAULT_ZOOM
        return int(np.log2(2 * size / max_amplitude))
