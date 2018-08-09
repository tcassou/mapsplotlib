# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np
import pandas as pd
from genty import genty
from genty import genty_dataset
from nose.tools import eq_
from nose.tools import ok_
from numpy.testing import assert_array_almost_equal
from numpy.testing import assert_array_equal

from mapsplotlib.google_static_maps_api import GoogleStaticMapsAPI
# from numpy.testing import assert_raises


@genty
class GoogleStaticMapsAPITest(unittest.TestCase):

    @genty_dataset(
        some_key=('foobarbaz',),
    )
    def test_register_api_key(self, key):
        ok_(not hasattr(GoogleStaticMapsAPI, '_api_key'))
        GoogleStaticMapsAPI.register_api_key(key)
        ok_(hasattr(GoogleStaticMapsAPI, '_api_key'))
        eq_(GoogleStaticMapsAPI._api_key, key)

    # For now it queries without an API key seem to be allowed still
    # def test_query_without_api_key(self):
    #     assert_raises(KeyError, GoogleStaticMapsAPI.map)

    @genty_dataset(
        amsterdam=(52.3702, 4.8952, 131.481031, 84.131289),
        paris=(48.8566, 2.3522, 129.672676, 88.071271),
        nan_lat=(np.nan, 4.8952, 131.481031, np.nan),
        nan_lon=(52.3702, np.nan, np.nan, 84.131289),
        all_nan=(np.nan, np.nan, np.nan, np.nan),
    )
    def test_to_pixel(self, latitude, longitude, x_pixel, y_pixel):
        pixels = GoogleStaticMapsAPI.to_pixel(latitude, longitude)
        ok_(isinstance(pixels, pd.Series))
        assert_array_equal(pixels.index, ['x_pixel', 'y_pixel'])
        assert_array_almost_equal(pixels.values, [x_pixel, y_pixel], decimal=6)

    @genty_dataset(
        one_entry=([52.3702], [4.8952], [[131.481031, 84.131289]]),
        multiple_entries=([52.3702, 48.8566], [4.8952, 2.3522], [[131.481031, 84.131289], [129.672676, 88.071271]]),
        one_lat_nan=([52.3702, np.nan], [4.8952, 2.3522], [[131.481031, 84.131289], [129.672676, np.nan]]),
        one_lon_nan=([52.3702, 48.8566], [np.nan, 2.3522], [[np.nan, 84.131289], [129.672676, 88.071271]]),
        one_row_nan=([np.nan, 48.8566], [np.nan, 2.3522], [[np.nan, np.nan], [129.672676, 88.071271]]),
    )
    def test_to_pixels(self, latitudes, longitudes, expected):
        pixels = GoogleStaticMapsAPI.to_pixels(pd.Series(latitudes), pd.Series(longitudes))
        ok_(isinstance(pixels, pd.DataFrame))
        assert_array_equal(pixels.index, range(len(latitudes)))
        assert_array_equal(pixels.columns, ['x_pixel', 'y_pixel'])
        assert_array_almost_equal(pixels, expected, decimal=6)

    @genty_dataset(
        one_entry=([52.3702], [4.8952], 52.3702, 4.8952, 10, 640, 2, [[640.0, 640.0]]),
        one_entry_some_zoom=([52.3702], [4.8952], 52.3702, 4.8952, 6, 640, 2, [[640.0, 640.0]]),
        one_entry_some_size=([52.3702], [4.8952], 52.3702, 4.8952, 10, 400, 2, [[400.0, 400.0]]),
        one_entry_some_scale=([52.3702], [4.8952], 52.3702, 4.8952, 10, 640, 1, [[320.0, 320.0]]),
        multiple_entries=(
            [52.3702, 48.8566], [4.8952, 2.3522], 50.61, 3.62, 8, 640, 2,
            [[1104.3, -389.4], [178.4, 1627.8]]
        ),
        multiple_entries_some_zoom=(
            [52.3702, 48.8566], [4.8952, 2.3522], 50.61, 3.62, 5, 640, 2,
            [[698., 511.3], [582.3, 763.5]]
        ),
        multiple_entries_some_size=(
            [52.3702, 48.8566], [4.8952, 2.3522], 50.61, 3.62, 8, 320, 2,
            [[784.3, -709.4], [-141.6, 1307.8]]
        ),
        multiple_entries_some_scale=(
            [52.3702, 48.8566], [4.8952, 2.3522], 50.61, 3.62, 8, 640, 1,
            [[552.1, -194.7], [89.2, 813.9]]
        ),
        one_lat_nan=(
            [np.nan, 48.8566], [4.8952, 2.3522], 48.8566, 3.62, 8, 640, 2,
            [[1104.3, np.nan], [178.4, 640.]]
        ),
        one_lon_nan=(
            [52.3702, 48.8566], [np.nan, 2.3522], 50.61, 2.3522, 8, 640, 2,
            [[np.nan, -389.4], [640., 1627.8]]
        ),
        one_row_nan=(
            [52.3702, np.nan], [4.8952, np.nan], 52.3702, 4.8952, 8, 640, 2,
            [[640., 640.], [np.nan, np.nan]]
        ),
    )
    def test_to_tile_coordinates(self, latitudes, longitudes, center_lat, center_lon, zoom, size, scale, expected):
        coords = GoogleStaticMapsAPI.to_tile_coordinates(
            pd.Series(latitudes), pd.Series(longitudes), center_lat, center_lon, zoom, size, scale)
        ok_(isinstance(coords, pd.DataFrame))
        assert_array_equal(coords.index, range(len(latitudes)))
        assert_array_equal(coords.columns, ['x_pixel', 'y_pixel'])
        assert_array_almost_equal(coords, expected, decimal=1)

    @genty_dataset(
        one_entry=([52.3702], [4.8952], 640, 2, 10),
        one_entry_some_size=([52.3702], [4.8952], 238, 2, 10),
        one_entry_some_scale=([52.3702], [4.8952], 640, 1, 10),
        multiple_entries=([52.3702, 48.8566], [4.8952, 2.3522], 640, 2, 8),
        multiple_entries_some_size=([52.3702, 48.8566], [4.8952, 2.3522], 430, 2, 7),
        multiple_entries_some_scale=([52.3702, 48.8566], [4.8952, 2.3522], 640, 1, 8),
        one_lat_nan=([np.nan, 48.8566], [4.8952, 2.3522], 640, 2, 8),
        one_lon_nan=([52.3702, 48.8566], [np.nan, 2.3522], 640, 2, 8),
        one_row_nan=([np.nan, 48.8566], [np.nan, 2.3522], 640, 2, 10),
        no_entry=([], [], 640, 2, 10),
    )
    def test_get_zoom(self, latitudes, longitudes, size, scale, expected):
        eq_(GoogleStaticMapsAPI.get_zoom(pd.Series(latitudes), pd.Series(longitudes), size, scale), expected)
