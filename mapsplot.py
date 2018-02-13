# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import unicode_literals

import warnings

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import math
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import utm
import os

from mapsplotlib.google_static_maps_api import GoogleStaticMapsAPI
from mapsplotlib.google_static_maps_api import MAPTYPE
from mapsplotlib.google_static_maps_api import MAX_SIZE
from mapsplotlib.google_static_maps_api import SCALE
from mapsplotlib.google_static_maps_api import INITIALRESOLUTION



BLANK_THRESH = 2 * 1e-3     # Value below which point in a heatmap should be blank


def register_api_key(api_key):
    """Register a Google Static Maps API key to enable queries to Google.
    Create your own Google Static Maps API key on https://console.developers.google.com.

    :param str api_key: the API key

    :return: None
    """
    GoogleStaticMapsAPI.register_api_key(api_key)


def background_and_pixels(latitudes, longitudes, size, maptype):
    """Queries the proper background map and translate geo coordinated into pixel locations on this map.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param int size: target size of the map, in pixels
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: map and pixels
    :rtype: (PIL.Image, pandas.DataFrame)
    """
    # From lat/long to pixels, zoom and position in the tile
    center_lat = (latitudes.max() + latitudes.min()) / 2
    center_long = (longitudes.max() + longitudes.min()) / 2
    zoom = GoogleStaticMapsAPI.get_zoom(latitudes, longitudes, size, SCALE)
    pixels = GoogleStaticMapsAPI.to_tile_coordinates(latitudes, longitudes, center_lat, center_long, zoom, size, SCALE)
    # Google Map
    img = GoogleStaticMapsAPI.map(
        center=(center_lat, center_long),
        zoom=zoom,
        scale=SCALE,
        size=(size, size),
        maptype=maptype,
    )
    return img, pixels


def scatter(latitudes, longitudes, colors=None, maptype=MAPTYPE):
    """Scatter plot over a map. Can be used to visualize clusters by providing the marker colors.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param pandas.Series colors: marker colors, as integers
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: None
    """
    width = SCALE * MAX_SIZE-50
    colors = pd.Series(0, index=latitudes.index) if colors is None else colors
    img, pixels = background_and_pixels(latitudes, longitudes, MAX_SIZE, maptype)
    t = np.array(img)
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(img),origin='lower')                                              # Background map
    plt.scatter(                                                            # Scatter plot
        pixels['x_pixel'],
        pixels['y_pixel'],
        c=colors,
        s=width / 40,
        linewidth=0,
        alpha=0.5,
    )
    plt.gca().invert_yaxis()                                                # Origin of map is upper left
    plt.axis([0, width, width, 0])                                          # Remove margin
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def plot_markers(markers, maptype=MAPTYPE):
    """Plot markers on a map.

    :param pandas.DataFrame markers: DataFrame with at least 'latitude' and 'longitude' columns, and optionally
        * 'color' column, see GoogleStaticMapsAPI docs for more info
        * 'label' column, see GoogleStaticMapsAPI docs for more info
        * 'size' column, see GoogleStaticMapsAPI docs for more info
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: None
    """
    # Checking input columns
    fields = markers.columns.intersection(['latitude', 'longitude', 'color', 'label', 'size'])
    if len(fields) == 0 or 'latitude' not in fields or 'longitude' not in fields:
        msg = 'Input dataframe should contain at least colums \'latitude\' and \'longitude\' '
        msg += '(and columns \'color\', \'label\', \'size\' optionally).'
        raise KeyError(msg)
    # Checking NaN input
    nans = (markers.latitude.isnull() | markers.longitude.isnull())
    if nans.sum() > 0:
        warnings.warn('Ignoring {} example(s) containing NaN latitude or longitude.'.format(nans.sum()))
    # Querying map
    img = GoogleStaticMapsAPI.map(
        scale=SCALE,
        markers=markers[fields].loc[~nans].T.to_dict().values(),
        maptype=maptype,
    )
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(img))
    plt.tight_layout()
    plt.axis('off')
    plt.show()


def heatmap(latitudes, longitudes, values, resolution=None, maptype=MAPTYPE):
    """Plot a geographical heatmap of the given metric.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param pandas.Series values: series of sample values
    :param int resolution: resolution (in pixels) for the heatmap
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: None
    """
    img, pixels = background_and_pixels(latitudes, longitudes, MAX_SIZE, maptype)
    # Smooth metric
    z = grid_density_gaussian_filter(
        zip(pixels['x_pixel'], pixels['y_pixel'], values),
        MAX_SIZE * SCALE,
        resolution=resolution if resolution else MAX_SIZE * SCALE,          # Heuristic for pretty plots
    )
    # Plot
    width = SCALE * MAX_SIZE
    plt.figure(figsize=(10, 10))
    plt.imshow(np.array(img))                                               # Background map
    plt.imshow(z, origin='lower', extent=[0, width, 0, width], alpha=0.5)  # Foreground, transparent heatmap
    plt.scatter(pixels['x_pixel'], pixels['y_pixel'], s=1)                  # Markers of all points
    plt.gca().invert_yaxis()                                                # Origin of map is upper left
    plt.axis([0, width, width, 0])                                          # Remove margin
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def density_plot(latitudes, longitudes, resolution=None, maptype=MAPTYPE):
    """Given a set of geo coordinates, draw a density plot on a map.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param int resolution: resolution (in pixels) for the heatmap
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: None
    """
    heatmap(latitudes, longitudes, np.ones(latitudes.shape[0]), resolution=resolution, maptype=maptype)


def grid_density_gaussian_filter(data, size, resolution=None, smoothing_window=None):
    """Smoothing grid values with a Gaussian filter.

    :param [(float, float, float)] data: list of 3-dimensional grid coordinates
    :param int size: grid size
    :param int resolution: desired grid resolution
    :param int smoothing_window: size of the gaussian kernels for smoothing

    :return: smoothed grid values
    :rtype: numpy.ndarray
    """
    resolution = resolution if resolution else size
    k = (resolution - 1) / size
    w = smoothing_window if smoothing_window else int(0.01 * resolution)    # Heuristic
    imgw = (resolution + 2 * w)
    img = np.zeros((imgw, imgw))
    for x, y, z in data:
        ix = int(x * k) + w
        iy = int(y * k) + w
        if 0 <= ix < imgw and 0 <= iy < imgw:
            img[iy][ix] += z
    z = ndi.gaussian_filter(img, (w, w))                                    # Gaussian convolution
    z[z <= BLANK_THRESH] = np.nan                                           # Making low values blank
    return z[w:-w, w:-w]


def polygons(latitudes, longitudes, clusters, maptype=MAPTYPE):
    """Plot clusters of points on map, including them in a polygon defining their convex hull.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param pandas.Series clusters: marker clusters, as integers
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :return: None
    """
    width = SCALE * MAX_SIZE
    img, pixels = background_and_pixels(latitudes, longitudes, MAX_SIZE, maptype)

    polygons = []
    for c in clusters.unique():
        in_polygon = clusters == c
        if in_polygon.sum() < 3:
            print('[WARN] Cannot draw polygon for cluster {} - only {} samples.'.format(c, in_polygon.sum()))
            continue
        cluster_pixels = pixels.loc[clusters == c]
        polygons.append(Polygon(cluster_pixels.iloc[ConvexHull(cluster_pixels).vertices], closed=True))

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111)
    plt.imshow(np.array(img))                                               # Background map
    p = PatchCollection(polygons, cmap='jet', alpha=0.15)                   # Collection of polygons
    p.set_array(clusters.unique())
    ax.add_collection(p)
    plt.scatter(                                                           # Scatter plot
        pixels['x_pixel'],
        pixels['y_pixel'],
        c=clusters,
        s=width / 40,
        linewidth=0,
        alpha=0.25,
    )
    plt.gca().invert_yaxis()                                                # Origin of map is upper left
    plt.axis([0, width, width, 0])                                          # Remove margin
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def scatter_with_wgs84(latitudes, longitudes, z, colors=None, maptype=MAPTYPE, **kargs):
    """Scatter plot over a map. Can be used to visualize clusters by providing the marker colors.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param pandas.Series colors: marker colors, as integers
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :param string title: title of the plot
    :param string xlabel: x label
    :param string ylabel: y label
    :param boolean cbar: add color bar to the graphic
    :param string cLabel: Label in color bar
    :param boolean saveFig: save Figs
    :param string figname: name of the file to save the figure

    :return: None
    """
    ###########################
    # Default optional values #
    ###########################
    title = kargs.pop('title', ' ')
    xlabel = kargs.pop('xlabel', ' ')
    ylabel = kargs.pop('ylabel', ' ')
    cLabel = kargs.pop('cLabel',' ')
    cbar = kargs.pop('cbar', False)
    saveFig = kargs.pop('saveFig', False)
    figName = kargs.pop('figName','googleMaps_wLines')

    if kargs: raise TypeError('extra keywords: %s' % kargs)


    width = SCALE * MAX_SIZE
    #########################
    # Image with google map #
    #########################
    img, pixels = background_and_pixels(latitudes, longitudes, MAX_SIZE, maptype)

    ###########################
    # zoom normally = 16      #
    ###########################
    center_lat = (latitudes.max() + latitudes.min()) / 2
    center_long = (longitudes.max() + longitudes.min()) / 2
    zoom = GoogleStaticMapsAPI.get_zoom(latitudes, longitudes, MAX_SIZE, SCALE)

    ###################################
    # Center of image in pixel/meters #
    ###################################
    centerPixelY = round(img.height / 2)
    centerPixelx = round(img.width / 2)

    ##################################################################
    # lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913 #
    ##################################################################
    [mx, my] = GoogleStaticMapsAPI.LatLonToMeters(center_lat, center_long)
    curResolution = INITIALRESOLUTION / 2 ** zoom / SCALE

    ############################
    # x and y vector in meters #
    ############################
    xVec = mx + (np.arange(img.width) - centerPixelx) * curResolution
    yVec = my + (np.arange(img.height) - centerPixelY) * curResolution

    ###################################################
    # From EPSG:900913 to WGS84 Datum lat lon vectors #
    ###################################################
    lat_north, lon_east = GoogleStaticMapsAPI.MetersToLatLon(xVec, yVec)

    ############################################################
    # plot google map image and the values from the input data #
    ############################################################
    fig, ax = plt.subplots()
    ax.set_title(title, y=1.04)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.imshow(np.flip(np.array(img),0), extent=[lon_east[0], lon_east[-1],lat_north[0], lat_north[-1]], origin='lower')

    sc = ax.scatter(
        longitudes,# Scatter plot
        latitudes,
        c=z,
        s=width / 400,
        linewidth=0,
        alpha=1,
        zorder=3
    )

    #############
    # Color Bar #
    #############
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2, aspect=15)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.set_title(cLabel)

    ######################
    # Hide google footer #
    ######################
    ax.axis([lon_east[50], lon_east[-1], lat_north[50], lat_north[-1]]) # This hides the google footer

    ###################
    # Save Fig option #
    ###################
    if saveFig:
        if os.path.exists('./Figs'):
            fig.savefig('./Figs/' + figName + '.png', format='png', dpi=600)
            fig.savefig('./Figs/' + figName + '.eps', format='eps', dpi=1000)
        else:
            os.makedirs('./Figs/')
            fig.savefig('./Figs/' + figName + '.png', format='png', dpi=600)
            fig.savefig('./Figs/' + figName + '.eps', format='eps', dpi=1000)


def scatter_with_utm(utmX, utmY, z, zone_number, zone_letter, colors=None, maptype=MAPTYPE, **kargs):
    """Scatter plot over a map. Can be used to visualize clusters by providing the marker colors.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param pandas.Series colors: marker colors, as integers
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info

    :param string title: title of the plot
    :param string xlabel: x label
    :param string ylabel: y label
    :param boolean cbar: add color bar to the graphic
    :param string cLabel: Label in color bar
    :param boolean saveFig: save Figs
    :param string figname: name of the file to save the figure

    :return: None
    """
    ###########################
    # Default optional values #
    ###########################
    title = kargs.pop('title', ' ')
    xlabel = kargs.pop('xlabel', 'East - Local Cordinates (UTM-REF) [m]')
    ylabel = kargs.pop('ylabel', 'North - Local Cordinates (UTM-REF) [m]')
    cLabel = kargs.pop('cLabel',' ')
    cbar = kargs.pop('cbar', False)
    saveFig = kargs.pop('saveFig', False)
    figName = kargs.pop('figName','googleMaps_wLines_utm')

    if kargs: raise TypeError('extra keywords: %s' % kargs)

    ############################################################################
    # Convert UTM to WGS84 because google maps api only acepts tuple (lat,lon) #
    ############################################################################
    lat = []
    lon = []

    for i in range(len(utmX)):
        latlon = pd.Series(utm.to_latlon(utmX[i], utmY[i], zone_number, zone_letter))
        lat.append(latlon[0])
        lon.append(latlon[1])


    ################################################
    # latitudes and longitudes to panda database #
    ################################################
    d_wgs = {'latitudes': lat, 'longitudes': lon}
    df_wgs = pd.DataFrame(data=d_wgs)

    latitudes = df_wgs['latitudes']
    longitudes = df_wgs['longitudes']

    #################################
    # Get google maps static image #
    ################################
    width = SCALE * MAX_SIZE
    img, pixels = background_and_pixels(latitudes, longitudes, MAX_SIZE, maptype)

    ##########################
    # get zoom normally = 16 #
    ##########################
    center_lat = (latitudes.max() + latitudes.min()) / 2
    center_long = (longitudes.max() + longitudes.min()) / 2
    zoom = GoogleStaticMapsAPI.get_zoom(latitudes, longitudes, MAX_SIZE, SCALE)

    ###################################
    # Center of image in pixel/meters #
    ###################################
    centerPixelY = round(img.height / 2)
    centerPixelX = round(img.width / 2)

    ##################################################################
    # lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913 #
    ##################################################################
    [mx, my] = GoogleStaticMapsAPI.LatLonToMeters(center_lat, center_long)
    curResolution = INITIALRESOLUTION / 2 ** zoom / SCALE

    ###################
    # x and y vector #
    ##################
    xVec = mx + (np.arange(img.width) - centerPixelX) * curResolution
    yVec = my + (np.arange(img.height) - centerPixelY) * curResolution

    #####################################
    # Convert from EPSG:900913 to WGS84 #
    #####################################
    lat_north, lon_east = GoogleStaticMapsAPI.MetersToLatLon(xVec, yVec)

    ########################
    # Convert WGS84 to UTM #
    ########################
    east = []
    north = []
    for i in range(len(lat_north)):
        eastNorth = utm.from_latlon(lat_north[i], lon_east[i], force_zone_number=zone_number)
        east.append(eastNorth[0])
        north.append(eastNorth[1])

    ############################################################
    # plot google map image and the values from the input data #
    ############################################################
    fig, ax = plt.subplots()
    ax.set_title(title, y=1.04)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.imshow(np.flip(np.array(img),0), extent=[east[0], east[-1],north[0], north[-1]], origin='lower')

    sc = ax.scatter(
        utmX,# Scatter plot
        utmY,
        c=z,
        s=width / 400,
        linewidth=0,
        alpha=1,
        zorder=3
    )

    #############
    # Color Bar #
    #############
    if cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2, aspect=15)
        cbar = fig.colorbar(sc, cax=cax)
        cbar.ax.set_title(cLabel)

    ######################
    # Hide google footer #
    ######################
    ax.axis([east[50], east[-1], north[50], north[-1]])

    ###################
    # Save Fig option #
    ###################
    if saveFig:
        if os.path.exists('./Figs'):
            fig.savefig('./Figs/' + figName + '.png', format='png', dpi=600)
            fig.savefig('./Figs/' + figName + '.eps', format='eps', dpi=1000)
        else:
            os.makedirs('./Figs/')
            fig.savefig('./Figs/' + figName + '.png', format='png', dpi=600)
            fig.savefig('./Figs/' + figName + '.eps', format='eps', dpi=1000)


def background_map_wgs84(latitudes, longitudes, maptype='satellite', **kargs):
    """Queries the proper background map and translate geo coordinated into pixel locations on this map.

    :param pandas.Series latitudes: series of sample latitudes
    :param pandas.Series longitudes: series of sample longitudes
    :param int size: target size of the map, in pixels
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info
    :param string xlabel: x label
    :param string ylabel: y label
    :param int zoom level: 14 to get extra area

    :return: fig and ax
    :rtype: (PIL.Image, pandas.DataFrame)
        """
    ###########################
    # Default optional values #
    ###########################
    xlabel = kargs.pop('xlabel', 'Longitudes')
    ylabel = kargs.pop('ylabel', 'Latitudes')

    ##########################################################
    # From lat/long to pixels, zoom and position in the tile #
    ##########################################################
    center_lat = (latitudes.max() + latitudes.min()) / 2
    center_long = (longitudes.max() + longitudes.min()) / 2
    zoom = kargs.pop('zoom', GoogleStaticMapsAPI.get_zoom(latitudes, longitudes, MAX_SIZE, SCALE)) #default zoom value

    ##############
    # Google Map #
    ##############
    img = GoogleStaticMapsAPI.map(
            center=(center_lat, center_long),
            zoom=zoom,
            scale=SCALE,
            size=(MAX_SIZE, MAX_SIZE),
            maptype=maptype,
    )

    centerPixelY = round(img.height/2)
    centerPixelx = round(img.width/2)

    ##################################################################
    # lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913 #
    ##################################################################
    [mx, my] = GoogleStaticMapsAPI.LatLonToMeters(center_lat,center_long)
    curResolution = INITIALRESOLUTION / 2 ** zoom / SCALE

    ###################
    # x and y vector #
    ##################
    xVec = mx + (np.arange(img.width)-centerPixelx) * curResolution
    yVec = my + (np.arange(img.height)-centerPixelY) * curResolution

    #####################################
    # Convert from EPSG:900913 to WGS84 #
    #####################################
    lat_north, lon_east = GoogleStaticMapsAPI.MetersToLatLon(xVec, yVec)

    ##############
    # fig output #
    ##############
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.imshow(np.flip(np.array(img), 0), extent=[lon_east[0], lon_east[-1], lat_north[0], lat_north[-1]], origin='lower')
    ax.axis([lon_east[50], lon_east[-1], lat_north[50], lat_north[-1]])

    return fig, ax

def background_map_utm(utmX, utmY, zone_number, zone_letter, maptype='satellite', **kargs):
    """Queries the proper background map and translate geo coordinated into pixel locations on this map.

    :param pandas.Series utmX: series of sample easting
    :param pandas.Series utmY: series of sample northing
    :param int zone_number: number of the zone in the globe
    :param string zone_letter: string with 'S' or 'N'
    :param string maptype: type of maps, see GoogleStaticMapsAPI docs for more info
    :param string xlabel: x label
    :param string ylabel: y label
    :param int zoom level: 14 to get extra area

    :return: fig and ax
    :rtype: (PIL.Image, pandas.DataFrame)
        """
    ############################################################################
    # Convert UTM to WGS84 because google maps api only acepts tuple (lat,lon) #
    ############################################################################
    lat = []
    lon = []

    for i in range(len(utmX)):
        latlon = pd.Series(utm.to_latlon(utmX[i], utmY[i], zone_number, zone_letter))
        lat.append(latlon[0])
        lon.append(latlon[1])

    ###########################
    # Default optional values #
    ###########################
    xlabel = kargs.pop('xlabel', 'East - Local Cordinates (UTM-REF) [m]')
    ylabel = kargs.pop('ylabel', 'North - Local Cordinates (UTM-REF) [m]')

    ################################################
    # latitudes and longitudes from panda database #
    ################################################
    d_wgs = {'latitudes': lat, 'longitudes': lon}
    df_wgs = pd.DataFrame(data=d_wgs)

    latitudes = df_wgs['latitudes']
    longitudes  = df_wgs['longitudes']

    ##########################################################
    # From lat/long to pixels, zoom and position in the tile #
    ##########################################################
    center_lat = (latitudes.max() + latitudes.min()) / 2
    center_long = (longitudes.max() + longitudes.min()) / 2
    zoom = kargs.pop('zoom', GoogleStaticMapsAPI.get_zoom(latitudes, longitudes, MAX_SIZE, SCALE)) # default zoom value

    ##############
    # Google Map #
    ##############
    img = GoogleStaticMapsAPI.map(
            center=(center_lat, center_long),
            zoom=zoom,
            scale=SCALE,
            size=(MAX_SIZE, MAX_SIZE),
            maptype=maptype,
    )

    centerPixelY = round(img.height/2)
    centerPixelx = round(img.width/2)

    ##################################################################
    # lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913 #
    ##################################################################
    [mx, my] = GoogleStaticMapsAPI.LatLonToMeters(center_lat,center_long)
    curResolution = INITIALRESOLUTION / 2 ** zoom / SCALE

    # x and y vector
    xVec = mx + (np.arange(img.width)-centerPixelx) * curResolution
    yVec = my + (np.arange(img.height)-centerPixelY) * curResolution

    ######################
    # Convert to lat lon #
    ######################
    lat_north, lon_east = GoogleStaticMapsAPI.MetersToLatLon(xVec, yVec)

    #Convert to UTM
    east = []
    north = []
    for i in range(len(lat_north)):
        eastNorth = utm.from_latlon(lat_north[i],lon_east[i],force_zone_number=zone_number)
        east.append(eastNorth[0])
        north.append(eastNorth[1])

    #################
    # fig,ax output #
    #################
    fig, ax = plt.subplots()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.imshow(np.flip(np.array(img), 0), extent=[east[0], east[-1], north[0], north[-1]], origin='lower')
    ax.axis([east[50], east[-1], north[50], north[-1]])

    return fig, ax


