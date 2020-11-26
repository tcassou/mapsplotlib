# Mapsplotlib

[![Build](https://github.com/tcassou/mapsplotlib/workflows/Test%20and%20Release/badge.svg)](https://github.com/tcassou/mapsplotlib/actions)

Custom Python plots on a Google Maps background. A flexible matplotlib like interface to generate many types of plots on top of Google Maps.

This package was renamed from the legacy `tcassou/gmaps` due to an unfortunate conflict in names with a package from Pypi.

## Setup

Simply install from `pip`:
```
pip install mapsplotlib
```

You need to have a Google Static Maps API key, go to https://console.cloud.google.com/google/maps-apis, create a project, enable Google Static Maps API and get your API key. Billing details have to be enabled for your account for the API calls to succeed.
Before plotting maps, you'll have to register your key (only once for each session you start):
```
from mapsplotlib import mapsplot as mplt

mplt.register_api_key('your_google_api_key_here')

# all plots can now be performed here
```

## Examples

### Marker Plots

Simply plotting markers on a map. Consider a pandas DataFrame `df` defined as follows:

```
|   | latitude | longitude |  color |  size | label |
|---|----------|-----------|--------|-------|-------|
| 0 |  48.8770 |  2.30698  |  blue  |  tiny |       |
| 1 |  48.8708 |  2.30523  |   red  | small |       |
| 2 |  48.8733 |  2.32403  | orange |  mid  |   A   |
| 3 |  48.8728 |  2.30491  |  black |  mid  |   Z   |
| 4 |  48.8644 |  2.33160  | purple |  mid  |   0   |
```

Simply use (assuming `mapsplot` was imported already, and your key registered)
```
mplt.plot_markers(df)
```
will produce

![Marker Plot](https://github.com/tcassou/mapsplotlib/blob/master/examples/markers.png)

### Density Plots

The only thing you need is a pandas DataFrame `df` containing a `'latitude'` and a `'longitude'` columns, describing locations.

```
mplt.density_plot(df['latitude'], df['longitude'])
```

![Density Plot](https://github.com/tcassou/mapsplotlib/blob/master/examples/density.png)

### Heat Maps

This time your pandas DataFrame `df` will need an extra `'value'` column, describing the metric you want to plot (you may have to normalize it properly for a good rendering).

```
mplt.heatmap(df['latitude'], df['longitude'], df['value'])
```
![Heat Map](https://github.com/tcassou/mapsplotlib/blob/master/examples/heatmap.png)

### Scatter Plots

Let's assume your pandas DataFrame `df` has a numerical `'cluster'` column, describing clusters of geographical points. You can produce plots like the following:

```
mplt.scatter(df['latitude'], df['longitude'], colors=df['cluster'])
```
![Scatter Plot](https://github.com/tcassou/mapsplotlib/blob/master/examples/clusters.png)

### Polygon Plots

Still with the same DataFrame `df` and its `'cluster'` column, plotting clusters and their convex hull.

```
mplt.polygons(df['latitude'], df['longitude'], df['cluster'])
```
![Polygons Plot](https://github.com/tcassou/mapsplotlib/blob/master/examples/polygons.png)

### Polygon Plots

Given a DataFrame `df` with `'latitude'` & `'longitude'` columns, plotting a line joining all `(lat, lon)` pairs (with the option to close the line).

```
mplt.polyline(df['latitude'], df['longitude'], closed=True)
```
![Polyline Plot](https://github.com/tcassou/mapsplotlib/blob/master/examples/polyline.png)

### More to come!

## Requirements

* `pandas >= 0.13.1`
* `numpy >= 1.8.2`
* `scipy >= 0.13.3`
* `matplotlib >= 1.3.1`
* `requests >= 2.18.4`
* `pillow >= 4.3.0`
