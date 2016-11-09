# GMaps

Custom Python plots on a Google Maps background.

## Installation

You'll need to have a Google Static Maps API key, go to https://console.developers.google.com, create a project, enable Google Static Maps API, get your server key and paste it in `google_static_maps_api.py`.

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

Simply use
```
import gmaps

gmaps.plot_markers(df)
```
will produce

![Marker Plot](https://github.com/thomasCassou/gmaps/blob/master/examples/markers.png)

### Density Plots

The only thing you need is a pandas DataFrame `df` containing a `'latitude'` and a `'longitude'` columns, describing locations.

```
import gmaps

gmaps.density_plot(df['latitude'], df['longitude'])
```

![Density Plot](https://github.com/thomasCassou/gmaps/blob/master/examples/density.png)

### Heat Maps

This time your pandas DataFrame `df` will need an extra `'value'` column, describing the metric you want to plot (you may have to normalize it properly for a good rendering).

```
import gmaps

gmaps.heatmap(df['latitude'], df['longitude'], df['value'])
```
![Heat Map](https://github.com/thomasCassou/gmaps/blob/master/examples/heatmap.png)

### Scatter Plots

Let's assume your pandas DataFrame `df` has a numerical `'cluster'` column, describing clusters of geographical points. You can produce plots like the following:

```
import gmaps

gmaps.scatter(df['latitude'], df['longitude'], colors=df['cluster'])
```
![Heat Map](https://github.com/thomasCassou/gmaps/blob/master/examples/clusters.png)

### Polygon Plots

Still with the same DataFrame `df` and its `'cluster'` column, plotting clusters and their convex hull.

```
import gmaps

gmaps.polygons(df['latitude'], df['longitude'], df['cluster'])
```
![Heat Map](https://github.com/thomasCassou/gmaps/blob/master/examples/polygons.png)

### More to come!

## Requirements

* `pandas >= 0.13.1`
* `numpy >= 1.8.2`
* `scipy >= 0.13.3`
* `matplotlib >= 1.3.1`
* `requests >= 2.7.0`
