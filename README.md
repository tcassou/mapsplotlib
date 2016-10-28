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

gmaps.plot_markers()
```
will produce

![Marker Plot](http://imgur.com/a/2pdbG)

### Density Plots

The only thing you need is a pandas DataFrame `df` containing a `latitude` and a `longitude` columns, describing locations.

```
import gmaps

gmaps.density_plot(df['latitude'], df['longitude'])
```

![Density Plot](http://imgur.com/a/OPEw7)

### Heat Maps

This time your pandas DataFrame `df` will need an extra `value` column, describing the metric you want to plot (you may have to normalize it properly for a good rendering).

```
import gmaps

gmaps.heatmap(df['latitude'], df['longitude'], df['value'])
```
![Heat Map](http://imgur.com/a/qHj0j)

### More to come!

## Requirements

* `pandas >= 0.13.1`
* `numpy >= 1.8.2`
* `scipy >= 0.13.3`
* `matplotlib >= 1.3.1`
* `requests >= 2.7.0`
