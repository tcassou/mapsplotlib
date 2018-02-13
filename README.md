# Mapsplotlib

Custom Python plots on a Google Maps background. A flexible matplotlib like interface to generate many types of plots on top of Google Maps.

This package was renamed from the legacy `tcassou/gmaps` due to an unfortunate conflict in names with a package from Pypi.

## Setup

Simply install from `pip`:
```
pip install mapsplotlib
```

You'll then need to have a Google Static Maps API key, go to https://console.developers.google.com, create a project, enable Google Static Maps API and get your server key. Before plotting maps, you'll have to register your key (only once for each session you start):
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
![Heat Map](https://github.com/tcassou/mapsplotlib/blob/master/examples/clusters.png)

### Polygon Plots

Still with the same DataFrame `df` and its `'cluster'` column, plotting clusters and their convex hull.

```
mplt.polygons(df['latitude'], df['longitude'], df['cluster'])
```
![Heat Map](https://github.com/tcassou/mapsplotlib/blob/master/examples/polygons.png)


### UTM scatter plot
Put some utm coordinates in a pandas database, ex: 
```
from mapsplotlib import mapsplot as mplt
import matplotlib.pyplot as plt
from scipy.io import loadmat

mplt.register_api_key('xxx') # xxx = your Google api key

fileLines_survey = 'lines_survey'
survey_data = loadmat('Lines/' + fileLines_survey + '.mat')

east_survey = survey_data['lines_survey'][:, 0]
north_survey = survey_data['lines_survey'][:, 1]
z_survey = survey_data['lines_survey'][:, 2]

eastn = east_survey[~np.isnan(east_survey)]
northn = north_survey[~np.isnan(north_survey)]
zn = z_survey[~np.isnan(z_survey)]

d_utm = {'east': eastn, 'north': northn, 'value': zn}
df_utm = pd.DataFrame(data=d_utm)

mplt.scatter_with_utm(df_utm['east'],df_utm['north'],df_utm['value'],29,'N',maptype='satellite', cbar=True,
                      title='Some Survey', cLabel='Val', saveFig=True)
```
![UTM Scatter](https://github.com/tcassou/mapsplotlib/blob/master/examples/utm_scatter.png)


### UTM background google map and plot some values after
Same data used in the previous example
```
fig, ax = mplt.background_map_utm(df_utm['east'], df_utm['north'], 29, 'N', zoom = 14)
ax.plot(east_survey,north_survey,'b-',label='Map Area', zorder=2)
plt.legend(loc='best', shadow=True, fancybox=True)
plt.title('Some Survey ', y=1.04)

plt.show()
```
![UTM Background](https://github.com/tcassou/mapsplotlib/blob/master/examples/utm_background.png)

### WGS84 Lat/Lon scatter plot WGS84 background
Same as previous examples, just input Latitudes and Longitudes values


### More to come!

## Requirements

* `pandas >= 0.13.1`
* `numpy >= 1.8.2`
* `scipy >= 0.13.3`
* `matplotlib >= 1.3.1`
* `requests >= 2.7.0`
* `requests>=2.18.4`
* `pillow>=4.3.0`
* `utm>=0.4.2`


