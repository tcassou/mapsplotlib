from mapsplotlib import mapsplot as mplt
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

# UTM background google map and plot some values after
# Same data used in the previous example

fig, ax = mplt.background_map_utm(df_utm['east'], df_utm['north'], 29, 'N', zoom = 14)
ax.plot(east_survey,north_survey,'b-',label='Map Area', zorder=2)
plt.legend(loc='best', shadow=True, fancybox=True)
plt.title('Some Survey ', y=1.04)

fig.savefig('./examples/utm_background.png', format='png', dpi=600)


plt.show()