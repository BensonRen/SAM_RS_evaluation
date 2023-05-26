import numpy as np
import matplotlib.pyplot as plt
import cv2
import shapely.wkt
import pandas as pd
from shapely.geometry.polygon import Polygon
import rasterio.features

##################################################################################################
# Below code is the first data exploration attempt to understand this dataset and how to use it #
##################################################################################################
img = cv2.imread('SN6_Train_AOI_11_Rotterdam_PS-RGB_20190822070610_20190822070846_tile_3721.tif')
building_info = pd.read_csv('test_small_snippet.csv')

f = plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
for i in range(len(building_info)):
    plg = shapely.wkt.loads(building_info['PolygonWKT_Pix'][i])
    x, y = plg.exterior.xy
    plt.plot(x, y)
    mask = rasterio.features.rasterize([plg], out_shape=np.shape(img)[:2])
#     plt.imshow(img)

