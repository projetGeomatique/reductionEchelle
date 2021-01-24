import os
import re  # regular expressions
import warnings
import numpy as np
import rasterio as rio


print("Test")

warnings.simplefilter('ignore')


link = "C:/Users/le_lo/Downloads/MOD11A1.A2015118.h12v04.006.2016218134449.hdf"

class HdfData:
    def __init__(self, link):
        self.link = link
        self.data = []
        self.dataNp = None
        self.meta = None

    def getData(self):
        return self.data

    def initialiseCouche(self):
        # Open the pre-fire HDF4 file
        with rio.open(self.link) as dataset:
            self.meta = dataset.meta
            # Loop through each subdataset in HDF4 file
            for nom in dataset.subdatasets:

                    # Open the band subdataset
                    with rio.open(nom) as subdataset:
                        # Read band data as a 2 dim arr and append to list
                        self.data.append(subdataset.read(1))

        self.dataNp = np.stack(self.data)


couche1 = HdfData(link)
couche1.initialiseCouche()
print(couche1.dataNp[0][0][0])
print(couche1.data[0][0][0])
print(couche1.dataNp.shape)
print(couche1.meta)
