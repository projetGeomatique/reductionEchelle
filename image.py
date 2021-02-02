import gdal
import numpy as np

class Image():
    def __init__(self, filename):
        self.xsize = gdal.Open(filename).RasterXSize
        self.ysize = gdal.Open(filename).RasterYSize
        self.proj = gdal.Open(filename).GetProjection()
        self.gt = gdal.Open(filename).GetGeoTransform()


    def getArray(filename):
        ds = gdal.Open(filename)
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray()
        return array


    def save_band(self, array, filename):
        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        outds = driver.Create(filename, xsize=self.xsize, ysize=self.ysize, bands=1, eType=gdal.GDT_Float32)
        outds.SetGeoTransform(self.gt)
        outds.SetProjection(self.proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(array)
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()
        outband = None
        outds = None


def main():
    b1 = r'data/ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif'
    aster = Image(b1)

    print(aster.gt)


if __name__ == '__main__':
    main()