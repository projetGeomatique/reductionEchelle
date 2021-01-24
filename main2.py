import gdal
import numpy as np
from sklearn.neural_network import MLPClassifier

class Landsat():
    def __init__(self, b1, b2, b3, b4, b5, b6, b7, qa):
        self.b1 = self.getArray(b1)
        self.b2 = self.getArray(b2)
        self.b3 = self.getArray(b3)
        self.b4 = self.getArray(b4)
        self.b5 = self.getArray(b5)
        self.b6 = self.getArray(b6)
        self.b7 = self.getArray(b7)
        self.qa = self.getArray(qa)
        self.cloud = self.getCloud()
        self.ndvi = self.getNdvi()
        self.ndbi = self.getNdbi()
        self.ndwi = self.getNdwi()

        self.gt = self.getGt(b1)
        self.proj = self.getProj(b1)
        self.xsize = self.ndvi.shape[1]
        self.ysize = self.ndvi.shape[0]
        self.predicteurs()

    def getArray(self, filename):
        ds = gdal.Open(filename)
        band = ds.GetRasterBand(1)
        array = band.ReadAsArray()
        return array

    def getGt(self, filename):
        ds = gdal.Open(filename)
        gt = ds.GetGeoTransform()
        return gt

    def getProj(self, filename):
        ds = gdal.Open(filename)
        proj = ds.GetProjection()
        return proj

    def getNdvi(self):
        ndvi = np.divide(np.subtract(self.b5, self.b4), np.add(self.b5, self.b4))
        return ndvi

    def getNdbi(self):
        ndbi = np.divide(np.subtract(self.b6, self.b5), np.add(self.b6, self.b5))
        return ndbi

    def getNdwi(self):
        ndwi = np.divide(np.subtract(self.b5, self.b6), np.add(self.b5, self.b6))
        return ndwi

    def getCloud(self):
        shape = self.qa.shape
        cloud = np.empty(shape, dtype=float)

        for i in range(shape[0]):
            for j in range(shape[1]):
                if self.qa[i][j] == 480:
                    cloud[i][j] = 0
                else:
                    cloud[i][j] = 1
        
        return cloud
        

    def showArray(self):
        print(self.ndwi)

    def save_band(self, band):
        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        outds = driver.Create('cloud_neural_net.tif', xsize=self.xsize, ysize=self.ysize, bands=1, eType=gdal.GDT_Float32)
        outds.SetGeoTransform(self.gt)
        outds.SetProjection(self.proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(band)
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()
        outband = None
        outds = None

    def predicteurs(self):
        self.pred = np.zeros((145548, 4))
        i = -1

        for b in [self.b1, self.b2, self.b3, self.b4]:
            i = i + 1
            x = b.flatten()
            n = x.size
            x.resize(n, 1)

            self.pred[:, i:i+1] = x

    def neural_net(self):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 1), random_state=1)

        X = self.pred
        y = self.cloud.flatten()
        y.resize(y.size, 1)

        clf.fit(X, y.ravel())

        return clf

    def setNeuralNet(self, clf):
        self.clf = clf

    def predictNeuralNet(self):
        pred = np.nan_to_num(self.pred)
        print(pred)
        cloud = self.clf.predict(pred)
        print(cloud)
        cloud.resize(self.ysize, self.xsize)
        print(cloud)
        self.save_band(cloud)




def main():
    b1 = r'data/CU_LC08.001_SRB1_doy2020229_aid0001.tif'
    b2 = r'data/CU_LC08.001_SRB2_doy2020229_aid0001.tif'
    b3 = r'data/CU_LC08.001_SRB3_doy2020229_aid0001.tif'
    b4 = r'data/CU_LC08.001_SRB4_doy2020229_aid0001.tif'
    b5 = r'data/CU_LC08.001_SRB5_doy2020229_aid0001.tif'
    b6 = r'data/CU_LC08.001_SRB6_doy2020229_aid0001.tif'
    b7 = r'data/CU_LC08.001_SRB7_doy2020229_aid0001.tif'
    qa = r'data/CU_LC08.001_PIXELQA_doy2020229_aid0001.tif'
    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa)

    #landsat.showArray()
    landsat.predicteurs()
    clf = landsat.neural_net()

    b1 = r'data/CU_LC08.001_SRB1_doy2020213_aid0001.tif'
    b2 = r'data/CU_LC08.001_SRB2_doy2020213_aid0001.tif'
    b3 = r'data/CU_LC08.001_SRB3_doy2020213_aid0001.tif'
    b4 = r'data/CU_LC08.001_SRB4_doy2020213_aid0001.tif'
    b5 = r'data/CU_LC08.001_SRB5_doy2020213_aid0001.tif'
    b6 = r'data/CU_LC08.001_SRB6_doy2020213_aid0001.tif'
    b7 = r'data/CU_LC08.001_SRB7_doy2020213_aid0001.tif'
    qa = r'data/CU_LC08.001_SRB7_doy2020213_aid0001.tif'
    landsat2 = Landsat(b1, b2, b3, b4, b5, b6, b7, qa)
    landsat2.setNeuralNet(clf)
    landsat2.predictNeuralNet()





if __name__ == '__main__':
    main()