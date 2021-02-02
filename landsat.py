import numpy as np
from sklearn.neural_network import MLPClassifier
from image import Image

class Landsat(Image):
    def __init__(self, b1, b2, b3, b4, b5, b6, b7, qa):
        super().__init__(b1)
        self.b1 = Image.getArray(b1)
        self.b2 = Image.getArray(b2)
        self.b3 = Image.getArray(b3)
        self.b4 = Image.getArray(b4)
        self.b5 = Image.getArray(b5)
        self.b6 = Image.getArray(b6)
        self.b7 = Image.getArray(b7)
        self.qa = Image.getArray(qa)
        self.cloud = self.getCloud()
        self.ndvi = self.getNdvi()
        self.ndbi = self.getNdbi()
        self.ndwi = self.getNdwi()
        self.predicteurs = self.getPredicteurs()

    def getNdvi(self):
        return np.divide(np.subtract(self.b5, self.b4), np.add(self.b5, self.b4))

    def getNdbi(self):
        return np.divide(np.subtract(self.b6, self.b5), np.add(self.b6, self.b5))

    def getNdwi(self):
        return np.divide(np.subtract(self.b5, self.b6), np.add(self.b5, self.b6))

    def getCloud(self):
        shape = self.qa.shape
        cloud = np.empty(shape, dtype=float)

        for i in range(self.ysize):
            for j in range(self.xsize):
                if self.qa[i][j] == 480:
                    cloud[i][j] = 0
                else:
                    cloud[i][j] = 1
        
        return cloud
        

    def getPredicteurs(self):
        predicteurs = np.zeros((145548, 4))
        i = -1

        for b in [self.b1, self.b2, self.b3, self.b4]:
            i = i + 1
            x = b.flatten()
            n = x.size
            x.resize(n, 1)

            predicteurs[:, i:i+1] = x

        return predicteurs

    def neural_net(self):
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(4, 1), random_state=1)

        X = self.predicteurs
        y = self.cloud.flatten()
        y.resize(y.size, 1)

        clf.fit(X, y.ravel())

        return clf

    def setNeuralNet(self, clf):
        self.clf = clf

    def predictNeuralNet(self):
        predicteurs = np.nan_to_num(self.predicteurs)
        cloud = self.clf.predict(predicteurs)
        cloud.resize(self.ysize, self.xsize)
        self.save_band(cloud, 'cloud.tif')


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

    #landsat.predicteurs()
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