import numpy as np
from sklearn.neural_network import MLPClassifier
from image import Image




class Landsat(Image):
    def __init__(self, b1, b2, b3, b4, b5, b6, b7, qa):
        super().__init__(b1)
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b5 = b5
        self.b6 = b6
        self.b7 = b7
        self.qa = qa


    def getNdvi(self):
        # ajout du where pour éviter la division par 0 (à voir quel résultat est obtenu dans ce cas...)
        b4 = self.getArray(self.b4)
        b5 = self.getArray(self.b5)

        return np.divide(np.subtract(b5, b4), np.add(b5, b4), where=((np.add(b5, b4)) != 0))


    def getNdbi(self):
        b5 = self.getArray(self.b5)
        b6 = self.getArray(self.b6)

        return np.divide(np.subtract(b6, b5), np.add(b6, b5), where=((np.add(b6, b5)) != 0))


    def getNdwi(self):
        b5 = self.getArray(self.b5)
        b6 = self.getArray(self.b6)

        return np.divide(np.subtract(b5, b6), np.add(b5, b6), where=((np.add(b5, b6)) != 0))


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
        

    def reprojectLandsat(self, referenceFile):
        bandsPaths = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.qa]
        newBandsPaths = []

        for band in bandsPaths:
            image = Image(band)
            newBandsPaths.append(image.reprojectImage(referenceFile))

        self.b1 = newBandsPaths[0]
        self.b2 = newBandsPaths[1]
        self.b3 = newBandsPaths[2]
        self.b4 = newBandsPaths[3]
        self.b5 = newBandsPaths[4]
        self.b6 = newBandsPaths[5]
        self.b7 = newBandsPaths[6]
        self.qa = newBandsPaths[7]

        print("          Reprojection termine")



def main():
    b1 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB1_doy2020229_aid0001.tif'
    b2 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB2_doy2020229_aid0001.tif'
    b3 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB3_doy2020229_aid0001.tif'
    b4 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB4_doy2020229_aid0001.tif'
    b5 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB5_doy2020229_aid0001.tif'
    b6 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB6_doy2020229_aid0001.tif'
    b7 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB7_doy2020229_aid0001.tif'
    qa = r'fonctionReprojectAjoutees/data/CU_LC08.001_PIXELQA_doy2020229_aid0001.tif'
    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa)

    landsat.reprojectLandsat(r"data/MOD11A1.006_Clear_day_cov_doy2020229_aid0001.tif")

    


if __name__ == '__main__':
    main()