import numpy as np
from sklearn.neural_network import MLPClassifier
from image import Image

class Aster(Image):
    def __init__(self, mnt, qa):
        super().__init__(mnt)
        self.mnt = mnt
        self.qa = qa
        self.resolution = 30

        #self.save_band(self.pente, 'pente.tif')


    def getPente(self):
        mnt = self.getArray(self.mnt)

        shape = mnt.shape
        pente = np.empty(shape, dtype=float)
        
        for i in range(shape[0] - 1):
            for j in range(shape[1] - 1):
                if i == 0 or j == 0:
                    pente[i, j] = 0
                else:
                    za = mnt[i - 1, j - 1]
                    zb = mnt[i - 1, j]
                    zc = mnt[i - 1, j + 1]
                    zd = mnt[i, j - 1]
                    ze = mnt[i, j]
                    zf = mnt[i, j + 1]
                    zg = mnt[i + 1, j - 1]
                    zh = mnt[i + 1, j]
                    zi = mnt[i + 1, j + 1]

                    dzdx = ((zc + 2*zf + zi) - (za + 2*zd + zg)) / (8 * self.resolution)
                    dzdy = ((zg + 2*zh + zi) - (za + 2*zb + zc)) / (8 * self.resolution)

                    pente[i, j] = (dzdx**2 + dzdy**2)**(1/2)
        
        return pente

    def reprojectAster(self, referenceFile):
        bandsPaths = [self.mnt, self.qa]
        newBandsPaths = []

        for band in bandsPaths:
            image = Image(band)
            newBandsPaths.append(image.reprojectImage(referenceFile))

        self.mnt = newBandsPaths[0]
        self.qa = newBandsPaths[1]

        print("          Reprojection termine")

def main():
    b1 = r'aster/aster_petit.tif'
    qa = r'aster/aster_petit_qa.tif'
    aster = Aster(b1, qa)




if __name__ == '__main__':
    main()
