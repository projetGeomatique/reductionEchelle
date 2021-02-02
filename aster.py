import numpy as np
from sklearn.neural_network import MLPClassifier
from image import Image

class Aster(Image):
    def __init__(self, b1, qa):
        super().__init__(b1)
        self.mnt = Image.getArray(b1)
        self.qa = Image.getArray(qa)
        self.resolution = 30
        self.pente = self.getPente()

        self.save_band(self.pente, 'pente.tif')

        print(self.mnt)
        print(self.proj)


    def getPente(self):
        shape = self.mnt.shape
        pente = np.empty(shape, dtype=float)
        
        for i in range(self.ysize - 1):
            for j in range(self.xsize - 1):
                if i == 0 or j == 0:
                    pente[i, j] = 0
                else:
                    za = self.mnt[i - 1, j - 1]
                    zb = self.mnt[i - 1, j]
                    zc = self.mnt[i - 1, j + 1]
                    zd = self.mnt[i, j - 1]
                    ze = self.mnt[i, j]
                    zf = self.mnt[i, j + 1]
                    zg = self.mnt[i + 1, j - 1]
                    zh = self.mnt[i + 1, j]
                    zi = self.mnt[i + 1, j + 1]

                    dzdx = ((zc + 2*zf + zi) - (za + 2*zd + zg)) / (8 * self.resolution)
                    dzdy = ((zg + 2*zh + zi) - (za + 2*zb + zc)) / (8 * self.resolution)

                    pente[i, j] = (dzdx**2 + dzdy**2)**(1/2)
        
        return pente

def main():
    b1 = r'aster/aster_petit.tif'
    qa = r'aster/aster_petit_qa.tif'
    aster = Aster(b1, qa)




if __name__ == '__main__':
    main()
