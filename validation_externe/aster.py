import numpy as np
from image import Image


class Aster:
    """ Classe modélisant une collection d'images ASTER. Plus précisément, elle modélise la collection ASTER GDEM
        (soit un modèle numérique de terrain global). Cette classe est composée d'objets de type Image, mais ne les
        appelle pas dans son constructeur afin d'économiser de l'espace mémoire et minimiser le temps d'exécution.

        Attributes:
            mnt (str): Path vers le fichier .tiff du produit de modèle numérique de terrain de la collection d'images
                       ASTER.
            qa (str): Path vers le fichier .tiff de la bande de qualité de la collection d'images ASTER (fichier NUM).
            resolution (int): Résolution spatiale des images (en mètres). Cet attribut permet de calculer le produit
                              dérivé de la pente du modèle numérique de terrain (MNT).
    """
    def __init__(self, mnt, qa):
        self.mnt = mnt
        self.qa = qa
        self.resolution = 30

    def getMNT(self):
        """ Permet de récupérer un array Numpy du modèle numérique de terrain. Les valeurs à l'extérieur de
            l'intervalle valide sont masquées.
                Returns:
                    mnt_array (Numpy.ma.masked_array): Array des valeurs de modèle numérique de terrain (MNT).
        """
        mnt_image = Image(self.mnt)

        # 0 = niveau moyen des mers
        mnt_array = mnt_image.getArray(masked=True, lower_valid_range=-500, upper_valid_range=9000)

        return mnt_array

    def getPente(self):
        """ Permet de récupérer un array Numpy de la pente (produit dérivé du modèle numérique de terrain).
                Returns:
                    pente (Numpy.ma.masked_array): Array des valeurs de pente du modèle numérique de terrain (MNT).
        """
        mnt = self.getMNT()

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

    def reprojectAster(self, referenceFile, reduce_zone=True):
        """ Permet de reprojeter, découper, aligner et rééchantillonner une image à partir d'une image de référence.
            Cette méthode effectue ces traitements pour l'ensemble des images de la collection ASTER. Elle fait
            appel à la méthode reprojectMatch() de la classe Image.
                Args:
                    referenceFile (string): Path du fichier de l'image de référence à utiliser.
                    reduce_zone (bool): Indicateur permettant de choisir si on souhaite réduire la zone d'étude
                                        sur laquelle les images sont "matchées". Ceci est utile pour éviter des
                                        problèmes avec des valeurs nulles sur les bords des images qui s'alignent
                                        sur le referenceFile. Par défaut, cette option est égale à True (donc, on
                                        effectue le rétrécissement de zone).
        """
        bandsPaths = [self.mnt, self.qa]
        newBandsPaths = []

        for band in bandsPaths:
            image = Image(band)
            newBandsPaths.append(image.reprojectMatch(referenceFile, reduce_zone))

        self.mnt = newBandsPaths[0]
        self.qa = newBandsPaths[1]

        print("ASTER:            Reprojection termine")


def main():
    """ Tests de la classe et de ses méthodes.
    """
    b1 = r'aster/aster_petit.tif'
    qa = r'aster/aster_petit_qa.tif'
    aster = Aster(b1, qa)


if __name__ == '__main__':
    main()
