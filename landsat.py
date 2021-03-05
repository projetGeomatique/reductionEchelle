import numpy as np
from image import Image
from osgeo import gdal, gdal_array, gdalconst, osr


class Landsat:
    """ Classe modélisant une collection d'images Landsat 8.
        Cette classe est composée d'objets de type Image, mais ne les appelle pas dans son constructeur afin d'économiser
        de l'espace mémoire et minimiser le temps d'exécution.

        Attributes:
            b1 (str): Path vers le fichier .tiff de la bande 1 de la collection d'images Landsat 8
            b2 (str): Path vers le fichier .tiff de la bande 2 de la collection d'images Landsat 8
            b3 (str): Path vers le fichier .tiff de la bande 3 de la collection d'images Landsat 8
            b4 (str): Path vers le fichier .tiff de la bande 4 de la collection d'images Landsat 8
            b5 (str): Path vers le fichier .tiff de la bande 5 de la collection d'images Landsat 8
            b6 (str): Path vers le fichier .tiff de la bande 6 de la collection d'images Landsat 8
            b7 (str): Path vers le fichier .tiff de la bande 7 de la collection d'images Landsat 8
            qa (str): Path vers le fichier .tiff de la bande de qualité de la collection d'images Landsat 8
    """
    def __init__(self, b1, b2, b3, b4, b5, b6, b7, qa):
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b5 = b5
        self.b6 = b6
        self.b7 = b7
        self.qa = qa

    def setBands(self, b1, b2, b3, b4, b5, b6, b7, qa):
        """ Méthode permettant d'assigner des nouveaux fichiers aux bandes de la collection.
                Args:
                    b1 (str): Nouveau path vers le fichier .tiff de la bande 1 de la collection d'images Landsat 8
                    b2 (str): Nouveau path vers le fichier .tiff de la bande 2 de la collection d'images Landsat 8
                    b3 (str): Nouveau path vers le fichier .tiff de la bande 3 de la collection d'images Landsat 8
                    b4 (str): Nouveau path vers le fichier .tiff de la bande 4 de la collection d'images Landsat 8
                    b5 (str): Nouveau path vers le fichier .tiff de la bande 5 de la collection d'images Landsat 8
                    b6 (str): Nouveau path vers le fichier .tiff de la bande 6 de la collection d'images Landsat 8
                    b7 (str): Nouveau path vers le fichier .tiff de la bande 7 de la collection d'images Landsat 8
                    qa (str): Nouveau path vers le fichier .tiff de la bande de qualité de la collection d'images Landsat 8
        """
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3
        self.b4 = b4
        self.b5 = b5
        self.b6 = b6
        self.b7 = b7
        self.qa = qa

    def getNdvi(self):
        """ Permet de récupérer un array Numpy contenant l'indice de végétation NDVI calculé à partir des bandes b4 et b5
            de la collection d'images Landsat 8. La correction TOA des niveaux de gris de l'image est faite au préalable
            et les valeurs à l'extérieur de l'intervalle valide sont masquées.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de NDVI pour la collection d'images.
        """
        # ajout du where pour éviter la division par 0 (à voir quel résultat est obtenu dans ce cas...)
        b4_img = Image(self.b4)
        b4_meta = b4_img.getMetadata()
        b4_DN = b4_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b4_meta:
            b4 = np.add(np.multiply(b4_DN, float(b4_meta['scale_factor'])), float(b4_meta['add_offset']))  # conversion en réflectances
        else:
            b4 = np.add(np.multiply(b4_DN, float(0.0001)), float(0))

        b5_img = Image(self.b5)
        b5_meta = b5_img.getMetadata()
        b5_DN = b5_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b5_meta:
            b5 = np.add(np.multiply(b5_DN, float(b5_meta['scale_factor'])), float(b5_meta['add_offset']))
        else:
            b5 = np.add(np.multiply(b5_DN, float(0.0001)), float(0))

        return np.divide(np.subtract(b5, b4), np.add(b5, b4), where=((np.add(b5, b4)) != 0))

    def getNdbi(self):
        """ Permet de récupérer un array Numpy contenant l'indice de bâti NDBI calculé à partir des bandes b5 et b6
            de la collection d'images Landsat 8. La correction TOA des niveaux de gris de l'image est faite au préalable
            et les valeurs à l'extérieur de l'intervalle valide sont masquées.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de NDBI pour la collection d'images.
        """
        b5_img = Image(self.b5)
        b5_meta = b5_img.getMetadata()
        b5_DN = b5_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b5_meta:
            b5 = np.add(np.multiply(b5_DN, float(b5_meta['scale_factor'])), float(b5_meta['add_offset']))
        else:
            b5 = np.add(np.multiply(b5_DN, float(0.0001)), float(0))

        b6_img = Image(self.b6)
        b6_meta = b6_img.getMetadata()
        b6_DN = b6_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b6_meta:
            b6 = np.add(np.multiply(b6_DN, float(b6_meta['scale_factor'])), float(b6_meta['add_offset']))
        else:
            b6 = np.add(np.multiply(b6_DN, float(0.0001)), float(0))

        return np.divide(np.subtract(b6, b5), np.add(b6, b5), where=((np.add(b6, b5)) != 0))

    def getNdwi(self):
        """ Permet de récupérer un array Numpy contenant l'indice d'humidité NDWI calculé à partir des bandes b5 et b6
            de la collection d'images Landsat 8. La correction TOA des niveaux de gris de l'image est faite au préalable
            et les valeurs à l'extérieur de l'intervalle valide sont masquées.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de NDWI pour la collection d'images.
        """
        b5_img = Image(self.b5)
        b5_meta = b5_img.getMetadata()
        b5_DN = b5_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b5_meta:
            b5 = np.add(np.multiply(b5_DN, float(b5_meta['scale_factor'])), float(b5_meta['add_offset']))
        else:
            b5 = np.add(np.multiply(b5_DN, float(0.0001)), float(0))

        b6_img = Image(self.b6)
        b6_meta = b6_img.getMetadata()
        b6_DN = b6_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b6_meta:
            b6 = np.add(np.multiply(b6_DN, float(b6_meta['scale_factor'])), float(b6_meta['add_offset']))
        else:
            b6 = np.add(np.multiply(b6_DN, float(0.0001)), float(0))

        return np.divide(np.subtract(b5, b6), np.add(b5, b6), where=((np.add(b5, b6)) != 0))


    """
    def getClouds(self):
        shape = self.qa.shape
        cloud = np.empty(shape, dtype=float)

        for i in range(self.ysize):
            for j in range(self.xsize):
                if self.qa[i][j] == 480:
                    cloud[i][j] = 0
                else:
                    cloud[i][j] = 1
        
        return cloud
    """

    def reprojectLandsat(self, referenceFile):
        """ Permet de reprojeter, découper, aligner et rééchantillonner une image à partir d'une image de référence.
            Cette méthode effectue ces traitements pour l'ensemble des bandes de la collection Landsat 8. Elle fait
            appel à la méthode reprojectMatch() de la classe Image.
                Args:
                    referenceFile (string): Path du fichier de l'image de référence à utiliser.
        """
        bandsPaths = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.qa]
        newBandsPaths = []

        for band in bandsPaths:
            image = Image(band)
            newBandsPaths.append(image.reprojectMatch(referenceFile))

        # assigner des nouveaux path aux bandes
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
    """ Tests de la classe et de ses méthodes.
    """
    b1 = r'data/CU_LC08.001_SRB1_doy2020229_aid0001.tif'
    b2 = r'data/CU_LC08.001_SRB2_doy2020229_aid0001.tif'
    b3 = r'data/CU_LC08.001_SRB3_doy2020229_aid0001.tif'
    b4 = r'data/CU_LC08.001_SRB4_doy2020229_aid0001.tif'
    b5 = r'data/CU_LC08.001_SRB5_doy2020229_aid0001.tif'
    b6 = r'data/CU_LC08.001_SRB6_doy2020229_aid0001.tif'
    b7 = r'data/CU_LC08.001_SRB7_doy2020229_aid0001.tif'
    qa = r'data/CU_LC08.001_PIXELQA_doy2020229_aid0001.tif'

    """dataset = gdal.Open(b5)
    band = dataset.GetRasterBand(1)
    print(band.GetNoDataValue())

    band = None
    dataset = None"""


    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa)

    landsat.reprojectLandsat(r"data/MOD11A1.006_Clear_day_cov_doy2020229_aid0001.tif")


    b1 = r'data/CU_LC08.001_SRB1_doy2020229_aid0001_reproject.tif'
    b2 = r'data/CU_LC08.001_SRB2_doy2020229_aid0001_reproject.tif'
    b3 = r'data/CU_LC08.001_SRB3_doy2020229_aid0001_reproject.tif'
    b4 = r'data/CU_LC08.001_SRB4_doy2020229_aid0001_reproject.tif'
    b5 = r'data/CU_LC08.001_SRB5_doy2020229_aid0001_reproject.tif'
    b6 = r'data/CU_LC08.001_SRB6_doy2020229_aid0001_reproject.tif'
    b7 = r'data/CU_LC08.001_SRB7_doy2020229_aid0001_reproject.tif'
    qa = r'data/CU_LC08.001_PIXELQA_doy2020229_aid0001_reproject.tif'
    
    dataset2 = gdal.Open(b1)
    band2 = dataset2.GetRasterBand(1)
    print(band2.GetNoDataValue())


if __name__ == '__main__':
    main()
