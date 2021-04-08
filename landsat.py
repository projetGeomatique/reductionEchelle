import numpy as np
from image import Image
from osgeo import gdal, gdal_array, gdalconst, osr
import numpy.ma as ma


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
        """ Permet de récupérer un array Numpy contenant l'indice de végétation NDVI (Normalized Difference Vegetation
            Index) calculé à partir des bandes b4 et b5 de la collection d'images Landsat 8. La correction TOA des
            niveaux de gris de l'image est faite au préalable et les valeurs à l'extérieur de l'intervalle valide sont
            masquées.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de NDVI pour la collection d'images.
        """
        # ajout du where pour éviter la division par 0 (à voir quel résultat est obtenu dans ce cas...)
        b4_img = Image(self.b4)
        b4_meta = b4_img.getMetadata()
        b4_DN = b4_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        # conversion en réflectances
        if 'scale_factor' in b4_meta:
            b4 = np.add(np.multiply(b4_DN, float(b4_meta['scale_factor'])), float(b4_meta['add_offset']))
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
        """ Permet de récupérer un array Numpy contenant l'indice de bâti NDBI (Normalized Difference Built-up Index)
            calculé à partir des bandes b5 et b6 de la collection d'images Landsat 8. La correction TOA des niveaux
            de gris de l'image est faite au préalable et les valeurs à l'extérieur de l'intervalle valide sont masquées.
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
        """ Permet de récupérer un array Numpy contenant l'indice d'humidité NDWI (Normalized Difference Water Index)
            calculé à partir des bandes b3 et b5 de la collection d'images Landsat 8. La correction TOA des niveaux de
            gris de l'image est faite au préalable et les valeurs à l'extérieur de l'intervalle valide sont masquées.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de NDWI pour la collection d'images.
        """
        b3_img = Image(self.b3)
        b3_meta = b3_img.getMetadata()
        b3_DN = b3_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b3_meta:
            b3 = np.add(np.multiply(b3_DN, float(b3_meta['scale_factor'])), float(b3_meta['add_offset']))
        else:
            b3 = np.add(np.multiply(b3_DN, float(0.0001)), float(0))

        b5_img = Image(self.b5)
        b5_meta = b5_img.getMetadata()
        b5_DN = b5_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b5_meta:
            b5 = np.add(np.multiply(b5_DN, float(b5_meta['scale_factor'])), float(b5_meta['add_offset']))
        else:
            b5 = np.add(np.multiply(b5_DN, float(0.0001)), float(0))

        return np.divide(np.subtract(b3, b5), np.add(b3, b5), where=((np.add(b3, b5)) != 0))

    def getMndwi(self):
        """ Permet de récupérer un array Numpy contenant l'indice d'humidité MNDWI (Modified Normalized Difference Water
            Index) calculé à partir des bandes b3 et b6 de la collection d'images Landsat 8. La correction TOA des
            niveaux de gris de l'image est faite au préalable et les valeurs à l'extérieur de l'intervalle valide sont
            masquées.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de MNDWI pour la collection d'images.
        """
        b3_img = Image(self.b3)
        b3_meta = b3_img.getMetadata()
        b3_DN = b3_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b3_meta:
            b3 = np.add(np.multiply(b3_DN, float(b3_meta['scale_factor'])), float(b3_meta['add_offset']))
        else:
            b3 = np.add(np.multiply(b3_DN, float(0.0001)), float(0))

        b6_img = Image(self.b6)
        b6_meta = b6_img.getMetadata()
        b6_DN = b6_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b6_meta:
            b6 = np.add(np.multiply(b6_DN, float(b6_meta['scale_factor'])), float(b6_meta['add_offset']))
        else:
            b6 = np.add(np.multiply(b6_DN, float(0.0001)), float(0))

        return np.divide(np.subtract(b3, b6), np.add(b3, b6), where=((np.add(b3, b6)) != 0))

    def getSAVI(self):
        """ Permet de récupérer un array Numpy contenant l'indice de végétation ajusté au sol SAVI (Soil Adjusted
            Vegetation Index) calculé à partir des bandes b4 et b5 de la collection d'images Landsat 8. La correction
            TOA des niveaux de gris de l'image est faite au préalable et les valeurs à l'extérieur de l'intervalle
            valide sont masquées.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de SAVI pour la collection d'images.
        """
        b4_img = Image(self.b4)
        b4_meta = b4_img.getMetadata()
        b4_DN = b4_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b4_meta:
            b4 = np.add(np.multiply(b4_DN, float(b4_meta['scale_factor'])), float(b4_meta['add_offset']))
        else:
            b4 = np.add(np.multiply(b4_DN, float(0.0001)), float(0))

        b5_img = Image(self.b5)
        b5_meta = b5_img.getMetadata()
        b5_DN = b5_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b5_meta:
            b5 = np.add(np.multiply(b5_DN, float(b5_meta['scale_factor'])), float(b5_meta['add_offset']))
        else:
            b5 = np.add(np.multiply(b5_DN, float(0.0001)), float(0))

        return np.multiply(np.divide(np.subtract(b5, b4), (np.add(b5, b4)+0.5), where=(np.add(b5, b4)+0.5) != 0), 1.5)

    def getAlbedo(self):
        """ Permet de récupérer un array Numpy contenant l'albédo de surface calculé à partir des bandes b2, b4, b5, b6
            et b7 de la collection d'images Landsat 8. La correction TOA des niveaux de gris de l'image est faite au
            préalable et les valeurs à l'extérieur de l'intervalle valide sont masquées.
                Returns:
                    albedo_array (Numpy.ma.masked_array): Array des valeurs d'albédo de surface pour la collection d'images.
        """
        b2_img = Image(self.b2)
        b2_meta = b2_img.getMetadata()
        b2_DN = b2_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b2_meta:
            b2 = np.add(np.multiply(b2_DN, float(b2_meta['scale_factor'])), float(b2_meta['add_offset']))
        else:
            b2 = np.add(np.multiply(b2_DN, float(0.0001)), float(0))

        b4_img = Image(self.b4)
        b4_meta = b4_img.getMetadata()
        b4_DN = b4_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b4_meta:
            b4 = np.add(np.multiply(b4_DN, float(b4_meta['scale_factor'])), float(b4_meta['add_offset']))
        else:
            b4 = np.add(np.multiply(b4_DN, float(0.0001)), float(0))

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

        b7_img = Image(self.b7)
        b7_meta = b7_img.getMetadata()
        b7_DN = b7_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b7_meta:
            b7 = np.add(np.multiply(b7_DN, float(b7_meta['scale_factor'])), float(b7_meta['add_offset']))
        else:
            b7 = np.add(np.multiply(b7_DN, float(0.0001)), float(0))

        albedo_array = 0.356*b2 + 0.13*b4 + 0.373*b5 + 0.085*b6 + 0.072*b7 - 0.0018
        return albedo_array
  
    def getBSI(self):
        """ Permet de récupérer un array Numpy contenant l'indice de sol nu BSI (Bare Soil Index) calculé à partir des
            bandes b3 et b5 de la collection d'images Landsat 8. La correction TOA des niveaux de gris de l'image est
            faite au préalable et les valeurs à l'extérieur de l'intervalle valide sont masquées.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de BSI pour la collection d'images.
        """
        b3_img = Image(self.b3)
        b3_meta = b3_img.getMetadata()
        b3_DN = b3_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b3_meta:
            b3 = np.add(np.multiply(b3_DN, float(b3_meta['scale_factor'])), float(b3_meta['add_offset']))
        else:
            b3 = np.add(np.multiply(b3_DN, float(0.0001)), float(0))

        b5_img = Image(self.b5)
        b5_meta = b5_img.getMetadata()
        b5_DN = b5_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b5_meta:
            b5 = np.add(np.multiply(b5_DN, float(b5_meta['scale_factor'])), float(b5_meta['add_offset']))
        else:
            b5 = np.add(np.multiply(b5_DN, float(0.0001)), float(0))

        return np.divide(np.add(b3, b5), np.subtract(b3, b5), where=((np.subtract(b3, b5)) != 0))

    def getUI(self):
        """ Permet de récupérer un array Numpy contenant l'indice d'urbanisation UI (Urban Index) calculé à partir des
            bandes b5 et b7 de la collection d'images Landsat 8. La correction TOA des niveaux de gris de l'image est
            faite au préalable et les valeurs à l'extérieur de l'intervalle valide sont masquées.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de UI pour la collection d'images.
        """
        b5_img = Image(self.b5)
        b5_meta = b5_img.getMetadata()
        b5_DN = b5_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b5_meta:
            b5 = np.add(np.multiply(b5_DN, float(b5_meta['scale_factor'])), float(b5_meta['add_offset']))
        else:
            b5 = np.add(np.multiply(b5_DN, float(0.0001)), float(0))

        b7_img = Image(self.b7)
        b7_meta = b7_img.getMetadata()
        b7_DN = b7_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b7_meta:
            b7 = np.add(np.multiply(b7_DN, float(b7_meta['scale_factor'])), float(b7_meta['add_offset']))
        else:
            b7 = np.add(np.multiply(b7_DN, float(0.0001)), float(0))

        return np.divide(np.subtract(b7, b5), np.add(b7, b5), where=((np.add(b7, b5)) != 0))

    def getEVI(self):
        """ Permet de récupérer un array Numpy contenant l'indice de végétation renforcé EVI (Enhanced Vegetation Index)
            calculé à partir des bandes b2, b4 et b5 de la collection d'images Landsat 8. La correction TOA des niveaux
            de gris de l'image est faite au préalable et les valeurs à l'extérieur de l'intervalle valide sont masquées.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de EVI pour la collection d'images.
        """
        b2_img = Image(self.b2)
        b2_meta = b2_img.getMetadata()
        b2_DN = b2_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b2_meta:
            b2 = np.add(np.multiply(b2_DN, float(b2_meta['scale_factor'])), float(b2_meta['add_offset']))
        else:
            b2 = np.add(np.multiply(b2_DN, float(0.0001)), float(0))

        b4_img = Image(self.b4)
        b4_meta = b4_img.getMetadata()
        b4_DN = b4_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b4_meta:
            b4 = np.add(np.multiply(b4_DN, float(b4_meta['scale_factor'])), float(b4_meta['add_offset']))
        else:
            b4 = np.add(np.multiply(b4_DN, float(0.0001)), float(0))

        b5_img = Image(self.b5)
        b5_meta = b5_img.getMetadata()
        b5_DN = b5_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in b5_meta:
            b5 = np.add(np.multiply(b5_DN, float(b5_meta['scale_factor'])), float(b5_meta['add_offset']))
        else:
            b5 = np.add(np.multiply(b5_DN, float(0.0001)), float(0))

        return np.multiply(np.divide(np.subtract(b5, b4), (b5 + 6*b4 - 7.5*b2 + 1), where=((b5 + 6*b4 - 7.5*b2 + 1) != 0)), 2.5)

    def getIBI(self):
        """ Permet de récupérer un array Numpy contenant l'indice de bâti IBI (Index-Based Built-up Index) calculé à
            partir des indices NDBI, SAVI et MNDWI. La correction TOA des niveaux de gris de l'image est faite au
            préalable et les valeurs à l'extérieur de l'intervalle valide sont masquées.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de IBI pour la collection d'images.
        """
        ndbi = self.getNdbi()
        savi = self.getSAVI()
        mndwi = self.getMndwi()

        numerator = np.subtract(ndbi, np.divide(np.add(savi, mndwi), 2))
        denominator = np.add(ndbi, np.divide(np.add(savi, mndwi), 2))

        return np.divide(numerator, denominator, where=(denominator != 0))

    def getBand(self, bandNumber):
        """ Permet de récupérer un array Numpy contenant les valeurs d'une des bandes disponibles dans la collection
            d'images Landsat 8. La correction TOA des niveaux de gris de l'image est faite au préalable et les valeurs
            à l'extérieur de l'intervalle valide sont masquées.
                Args:
                    bandNumber (int): Numéro de la bande qu'on veut récupérer sous la forme d'un array Numpy.
                Returns:
                    array (Numpy.ma.masked_array): Array des valeurs de réflectances de la bande sélectionnée.
        """
        if bandNumber == 1:
            band_img = Image(self.b1)
        elif bandNumber == 2:
            band_img = Image(self.b2)
        elif bandNumber == 3:
            band_img = Image(self.b3)
        elif bandNumber == 4:
            band_img = Image(self.b4)
        elif bandNumber == 5:
            band_img = Image(self.b5)
        elif bandNumber == 6:
            band_img = Image(self.b6)
        elif bandNumber == 7:
            band_img = Image(self.b7)
        else:
            raise Exception("Ce numéro de bande n'est pas inclus dans cet objet Landsat 8 (bandes 1 à 7 seulement).")

        band_meta = band_img.getMetadata()
        band_DN = band_img.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000)

        if 'scale_factor' in band_meta:
            band = np.add(np.multiply(band_DN, float(band_meta['scale_factor'])), float(band_meta['add_offset']))
        else:
            band = np.add(np.multiply(band_DN, float(0.0001)), float(0))

        return band


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

    def reprojectLandsat(self, referenceFile, reduce_zone=True):
        """ Permet de reprojeter, découper, aligner et rééchantillonner une image à partir d'une image de référence.
            Cette méthode effectue ces traitements pour l'ensemble des bandes de la collection Landsat 8. Elle fait
            appel à la méthode reprojectMatch() de la classe Image.
                Args:
                    referenceFile (string): Path du fichier de l'image de référence à utiliser.
                                            reduce_zone (bool): Indicateur permettant de choisir si on souhaite réduire
                                            la zone d'étude sur laquelle les images sont "matchées". Ceci est utile pour
                                            éviter des problèmes avec des valeurs nulles sur les bords des images qui
                                            s'alignent sur le referenceFile. Par défaut, cette option est égale à True
                                            (donc, on effectue le rétrécissement de zone).
        """
        bandsPaths = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7, self.qa]
        newBandsPaths = []

        for band in bandsPaths:
            image = Image(band)
            newBandsPaths.append(image.reprojectMatch(referenceFile, reduce_zone))

        # assigner des nouveaux path aux bandes
        self.b1 = newBandsPaths[0]
        self.b2 = newBandsPaths[1]
        self.b3 = newBandsPaths[2]
        self.b4 = newBandsPaths[3]
        self.b5 = newBandsPaths[4]
        self.b6 = newBandsPaths[5]
        self.b7 = newBandsPaths[6]
        self.qa = newBandsPaths[7]

        print("Landsat:          Reprojection termine")

    def maskClouds30m(self):
        b1_img = Image(self.b1)
        b2_img = Image(self.b2)
        b3_img = Image(self.b3)
        b4_img = Image(self.b4)
        b5_img = Image(self.b5)
        b6_img = Image(self.b6)
        b7_img = Image(self.b7)

        bandsPaths = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7]
        newBandsPaths = []

        for band in bandsPaths:
            image = Image(band)

            print(np.shape(image.getArray()))
            print(np.shape(Image(self.qa).getArray()))

            band_masked = image.getArray(masked=True, lower_valid_range=0, upper_valid_range=10000,
                                         qa_filename=self.qa)
            band_masked = ma.filled(band_masked, np.nan)
            image.save_band(band_masked, image.filename.replace(".tif", "masked30m.tif"))
            newBandsPaths.append(image.filename.replace(".tif", "masked30m.tif"))

        self.b1 = newBandsPaths[0]
        self.b2 = newBandsPaths[1]
        self.b3 = newBandsPaths[2]
        self.b4 = newBandsPaths[3]
        self.b5 = newBandsPaths[4]
        self.b6 = newBandsPaths[5]
        self.b7 = newBandsPaths[6]

    def maskClouds1000m(self, filename):
        b1_img = Image(self.b1)
        b2_img = Image(self.b2)
        b3_img = Image(self.b3)
        b4_img = Image(self.b4)
        b5_img = Image(self.b5)
        b6_img = Image(self.b6)
        b7_img = Image(self.b7)

        bandsPaths = [self.b1, self.b2, self.b3, self.b4, self.b5, self.b6, self.b7]
        newBandsPaths = []

        for band in bandsPaths:
            image = Image(band)
            band_masked = image.getArray(masked=True, lower_valid_range=0, upper_valid_range=10,
                                         cloud_overlay_filename=filename)
            band_masked = ma.filled(band_masked, np.nan)
            image.save_band(band_masked, image.filename.replace("masked30m_reproject", "masked1000m"))

            #newBandsPaths.append(image.filename.replace("masked30m", "masked1000m"))
            newBandsPaths.append(image.filename.replace("masked30m_reproject", "masked1000m"))

        self.b1 = newBandsPaths[0]
        self.b2 = newBandsPaths[1]
        self.b3 = newBandsPaths[2]
        self.b4 = newBandsPaths[3]
        self.b5 = newBandsPaths[4]
        self.b6 = newBandsPaths[5]
        self.b7 = newBandsPaths[6]


def main():
    pass

  
if __name__ == '__main__':
    main()
