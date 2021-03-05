import numpy as np
import numpy.ma as ma
from landsat import Landsat
from image import Image
from modis import Modis
from aster import Aster
import pandas as pd


class Secteur:
    """ Classe modélisant un secteur qu'on veut utiliser pour entraîner et prédire la température de surface.
        Ce secteur est composé d'images Landsat, MODIS et ASTER se superposant sur une même zone.
        On utilise l'image MODIS comme référence pour aligner, découper et rééchantillonner les autres images sur une
        zone de taille identique.

        Attributes:
            modis_image (Modis): Collection d'images MODIS préalablement instanciée dans la classe Modis.
            landsat_image (Landsat): Collection d'images Landsat 8 préalablement instanciée dans la classe Landsat.
            aster_image (Aster): Collection d'images Aster préalablement instanciée dans la classe Aster.
            mask (Numpy.ma.masked_array): Masque à appliquer à l'ensemble des images une fois alignées et rééchantillonnées.
    """
    def __init__(self, modis_image, landsat_image, aster_image):
        self.modis_image = modis_image
        self.landsat_image = landsat_image
        self.aster_image = aster_image

        self.mask = None

    def prepareData(self, train_model=True):
        """ Permet de préparer les images avant l'entraînement du modèle ou à la prédiction de la température de surface.

            Args:
                train_model (bool): Indicateur pour déterminer si les données sont préparées pour l'entraînement ou
                                    pour la prédiction.
                                    Si la valeur de l'indicateur est "False", on subdivise l'image MODIS de 1km à 100m
                                    et on reprojette les images pour avoir un nombre de pixels compatible (à 100m).
                                    Sinon, on fait le tout à 1km.
        """

        # ********** masquer nuages + ré-échantillonnage à ajouter **********

        if train_model:
            # reprojection de l'image Landsat et ré-échantillonnage à 100m
            self.landsat_image.reprojectLandsat(self.modis_image.lst)

            # reprojection de l'image Aster pour avoir la même taille que celle de Landsat préalablement reprojetée
            self.aster_image.reprojectAster(self.modis_image.lst)

            # reprojection de l'image MODIS pour avoir la même taille que celle de Landsat préalablement reprojetée
            self.modis_image.reprojectModis(self.modis_image.lst)

        else:
            # subdivision de l'image MODIS de 1km à 100m
            self.modis_image.subdividePixel(10, "file",
                                            self.modis_image.lst.split(".")[0] + '_subdivided_100m.tif')

            # réinitialise les bandes des images aux bandes originales (rééchantillonnage de 30m à 100m au lieu de 1km
            # à 100m pour Landsat)
            self.landsat_image.b1 = self.landsat_image.b1.replace("_reproject", "")
            self.landsat_image.b2 = self.landsat_image.b2.replace("_reproject", "")
            self.landsat_image.b3 = self.landsat_image.b3.replace("_reproject", "")
            self.landsat_image.b4 = self.landsat_image.b4.replace("_reproject", "")
            self.landsat_image.b5 = self.landsat_image.b5.replace("_reproject", "")
            self.landsat_image.b6 = self.landsat_image.b6.replace("_reproject", "")
            self.landsat_image.b7 = self.landsat_image.b7.replace("_reproject", "")
            self.landsat_image.qa = self.landsat_image.qa.replace("_reproject", "")

            self.modis_image.lst = self.modis_image.lst.replace("_reproject", "")
            self.modis_image.qa = self.modis_image.qa.replace("_reproject", "")

            self.aster_image.mnt = self.aster_image.mnt.replace("_reproject", "")
            self.aster_image.qa = self.aster_image.qa.replace("_reproject", "")

            # reprojection de l'image Landsat et ré-échantillonnage à 100m
            self.landsat_image.reprojectLandsat(self.modis_image.lst.split(".")[0] + '_subdivided_100m.tif')

            # reprojection de l'image Aster pour avoir la même taille que celle de Landsat préalablement reprojetée
            self.aster_image.reprojectAster(self.modis_image.lst.split(".")[0] + '_subdivided_100m.tif')

            # reprojection de l'image MODIS pour avoir la même taille que celle de Landsat préalablement reprojetée
            self.modis_image.reprojectModis(self.modis_image.lst.split(".")[0] + '_subdivided_100m.tif')

    def getDf(self, train=True):
        """ Aller chercher le DataFrame contenant l'ensemble des prédicteurs préparés pour le downscaling.
                Args:
                    train (bool): Indicateur pour déterminer si les données sont préparées pour l'entraînement ou
                                  pour la prédiction.
                                  Si la valeur de l'indicateur est "False", on combine les variables ensemble avec des
                                  array masqués pour qu'on puisse retirer les valeurs nulles qui ont été masquées.
                                  Sinon, on combine les variables ensemble avec un array standard et on masque les
                                  données par la suite (le Random Forest Regression n'accepte pas les valeurs nulles,
                                  il ignore les masques dans le Pandas DataFrame).
                Returns:
                    df (Pandas DataFrame): Combinaison des variables nécessaires pour l'entraînement du modèle ou la
                                           prédiction de la température de surface.
        """

        # ******** PRÉDICTEURS ***********

        # Utilisation de NDVI, NDWI et NDBI ici (masquage des valeurs nulles inclus dans les get pour Landsat)
        ndvi = self.landsat_image.getNdvi()
        print("NDVI termine")
        ndwi = self.landsat_image.getNdwi()
        print("NDWI termine")
        ndbi = self.landsat_image.getNdbi()
        print("NDBI termine")

        # pente = self.aster_image.getPente()
        # print("Pente termine")

        # Reshape les array en colonnes
        ndvi = ndvi.reshape(-1, 1)
        ndwi = ndwi.reshape(-1, 1)
        ndbi = ndbi.reshape(-1, 1)

        # pente = pente.reshape(-1, 1)

        # ******** MODIS LST ***********

        # Conversion du array MODIS LST en un format compatible au Random Forest Regression
        lst_image = Image(self.modis_image.lst)

        modis_array = lst_image.getArray(masked=True, lower_valid_range=7500, upper_valid_range=65535)

        # convertir à des températures de surface en Celsius
        lst_metadata = lst_image.getMetadata()

        if 'scale_factor' in lst_metadata:
            scale_factor = float(lst_metadata['scale_factor'])  # multiplier par 0.02
            add_offset = float(lst_metadata['add_offset'])
        else:
            scale_factor = float(0.02)
            add_offset = float(0)

        kelvin_array = np.add(np.multiply(modis_array, scale_factor), add_offset)
        lst_celsius_array = np.subtract(kelvin_array, 273.15)

        lst_celsius_array = lst_celsius_array.reshape(-1, 1)

        # appliquer les mêmes masques partout (si un pixel est masqué dans une image, il doit être masqué pour toutes
        # les images)
        ndvi_mask = ma.getmaskarray(ndvi)
        ndwi_mask = ma.getmaskarray(ndwi)
        ndbi_mask = ma.getmaskarray(ndbi)

        lst_mask = ma.getmaskarray(lst_celsius_array)

        mask = np.add(np.add(np.add(ndvi_mask, ndwi_mask), ndbi_mask), lst_mask)
        self.mask = mask

        ndvi = ma.masked_array(ndvi, mask)
        ndwi = ma.masked_array(ndwi, mask)
        ndbi = ma.masked_array(ndbi, mask)
        lst_celsius_array = ma.masked_array(lst_celsius_array, mask)

        # ********* Stack les colonnes ensemble **********
        # col_stack = np.column_stack((ndvi, ndwi, ndbi, pente))

        if train:
            col_stack = ma.column_stack((ndvi, ndwi, ndbi, lst_celsius_array))
        else:
            col_stack = np.column_stack((ndvi, ndwi, ndbi, lst_celsius_array))

        # Création d'un dataframe Pandas qui contient le array de 4 colonnes et qui indique les labels de chacune des
        # colonnes
        # df = pd.DataFrame(col_stack, columns=['NDVI', 'NDWI', 'NDBI', 'Pente'])
        df = pd.DataFrame(col_stack, columns=['NDVI', 'NDWI', 'NDBI', 'LST'])  # dataframe ne gère pas les masques?

        #df_test = df['LST']
        #print(df_test[162])

        # Sauvegarder en CSV pour visualisation et vérification plus tard
        df.to_csv(r'secteur3/donnees_utilisees.csv', index=False)

        return df

    """
    def getMODIS(self):
        # Aller chercher le array formatté de température de surface MODIS comme variable à prédire dans le
        #    Random Forest Regression.

        # Conversion du array MODIS LST en un format compatible au Random Forest Regression
        lst_image = Image(self.modis_image.lst)

        array = lst_image.getArray(masked=True, lower_valid_range=7500, upper_valid_range=65535)

        # convertir à des températures de surface en Celsius
        lst_metadata = lst_image.getMetadata()

        scale_factor = float(lst_metadata['scale_factor'])  # multiplier par 0.02
        add_offset = float(lst_metadata['add_offset'])

        kelvin_array = np.add(np.multiply(array, scale_factor), add_offset)
        celsius_array = np.subtract(kelvin_array, 273.15)

        print("Get MODIS termine")

        return celsius_array.ravel()
    """


def main():
    """ Tests de la classe et de ses méthodes.
    """

    # secteur1
    b1 = r'secteur/CU_LC08.001_SRB1_doy2020229_aid0001.tif'
    b2 = r'secteur/CU_LC08.001_SRB2_doy2020229_aid0001.tif'
    b3 = r'secteur/CU_LC08.001_SRB3_doy2020229_aid0001.tif'
    b4 = r'secteur/CU_LC08.001_SRB4_doy2020229_aid0001.tif'
    b5 = r'secteur/CU_LC08.001_SRB5_doy2020229_aid0001.tif'
    b6 = r'secteur/CU_LC08.001_SRB6_doy2020229_aid0001.tif'
    b7 = r'secteur/CU_LC08.001_SRB7_doy2020229_aid0001.tif'
    qa = r'secteur/CU_LC08.001_PIXELQA_doy2020229_aid0001.tif'
    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa)

    lst = r'secteur/MOD11A1.006_LST_Day_1km_doy2020221_aid0001.tif'
    qa = r'secteur/MOD11A1.006_QC_Day_doy2020229_aid0001.tif'  # pas la bonne image, mais juste pour un test, vu que je
    # trouve pas la bonne image (QA n'est pas utilisé dans le
    # test)
    modis = Modis(lst, qa)

    mnt = r'secteur/ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif'
    qa = r'secteur/ASTGTM_NUMNC.003_ASTER_GDEM_NUM_doy2000061_aid0001.tif'  # aussi un test (ne semble pas valide)
    aster = Aster(mnt, qa)

    rfr = Secteur(modis, landsat, aster)
    rfr.prepareData()
    df = rfr.getDf()

    #rfr.getMODIS()

    print(df)
    print(df.drop('LST', axis=1))
    print(df['LST'])

    a = df['LST']
    print(a.ravel())


if __name__ == '__main__':
    main()
