from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from landsat import Landsat
from image import Image
from modis import Modis
from aster import Aster
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns


class Secteur():
    def __init__(self, modis_image, landsat_image, aster_image):
        self.modis_image = modis_image
        self.landsat_image = landsat_image
        self.aster_image = aster_image

    def prepareData(self):
        """ Permet de préparer les images avant l'entraînement du modèle. Subdivise l'image MODIS de 1km à 100m et
            reprojette les images pour avoir un nombre de pixels compatible.
        """

        # masquer nuages + ré-échantillonnage à ajouter
        # nuages = self.pred.qa

        # subdivision de l'image MODIS de 1km à 100m
        self.modis_image.subdividePixel(10, "file", self.modis_image.filename.split(".")[0] + '_subdivided_100m.tif')

        # reprojection de l'image Landsat et ré-échantillonnage à 100m
        self.landsat_image.reprojectLandsat(self.modis_image.filename.split(".")[0] + '_subdivided_100m.tif')

        # reprojection de l'image MODIS pour avoir la même taille que celle de Landsat préalablement reprojetée
        self.modis_image.reprojectModis(self.modis_image.filename.split(".")[0] + '_subdivided_100m.tif')

        # reprojection de l'image Aster pour avoir la même taille que celle de Landsat préalablement reprojetée
        self.aster_image.reprojectAster(self.modis_image.filename.split(".")[0] + '_subdivided_100m.tif')

        # Affichage du shape pour vérifier si les tailles sont compatibles
        print("MODIS shape:")
        #print(self.modis_image.lst.shape)
        print("Landsat shape:")
        #print(self.landsat_image.b1.shape)


    def getDf(self):
        """ Entraîne un modèle de Random Forest Regression avec certains prédicteurs. Les prédicteurs pourront
            éventuellement être spécifiés en input arguments.
        """

        # Utilisation de NDVI, NDWI et NDBI ici
        ndvi = self.landsat_image.getNdvi()
        ndwi = self.landsat_image.getNdwi()
        ndbi = self.landsat_image.getNdbi()
        pente = self.aster_image.getPente()

        # Reshape les array en colonnes
        ndvi = ndvi.reshape(-1, 1)
        ndwi = ndwi.reshape(-1, 1)
        ndbi = ndbi.reshape(-1, 1)
        pente = pente.reshape(-1, 1)

        # Stack les colonnes ensemble (array de 3 colonnes)
        col_stack = np.column_stack((ndvi, ndwi, ndbi, pente))

        # Création d'un dataframe Pandas qui contient le array de 3 colonnes et qui indique les labels de chacune des
        # colonnes
        df = pd.DataFrame(col_stack, columns=['NDVI', 'NDWI', 'NDBI', 'Pente'])

        # Sauvegarder en CSV pour visualisation et vérification plus tard
        df.to_csv(r'donneesModele/donnees_utilisees.csv', index=False)

        # Conversion du array MODIS LST en un format compatible au Random Forest Regression
        print(self.modis_image.lst)

        #modis_lst_array = Image.getArray(self.modis_image.lst).ravel()
        modis_lst_array = self.modis_image.getArray(self.modis_image.lst).ravel()

        # Split de l'échantillon d'entraînement et de l'échantillon de test (échantillon de test = 20% de l'échantillon
        # total)
        #X_train, X_test, y_train, y_test = train_test_split(df, modis_lst_array, test_size=0.2, random_state=0)

        return df


def main():
    # secteur1
    b1 = r'data/CU_LC08.001_SRB1_doy2020229_aid0001.tif'
    b2 = r'data/CU_LC08.001_SRB2_doy2020229_aid0001.tif'
    b3 = r'data/CU_LC08.001_SRB3_doy2020229_aid0001.tif'
    b4 = r'data/CU_LC08.001_SRB4_doy2020229_aid0001.tif'
    b5 = r'data/CU_LC08.001_SRB5_doy2020229_aid0001.tif'
    b6 = r'data/CU_LC08.001_SRB6_doy2020229_aid0001.tif'
    b7 = r'data/CU_LC08.001_SRB7_doy2020229_aid0001.tif'
    qa = r'data/CU_LC08.001_PIXELQA_doy2020229_aid0001.tif'
    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa)

    lst = r'data/MOD11A1.006_Clear_day_cov_doy2020229_aid0001.tif'
    qa = r'data/MOD11A1.006_QC_Day_doy2020229_aid0001.tif'
    modis = Modis(lst, qa)

    mnt = r'data\ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif'
    qa = r'data\ASTGTM_NUMNC.003_ASTER_GDEM_NUM_doy2000061_aid0001.tif'
    aster = Aster(mnt, qa)

    rfr = Secteur(modis, landsat)
    rfr.prepareData()
    rfr.getDf()


if __name__ == '__main__':
    main()
