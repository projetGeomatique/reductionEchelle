from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
from landsat import Landsat
from modis import Modis
import pandas as pd
#import matplotlib.pyplot as plt
#import seaborn as sns


class ReductionEchelle():
    def __init__(self, modis_image, landsat_image):
        self.modis_image = modis_image
        self.landsat_image = landsat_image

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

        # Affichage du shape pour vérifier si les tailles sont compatibles
        print("MODIS shape:")
        print(self.modis_image.lst.shape)
        print("Landsat shape:")
        print(self.landsat_image.b1.shape)


    def applyDownscaling(self):
        """ Entraîne un modèle de Random Forest Regression avec certains prédicteurs. Les prédicteurs pourront
            éventuellement être spécifiés en input arguments.
        """

        # Utilisation de NDVI, NDWI et NDBI ici
        ndvi = self.landsat_image.getNdvi()
        ndwi = self.landsat_image.getNdwi()
        ndbi = self.landsat_image.getNdbi()

        # Reshape les array en colonnes
        ndvi = ndvi.reshape(-1, 1)
        ndwi = ndwi.reshape(-1, 1)
        ndbi = ndbi.reshape(-1, 1)

        # Stack les colonnes ensemble (array de 3 colonnes)
        col_stack = np.column_stack((ndvi, ndwi, ndbi))

        # Création d'un dataframe Pandas qui contient le array de 3 colonnes et qui indique les labels de chacune des
        # colonnes
        df = pd.DataFrame(col_stack, columns=['NDVI', 'NDWI', 'NDBI'])

        # Sauvegarder en CSV pour visualisation et vérification plus tard
        df.to_csv(r'donneesModele/donnees_utilisees.csv', index=False)

        # Conversion du array MODIS LST en un format compatible au Random Forest Regression
        modis_lst_array = self.modis_image.lst.ravel()

        # Split de l'échantillon d'entraînement et de l'échantillon de test (échantillon de test = 20% de l'échantillon
        # total)
        X_train, X_test, y_train, y_test = train_test_split(df, modis_lst_array, test_size=0.2, random_state=0)

        # Initialisation du régresseur avec 100 estimateurs
        regressor = RandomForestRegressor(n_estimators=100, random_state=0)

        # Entraînement du modèle
        regressor.fit(X_train, y_train)

        # Prédiction avec l'échantillon de test
        y_pred = regressor.predict(X_test)

        # Métriques de qualité sur le résultat prédit par rapport à l'échantillon de test (vérité)
        print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        # Importance de chacun des prédicteurs (NDVI, NDWI, NDBI) dans la prédiction
        print("Importance de chaque prédicteur", regressor.feature_importances_)

        # Affichage des résidus par rapport à l'échantillon de test (vérité)
        test_residuals = y_test - y_pred
        sns.scatterplot(x=y_test, y=test_residuals)
        plt.axhline(y=0, color='r', ls='--')
        plt.show()


def main():

    # Test
    b1 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB1_doy2020229_aid0001.tif'
    b2 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB2_doy2020229_aid0001.tif'
    b3 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB3_doy2020229_aid0001.tif'
    b4 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB4_doy2020229_aid0001.tif'
    b5 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB5_doy2020229_aid0001.tif'
    b6 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB6_doy2020229_aid0001.tif'
    b7 = r'fonctionReprojectAjoutees/data/CU_LC08.001_SRB7_doy2020229_aid0001.tif'
    qa = r'fonctionReprojectAjoutees/data/CU_LC08.001_PIXELQA_doy2020229_aid0001.tif'
    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa)

    lst = r'fonctionReprojectAjoutees/data/MOD11A1.006_Clear_day_cov_doy2020229_aid0001.tif'
    qa = r'fonctionReprojectAjoutees/data/MOD11A1.006_QC_Day_doy2020229_aid0001.tif'
    modis = Modis(lst, qa)

    rfr = ReductionEchelle(modis, landsat)
    rfr.prepareData()
    #rfr.applyDownscaling()


if __name__ == '__main__':
    main()
