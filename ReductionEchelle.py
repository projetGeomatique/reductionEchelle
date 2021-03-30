from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.ma as ma
from image import Image
from landsat import Landsat
from modis import Modis
from aster import Aster
from secteur import Secteur
import matplotlib.pyplot as plt
import seaborn as sns


class ReductionEchelle:
    """ Classe regroupant les méthodes nécessaires pour effectuer la réduction d'échelle à l'aide de l'algorithme
        d'apprentissage machine de Random Forest Regression.

        Attributes:
            secteur (Secteur): Secteur contenant tous les prédicteurs et variables dépendantes nécessaires pour
                               effectuer la régression non-linéaire.
    """
    def __init__(self, secteur):
        self.secteur = secteur

    def applyDownscaling(self, predictors, outputFile, residualCorrection=False, outputFile_withResidualCorrection=None):
        """ Entraîne un modèle de Random Forest Regression avec certains prédicteurs spécifiés dans une liste en entrée
            et applique par la suite la réduction d'échelle avec le modèle entraîné.
            Le résultat est sauvegardé dans une nouvelle image.
                Args:
                    predictors (list): Liste de string des prédicteurs à inclure dans l'entraînement ou la prédiction
                                       de la réduction d'échelle. Cet argument doit prendre la forme suivante:
                                       ['NDVI', 'NDWI', 'NDBI', 'MNT', 'Pente']
                                       avec des prédicteurs disponibles dans cette méthode.
                    outputFile (string): Path vers le fichier dans lequel on souhaite sauvegarder le résultat de la
                                         réduction d'échelle à 100m.
        """

        dataframe = self.secteur.getDf(predictors, train=True)  # on va cherche le Pandas DataFrame du secteur

        predicteurs = dataframe.drop('LST', axis=1)  # on retire la température de surface (LST) du DataFrame pour ne
                                                     # conserver que les prédicteurs
        predicteurs = predicteurs.dropna()  # pour l'entraînement, on retire les valeurs Nulles

        modis_LST = dataframe['LST']
        modis_LST = modis_LST.dropna()  # pour l'entraînement, on retire les valeurs Nulles
        modis_LST = modis_LST.ravel()  # format accepté par le Random Forest Regression pour la variable dépendante Y
                                       # (une seule ligne)

        # Split de l'échantillon d'entraînement et de l'échantillon de test (échantillon de test = 25% de l'échantillon
        # total)
        test_sample_size = 0.25
        X_train, X_test, y_train, y_test = train_test_split(predicteurs, modis_LST,
                                                            test_size=test_sample_size, random_state=42)

        # Initialisation du régresseur avec 100 estimateurs
        regressor = RandomForestRegressor(n_estimators=100, random_state=42)

        # Entraînement du modèle
        regressor.fit(X_train, y_train)

        # ----------- Validation interne ---------------
        # Prédiction avec l'échantillon de test
        y_pred = regressor.predict(X_test)

        # Métriques de qualité sur le résultat prédit par rapport à l'échantillon de test (vérité)
        print("\n")
        print("Validation interne avec {}% des échantillons".format(test_sample_size*100))
        print('Coefficient of determination (R2):', metrics.r2_score(y_test, y_pred))
        print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
        print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
        print('Accuracy:', 100 - np.mean(100 * ((abs(y_pred - y_test)) / y_test)))
        print('Explained variance score (EVS):', metrics.explained_variance_score(y_test, y_pred))

        print("")
        # Importance de chacun des prédicteurs dans la prédiction
        print("Prédicteurs utilisés:", list(predicteurs.columns))
        print("Importance de chaque prédicteur:", regressor.feature_importances_)
        print("")

        # Graphique de l'importance des prédicteurs
        fig, ax = plt.subplots()
        y_pos = np.arange(len(list(predicteurs.columns)))
        ax.barh(y_pos, regressor.feature_importances_, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(list(predicteurs.columns))
        plt.show()

        # Affichage des résidus par rapport à l'échantillon de test (vérité)
        test_residuals = y_test - y_pred
        sns.scatterplot(x=y_test, y=test_residuals)
        plt.axhline(y=0, color='r', ls='--')
        plt.show()

        # ------------- Prédiction ------------------
        # préparer les données pour la prédiction (downscaling) à 100m
        self.secteur.prepareData(train_model=False)

        # Prédiction
        dataframe_predict = self.secteur.getDf(predictors, train=False)
        y_downscale_100m = regressor.predict(dataframe_predict.drop('LST', axis=1))

        # *********** (à faire avec Landsat LST) ****************

        # Métriques de qualité sur le résultat prédit par rapport à l'échantillon de vérité terrain
        # print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(y_test, y_pred))
        # print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
        # print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

        # Importance de chacun des prédicteurs (NDVI, NDWI, NDBI) dans la prédiction
        #print("Importance de chaque prédicteur", regressor.feature_importances_)

        # Affichage des résidus par rapport à l'échantillon de vérité terrain
        # ************* (à faire avec Landsat LST) **************
        # test_residuals = y_test - y_pred
        # sns.scatterplot(x=y_test, y=test_residuals)
        # plt.axhline(y=0, color='r', ls='--')
        # plt.show()

        # sauvegarder le résultat avec une autre image (de mêmes dimensions et avec la même référence spatiale)
        # comme référence
        reference_image = Image(self.secteur.modis_image.lst)  # LST MODIS subdivisée à 100m

        y_downscale_100m_masked = ma.masked_array(y_downscale_100m, self.secteur.mask.ravel())
        y_downscale_100m_masked = ma.filled(y_downscale_100m_masked, np.nan)  # on retire les valeurs masquées du output

        y_downscale_100m_masked = y_downscale_100m_masked.reshape(reference_image.ysize, reference_image.xsize)

        reference_image.save_band(y_downscale_100m_masked, outputFile)

        # ------------- Correction pour les résidus ------------------

        if residualCorrection:

            predicted_100m_image = Image(outputFile)

            # ramener à 1km
            predicted_100m_image.reprojectMatch(self.secteur.modis_image.lst.replace("_reproject", ""), reduce_zone=True)
            resampled_predicted_100m_image = Image(outputFile.replace(".tif", "_reproject.tif"))

            # Conversion du array MODIS LST en un format compatible au Random Forest Regression
            tmp_lst_image_1km = Image(self.secteur.modis_image.lst)
            tmp_lst_image_1km.reprojectMatch(self.secteur.modis_image.lst.replace("_reproject", ""), reduce_zone=True)

            lst_image_1km = Image(self.secteur.modis_image.lst.replace("_reproject", "_reproject_reproject"))
            modis_array = lst_image_1km.getArray(masked=True, lower_valid_range=7500, upper_valid_range=65535)

            # convertir à des températures de surface en Celsius
            lst_metadata = lst_image_1km.getMetadata()

            # vérifier si le scale_factor est présent dans les métadonnées (c'est le cas pour AppEEARS, pas EarthData)
            if 'scale_factor' in lst_metadata:
                scale_factor = float(lst_metadata['scale_factor'])  # multiplier par 0.02
                add_offset = float(lst_metadata['add_offset'])
            else:
                scale_factor = float(0.02)
                add_offset = float(0)

            # conversion en Kelvin, puis en Celsius
            kelvin_array = np.add(np.multiply(modis_array, scale_factor), add_offset)
            lst_celsius_array = np.subtract(kelvin_array, 273.15)

            lst_image_1km.save_band(lst_celsius_array, r'secteur3/MODIS_1km_Celsius.tif')

            # Calcul des résidus à 1km
            residus_1km = ma.subtract(lst_celsius_array, resampled_predicted_100m_image.getArray())

            # Sauvegarder l'image 1km pour pouvoir la ramener à 100m par la suite
            residus_1km_masked = ma.filled(residus_1km, np.nan)  # on retire les valeurs masquées du output

            lst_image_1km.save_band(residus_1km_masked, r'secteur3/residus_1km.tif')

            # Résidus de 1km -> 100m
            residus_1km_load = Image(r'secteur3/residus_1km.tif')
            residus_1km_load.reproject(r'secteur3/residus_1km.tif', r'secteur3/residus_100m.tif', 'EPSG:32618', 'np.nan',
                                       '100.0', 'cubic')

            # Application des résidus sur le résultat
            residus_100m = Image(r'secteur3/residus_100m.tif')

            predicted_100m_image_with_residuals = ma.add(predicted_100m_image.getArray(), residus_100m.getArray())

            # Sauvegarder le résultat
            residus_100m.save_band(predicted_100m_image_with_residuals, outputFile_withResidualCorrection)


def main():
    """ Tests de la classe et de ses méthodes.
        ****** Exécuter ce fichier pour effectuer le downscaling ******
    """

    # secteur1
    """
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

    secteur1 = Secteur(modis, landsat, aster)
    secteur1.prepareData()

    rfr = ReductionEchelle(secteur1)
    rfr.applyDownscaling()
    """

    # *********** TEST FONCTIONNEL **************

    # secteur3
    b1 = r'secteur3/landsat8/CU_LC08.001_SRB1_doy2020133_aid0001.tif'
    b2 = r'secteur3/landsat8/CU_LC08.001_SRB2_doy2020133_aid0001.tif'
    b3 = r'secteur3/landsat8/CU_LC08.001_SRB3_doy2020133_aid0001.tif'
    b4 = r'secteur3/landsat8/CU_LC08.001_SRB4_doy2020133_aid0001.tif'
    b5 = r'secteur3/landsat8/CU_LC08.001_SRB5_doy2020133_aid0001.tif'
    b6 = r'secteur3/landsat8/CU_LC08.001_SRB6_doy2020133_aid0001.tif'
    b7 = r'secteur3/landsat8/CU_LC08.001_SRB7_doy2020133_aid0001.tif'
    qa = r'secteur3/landsat8/CU_LC08.001_PIXELQA_doy2020133_aid0001.tif'
    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa)

    lst = r'secteur3/modis/MOD11A1.006_LST_Day_1km_doy2020133_aid0001.tif'
    qa = r'secteur3/modis/MOD11A1.006_QC_Day_doy2020133_aid0001.tif'
    modis = Modis(lst, qa)

    # reprojection de l'image MODIS de départ en UTM18
    modis.reprojectModisSystem('EPSG:32618', '-9999.0', '1000.0', 'average')

    mnt = r'secteur3/aster/ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif'
    qa = r'secteur3/aster/ASTGTM_NUMNC.003_ASTER_GDEM_NUM_doy2000061_aid0001.tif'
    aster = Aster(mnt, qa)

    secteur3 = Secteur(modis, landsat, aster)
    secteur3.prepareData(train_model=True)

    rfr = ReductionEchelle(secteur3)

    # options fournies:
    #predictors = ['NDVI', 'NDWI', 'NDBI', 'MNDWI', 'SAVI', 'Albedo', 'BSI', 'UI', 'EVI', 'IBI', 'B1', 'B2', 'B3',
    #              'B4', 'B5', 'B6', 'B7', 'MNT', 'Pente']

    #predictors = ['NDVI', 'NDWI', 'NDBI', 'MNDWI', 'SAVI', 'Albedo', 'BSI', 'UI', 'EVI', 'IBI', 'MNT']
    #predictors = ['MNT', 'B7', 'B6', 'B1', 'B2', 'BSI', 'MNDWI']  ## Not bad!
    predictors = ['NDVI', 'MNT']
    #predictors = ['MNT', 'Albedo', 'MNDWI', 'BSI', 'B1', 'B2']
    #predictors = ['NDVI', 'NDWI', 'MNT']
    rfr.applyDownscaling(predictors, outputFile=r'secteur3/MODIS_predit_100m.tif',
                                     residualCorrection=True,
                                     outputFile_withResidualCorrection= r'secteur3/MODIS_predit_100m_avec_residus.tif')


if __name__ == '__main__':
    main()
