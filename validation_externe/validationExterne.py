from image import Image
from landsat import Landsat
from modis import Modis
from aster import Aster
from secteur import Secteur
from ReductionEchelle import ReductionEchelle
import numpy.ma as ma
import numpy as np
from sklearn import metrics
import os


def downscale():
    """ Permet d'effectuer la réduction d'échelle. Même principe que l'exécution à partir du fichier main pour le
        programme principal.

        On utilise MOD11_L2 et des images Landsat 8 provenant de EarthData qui ont été acquises à des heures très
        rapproches la même journée pour s'assurer d'avoir des températures de surface assez similaires (comparables).

        *** Il faut au préalable découper l'image MOD11_L2 pour qu'elle soit plus petite ou de même taille que les
            images Landsat et Aster afin que le découpage/alignement/rééchantillonnage de Landsat et Aster par rapport
            à l'image de référence Modis s'effectue correctement.
            Ce découpage peut être fait facilement dans QGIS selon l'étendue d'un shapefile ou une zone entrée
            manuellement.
        ***
    """

    # 19 mai 2020
    # données Landsat
    b1 = r'data/LC08_L1TP_014028_20200706_20200721_01_T1_B1.TIF'
    b2 = r'data/LC08_L1TP_014028_20200706_20200721_01_T1_B2.TIF'
    b3 = r'data/LC08_L1TP_014028_20200706_20200721_01_T1_B3.TIF'
    b4 = r'data/LC08_L1TP_014028_20200706_20200721_01_T1_B4.TIF'
    b5 = r'data/LC08_L1TP_014028_20200706_20200721_01_T1_B5.TIF'
    b6 = r'data/LC08_L1TP_014028_20200706_20200721_01_T1_B6.TIF'
    b7 = r'data/LC08_L1TP_014028_20200706_20200721_01_T1_B7.TIF'
    qa = r'data/LC08_L1TP_014028_20200706_20200721_01_T1_BQA.TIF'
    #src = "appeears"
    src = "earthdata"
    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa, src)

    # données Modis
    lst = r'data/MOD11_L2.clipped_test2.tif'  # image Modis découpée
    qa = r'data/MOD11A1.006_QC_Day_doy2020133_aid0001.tif'  # bande inutile pour la validation externe
    modis = Modis(lst, qa)

    # reprojection de l'image MODIS de départ en UTM18
    modis.reprojectModisSystem('EPSG:32618', 'np.nan', '1000.0', 'average')

    # données Aster
    mnt = r'data/ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif'
    qa = r'data/ASTGTM_NUMNC.003_ASTER_GDEM_NUM_doy2000061_aid0001.tif'
    aster = Aster(mnt, qa)

    # construction de l'objet Secteur
    secteur3 = Secteur(modis, landsat, aster)
    secteur3.prepareData(train_model=True)

    # construction de l'objet ReductionEchelle à partir du secteur
    rfr = ReductionEchelle(secteur3)

    # prédicteurs

    # choix de prédicteurs sous forme de liste
    # options possibles : 'NDVI', 'NDWI', 'NDBI', 'MNDWI', 'SAVI', 'Albedo', 'BSI', 'UI', 'EVI', 'IBI', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'MNT', 'Pente', 'Orientation'
    predictors = ['NDWI', 'NDWI', 'NDBI']

    # réduction d'échelle
    rfr.applyDownscaling(predictors, outputFile=r'data/MODIS_predit_100m.tif',
                                     residualCorrection=True,
                                     outputFile_withResidualCorrection= r'data/MODIS_predit_100m_avec_residus.tif',
                                     targetResolution=100)  # validation externe seulement possible à 100m


def validationExterne():
    """ Permet d'effectuer une validation externe entre l'image résultante de la réduction d'échelle et une image de
        température de surface calculée à partir des bandes 10 et 11 de Landsat 8 (disponibles sur EarthData).

        Les résultats de la validation externe sont des métriques de qualité en comparant les résultats de la réduction
        d'échelle à la température de surface calculée à 100m. Ces résultats sont présentés dans la console par des
        'print' (lignes 129 à 144).
    """

    # Match prediction result extent
    landsat_b10 = Image(r'data/LC08_L1TP_014028_20200706_20200721_01_T1_B10.TIF')
    landsat_b10.reprojectMatch(r'data/MOD11_L2.clipped_test2.tif'.split(".")[0] + '_subdivided_100m.tif', False)
    landsat_b10.setNewFile(landsat_b10.filename.replace(".TIF", "_reproject.tif"))

    # Get TOA radiance
    b10_array = landsat_b10.getArray(masked=True, lower_valid_range=1, upper_valid_range=65535)
    b10_array_radiance = ma.add(ma.multiply(b10_array, 0.00033420), 0.10000)

    # Get Brightness Temperature
    b10_array_brightness_temp = ( 1321.0789 / (ma.log((774.8853/b10_array_radiance) + 1)) ) - 273.15

    # Get NDVI
    landsat_b4 = Image(r'data/LC08_L1TP_014028_20200706_20200721_01_T1_B4_reproject.tif')
    b4_DN = landsat_b4.getArray(masked=True, lower_valid_range=1, upper_valid_range=65535)
    b4 = np.add(np.multiply(b4_DN, float(0.00002)), float(-0.10))

    landsat_b5 = Image(r'data/LC08_L1TP_014028_20200706_20200721_01_T1_B5_reproject.tif')
    b5_DN = landsat_b5.getArray(masked=True, lower_valid_range=1, upper_valid_range=65535)
    b5 = np.add(np.multiply(b5_DN, float(0.00002)), float(-0.10))

    ndvi = np.divide(np.subtract(b5, b4), np.add(b5, b4), where=((np.add(b5, b4)) != 0))

    # Get proportion of vegetation
    min_ndvi = ma.amin(ndvi)
    max_ndvi = ma.amax(ndvi)

    pv = ma.power(ma.divide(ma.subtract(ndvi, min_ndvi), (ma.subtract(max_ndvi, min_ndvi)), where=(ma.subtract(max_ndvi, min_ndvi))!=0), 2)

    # Get emissivity
    emissivity = 0.004 * pv + 0.986

    # Get Landsat 8 LST
    landsat_lst = b10_array_brightness_temp / (1 + (0.00115*b10_array_brightness_temp/1.4388) * ma.log(emissivity))

    # Save LST image for visualization
    landsat_b10.save_band(landsat_lst, r'data/landsat_lst.tif')

    # Validation between both arrays
    predicted_lst = ma.masked_invalid(Image(r'data/MODIS_predit_100m.tif').getArray())
    predicted_lst_with_residuals = ma.masked_invalid(Image(r'data/MODIS_predit_100m_avec_residus.tif').getArray())

    predicted_lst = ma.filled(predicted_lst, 0)
    predicted_lst_with_residuals = ma.filled(predicted_lst_with_residuals, 0)

    # Without residuals
    print('Without residual correction')
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(predicted_lst, landsat_lst))
    print('Mean Squared Error:', metrics.mean_squared_error(predicted_lst, landsat_lst))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(predicted_lst, landsat_lst)), "°C")
    print('Accuracy:', 100 - np.mean(100 * ((abs(predicted_lst - landsat_lst)) / landsat_lst)), "%")
    print('Explained variance score (EVS):', metrics.explained_variance_score(predicted_lst, landsat_lst))

    # With residuals
    print("\n")
    print('With residual correction')
    print('Mean Absolute Error (MAE):', metrics.mean_absolute_error(predicted_lst_with_residuals, landsat_lst))
    print('Mean Squared Error:', metrics.mean_squared_error(predicted_lst_with_residuals, landsat_lst))
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(predicted_lst_with_residuals, landsat_lst)), "°C")
    print('Accuracy:', 100 - np.mean(100 * ((abs(predicted_lst_with_residuals - landsat_lst)) / landsat_lst)), "%")
    print('Explained variance score (EVS):', metrics.explained_variance_score(predicted_lst_with_residuals, landsat_lst))


def delete_temp():
    """ Fonction permettant de supprimer les fichiers temporaires qui contiennent un des mots-clés listés ci-dessous dans
        le dossier de travail (sous-dossier 'data' où les données doivent se retrouver).
    """
    root = 'data'
    for f in os.listdir(root):
        if 'reproj' in f or 'subdivided' in f or 'Celsius' in f or 'residus_' in f or 'masked' in f:
            os.remove(os.path.join(root, f))


if __name__ == '__main__':
    # Exécution des fonctions précédentes en ordre pour effectuer la validation externe.
    downscale()
    validationExterne()
    delete_temp()
