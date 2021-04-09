import tkinter as tk
from tkinter import filedialog
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import numpy.ma as ma
from image import Image
from landsat import Landsat
from modis import Modis
from aster import Aster
from ReductionEchelle import ReductionEchelle
from secteur import Secteur
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    #######################################################################
    ###################### paramètres du programme ########################
    #######################################################################

    # Instructions :
    # - Les données doivent tous être directement dans le même dossiers
    # - Modifier seulement les paramètres dans l'entête de la fonction main()


    # données Landsat
    b1 = "data/LC08_L1TP_014028_20200706_20200721_01_T1_B1.TIF"
    b2 = "data/LC08_L1TP_014028_20200706_20200721_01_T1_B2.TIF"
    b3 = "data/LC08_L1TP_014028_20200706_20200721_01_T1_B3.TIF"
    b4 = "data/LC08_L1TP_014028_20200706_20200721_01_T1_B4.TIF"
    b5 = "data/LC08_L1TP_014028_20200706_20200721_01_T1_B5.TIF"
    b6 = "data/LC08_L1TP_014028_20200706_20200721_01_T1_B6.TIF"
    b7 = "data/LC08_L1TP_014028_20200706_20200721_01_T1_B7.TIF"
    qa = "data/LC08_L1TP_014028_20200706_20200721_01_T1_BQA.TIF"
    # source de données
    #options possibles : "appeears", "earthdata"
    src = "earthdata"
    
    # données Modis
    lst = r'data/MOD11_L2.clipped_test2.tif'
    qc = r'data/MOD11A1.006_QC_Day_doy2020133_aid0001.tif'
    
    # données Aster
    dem = r'data/ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif'
    num = r'data/ASTGTM_NUMNC.003_ASTER_GDEM_NUM_doy2000061_aid0001.tif'

    # predicteurs sous forme de liste
    # options possibles : 'NDVI', 'NDWI', 'NDBI', 'MNDWI', 'SAVI', 'Albedo', 'BSI', 'UI', 'EVI', 'IBI', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'MNT', 'Pente'
    predictors = ['MNT', 'NDVI']

    # paramètre pour la résolution à laquelle on veut effectuer la réduction d'échelle
    # options possibles : 30, 100
    target_resolution = 30

    # paramètre pour l'application des résidus
    residualCorrection = True

    # paramètre pour la suppresion des fichiers temporaire
    supprimer_les_fichiers_temporaires = True

    #######################################################################
    ######################### fin des paramètres ##########################
    #######################################################################


    # ne plus modifier




    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa, src)
    modis = Modis(lst, qc)
    aster = Aster(dem, num)
    # reprojection de l'image MODIS de départ en UTM18
    modis.reprojectModisSystem('EPSG:32618', 'np.nan', '1000.0', 'average')

    secteur = Secteur(modis, landsat, aster)
    secteur.prepareData(train_model=True)

    rfr = ReductionEchelle(secteur)

    rfr.applyDownscaling(predictors, outputFile=r'data/MODIS_predit_30m.tif',
                                     residualCorrection=residualCorrection,
                                     outputFile_withResidualCorrection= r'data/MODIS_predit_30m_avec_residus.tif',
                                     targetResolution=target_resolution)

    if supprimer_les_fichiers_temporaires:
        delete_temp()


def delete_temp():
    root = 'data'
    for f in os.listdir(root):
        if 'reproj' in f or 'subdivided' in f or 'Celsius' in f or 'residus_' in f or 'masked' in f:
            os.remove(os.path.join(root, f))


if __name__ == '__main__':
    main()
