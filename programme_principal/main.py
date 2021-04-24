from landsat import Landsat
from modis import Modis
from aster import Aster
from ReductionEchelle import ReductionEchelle
from secteur import Secteur
import os


def main():
    #######################################################################
    ###################### paramètres du programme ########################
    #######################################################################

    # Instructions :
    # - Les données doivent toutes être directement dans le même dossier
    # - Modifier seulement les paramètres dans l'entête de la fonction main()

    # données Landsat
    b1 = r'data/CU_LC08.001_SRB1_doy2020133_aid0001.tif'
    b2 = r'data/CU_LC08.001_SRB2_doy2020133_aid0001.tif'
    b3 = r'data/CU_LC08.001_SRB3_doy2020133_aid0001.tif'
    b4 = r'data/CU_LC08.001_SRB4_doy2020133_aid0001.tif'
    b5 = r'data/CU_LC08.001_SRB5_doy2020133_aid0001.tif'
    b6 = r'data/CU_LC08.001_SRB6_doy2020133_aid0001.tif'
    b7 = r'data/CU_LC08.001_SRB7_doy2020133_aid0001.tif'
    qa = r'data/CU_LC08.001_PIXELQA_doy2020133_aid0001.tif'

    # source de données (options possibles : "appeears", "earthdata")
    src = "appeears"
    
    # données Modis
    lst = r'data/MOD11A1.006_LST_Day_1km_doy2020133_aid0001.tif'
    qc = r'data/MOD11A1.006_QC_Day_doy2020133_aid0001.tif'
    
    # données Aster
    dem = r'data/ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif'
    num = r'data/ASTGTM_NUMNC.003_ASTER_GDEM_NUM_doy2000061_aid0001.tif'

    # choix de prédicteurs sous forme de liste
    # options possibles : 'NDVI', 'NDWI', 'NDBI', 'MNDWI', 'SAVI', 'Albedo', 'BSI', 'UI', 'EVI', 'IBI', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'MNT', 'Pente', 'Orientation'
    predictors = ['NDWI', 'Pente', 'Orientation']

    # paramètre pour la résolution à laquelle on veut effectuer la réduction d'échelle (options possibles : 30, 100)
    target_resolution = 100

    # paramètre pour l'application des résidus (options possibles: True, False)
    residualCorrection = True

    # paramètre pour la suppresion des fichiers temporaire (options possibles: True, False)
    supprimer_les_fichiers_temporaires = True

    #######################################################################
    ######################### fin des paramètres ##########################
    #######################################################################


    # ne plus modifier le code à partir de cette ligne


    # la chaîne de traitements débute ici

    # construction de l'objet Landsat
    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa, src)

    # construction de l'objet Modis
    modis = Modis(lst, qc)

    # construction de l'objet Aster
    aster = Aster(dem, num)

    # reprojection de l'image MODIS de départ en UTM18 (EPSG:32618)
    modis.reprojectModisSystem('EPSG:32618', '-9999.0', '1000.0', 'average')

    # construction de l'objet Secteur avec les images Modis, Landsat et Aster
    secteur = Secteur(modis, landsat, aster)
    secteur.prepareData(train_model=True)  # préparation des données pour l'entraînement du modèle de Random Forest

    rfr = ReductionEchelle(secteur)  # construction de l'objet ReductionEchelle à partir du secteur

    # application de la réduction d'échelle avec la méthode Random Forest Regression
    rfr.applyDownscaling(predictors, outputFile=r'data/MODIS_predit_100m.tif',
                                     residualCorrection=residualCorrection,
                                     outputFile_withResidualCorrection= r'data/MODIS_predit_100m_avec_residus.tif',
                                     targetResolution=target_resolution)

    # supprimer tous les fichiers temporaires si on a choisi cette option
    if supprimer_les_fichiers_temporaires:
        delete_temp()


def delete_temp():
    """ Fonction permettant de supprimer les fichiers temporaires qui contiennent un des mots-clés listés ci-dessous dans
        le dossier de travail (sous-dossier 'data' où les données doivent se retrouver).
    """
    root = 'data'
    for f in os.listdir(root):
        if 'reproj' in f or 'subdivided' in f or 'Celsius' in f or 'residus_' in f or 'masked' in f \
                or 'aspect' in f or 'slope' in f or 'bit_raster' in f or 'pourcentagenuage' in f:
            os.remove(os.path.join(root, f))


if __name__ == '__main__':
    main()  # exécution de la fonction main()

