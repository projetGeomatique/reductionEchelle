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
            # reprojection de l'image Landsat et ré-échantillonnage à 100m et masquage des nuages
            self.landsat_image.maskClouds30m()

            # Masquage de Landsat reechantillonee selon le pourcentage de nuages a l'interieur d'un pixel de 1000m

            imageLandsatQa = Image(self.landsat_image.qa)
            pourcentageNuage = imageLandsatQa.cloudOverlay(self.modis_image.lst)

            self.landsat_image.reprojectLandsat(self.modis_image.lst)

            self.landsat_image.maskClouds1000m(pourcentageNuage)
            
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
            self.landsat_image.reprojectLandsat(self.modis_image.lst.split(".")[0] + '_subdivided_100m.tif', reduce_zone=False)

            # reprojection de l'image Aster pour avoir la même taille que celle de Landsat préalablement reprojetée
            self.aster_image.reprojectAster(self.modis_image.lst.split(".")[0] + '_subdivided_100m.tif', reduce_zone=False)

            # reprojection de l'image MODIS pour avoir la même taille que celle de Landsat préalablement reprojetée
            self.modis_image.reprojectModis(self.modis_image.lst.split(".")[0] + '_subdivided_100m.tif', reduce_zone=False)

    def getDf(self, predictors, train=True):
        """ Aller chercher le DataFrame contenant l'ensemble des prédicteurs préparés pour le downscaling.
                Args:
                    predictors (list): Liste de string des prédicteurs à inclure dans l'entraînement ou la prédiction
                                       de la réduction d'échelle. Cet argument doit prendre la forme suivante:
                                       ['NDVI', 'NDWI', 'NDBI', 'MNT', 'Pente']
                                       avec des prédicteurs disponibles dans cette méthode.
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

        predictors_dict = {}

        # Calcul des prédicteurs (masquage des valeurs nulles inclus dans les get)
        # Landsat
        if 'NDVI' in predictors:
            ndvi = self.landsat_image.getNdvi()
            predictors_dict['NDVI'] = ndvi

        if 'NDWI' in predictors:
            ndwi = self.landsat_image.getNdwi()
            predictors_dict['NDWI'] = ndwi

        if 'NDBI' in predictors:
            ndbi = self.landsat_image.getNdbi()
            predictors_dict['NDBI'] = ndbi

        if 'MNDWI' in predictors:
            mndwi = self.landsat_image.getMndwi()
            predictors_dict['MNDWI'] = mndwi

        if 'SAVI' in predictors:
            savi = self.landsat_image.getSAVI()
            predictors_dict['SAVI'] = savi

        if 'Albedo' in predictors:
            albedo = self.landsat_image.getAlbedo()
            predictors_dict['Albedo'] = albedo

        if 'BSI' in predictors:
            bsi = self.landsat_image.getBSI()
            predictors_dict['BSI'] = bsi

        if 'UI' in predictors:
            ui = self.landsat_image.getUI()
            predictors_dict['UI'] = ui

        if 'EVI' in predictors:
            evi = self.landsat_image.getEVI()
            predictors_dict['EVI'] = evi

        if 'IBI' in predictors:
            ibi = self.landsat_image.getIBI()
            predictors_dict['IBI'] = ibi

        if 'B1' in predictors:
            b1 = self.landsat_image.getBand(1)
            predictors_dict['B1'] = b1

        if 'B2' in predictors:
            b2 = self.landsat_image.getBand(2)
            predictors_dict['B2'] = b2

        if 'B3' in predictors:
            b3 = self.landsat_image.getBand(3)
            predictors_dict['B3'] = b3

        if 'B4' in predictors:
            b4 = self.landsat_image.getBand(4)
            predictors_dict['B4'] = b4

        if 'B5' in predictors:
            b5 = self.landsat_image.getBand(5)
            predictors_dict['B5'] = b5

        if 'B6' in predictors:
            b6 = self.landsat_image.getBand(6)
            predictors_dict['B6'] = b6

        if 'B7' in predictors:
            b7 = self.landsat_image.getBand(7)
            predictors_dict['B7'] = b7

        # Aster
        if 'MNT' in predictors:
            mnt = self.aster_image.getMNT()
            predictors_dict['MNT'] = mnt

        if 'Pente' in predictors:
            pente = self.aster_image.getPente()
            predictors_dict['Pente'] = pente

        # Reshape les array en colonnes pour tous les prédicteurs
        for predictor in predictors_dict:
            predictors_dict[predictor] = (predictors_dict[predictor]).reshape(-1, 1)

        # ******** MODIS LST ***********

        # Conversion du array MODIS LST en un format compatible au Random Forest Regression
        lst_image = Image(self.modis_image.lst)

        modis_array = lst_image.getArray(masked=True, lower_valid_range=7500, upper_valid_range=65535)

        # convertir à des températures de surface en Celsius
        lst_metadata = lst_image.getMetadata()

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

        # Reshape en une colonne
        lst_celsius_array = lst_celsius_array.reshape(-1, 1)

        # appliquer les mêmes masques partout (si un pixel est masqué dans une image, il doit être masqué pour toutes
        # les images)
        masks = []  # ensemble des masques à remplir

        for predictor in predictors_dict:
            masks.append(ma.getmaskarray(predictors_dict[predictor]))

        masks.append(ma.getmaskarray(lst_celsius_array))

        # masque unique pour tous les jeux de données
        mask = sum(masks)  # somme de masques: toutes les valeurs qui ne sont pas égales à 0 corresponded à une position
                           # à masquer
        self.mask = mask

        # masquage identique de tous les jeux de données
        for predictor in predictors_dict:
            predictors_dict[predictor] = ma.masked_array(predictors_dict[predictor], mask)

        lst_celsius_array = ma.masked_array(lst_celsius_array, mask)

        # ********* Stack les colonnes ensemble **********

        # extraire les noms des prédicteurs et les array associés
        predictor_names = []
        predictor_values = []

        for predictor in predictors_dict:
            predictor_names.append(predictor)
            predictor_values.append(predictors_dict[predictor])

        # construction d'une matrice vide de dimensions (nombre_de_pixels, nombre_de_predicteurs + 1)
        if train:
            col_stack = ma.empty([predictor_values[0].shape[0], len(predictor_values) + 1])  # +1 = colonne pour LST
        else:
            col_stack = np.empty([predictor_values[0].shape[0], len(predictor_values) + 1])  # +1 = colonne pour LST

        # ajout des prédicteurs dans les colonnes (de la première jusqu'à l'avant-dernière)
        for i in range(0, len(predictor_values)):
            col_stack[:, [i]] = predictor_values[i]

        col_stack[:, [-1]] = lst_celsius_array  # ajout de la LST dans la dernière colonne
        predictor_names.append('LST')

        # Création d'un dataframe Pandas qui contient la matrice construite précédemment et qui indique les labels de
        # chacune des colonnes
        df = pd.DataFrame(col_stack, columns=predictor_names)  # dataframe ne gère pas les masques?

        #df_test = df['LST']
        #print(df_test[162])

        # changer pour que l'utilisateur puisse choisir où sauvegarder (et/ou s'il veut sauvegarder les CSV)
        if train:
            # Sauvegarder en CSV pour visualisation et vérification plus tard
            df.to_csv(r'data/donnees_utilisees_train.csv', index=False)
            print("DataFrame pour l'entraînement sauvegardé à: secteur3/donnees_utilisees_train.csv")
        else:
            df.to_csv(r'data/donnees_utilisees_predict.csv', index=False)
            print("DataFrame pour la prédiction sauvegardé à: secteur3/donnees_utilisees_predict.csv")

        return df


def main():
    """ Tests de la classe et de ses méthodes.
    """

    # secteur1
    b1 = r'27juin2019/CU_LC08.001_SRB1_doy2019178_aid0001.tif'
    b2 = r'27juin2019/CU_LC08.001_SRB2_doy2019178_aid0001.tif'
    b3 = r'27juin2019/CU_LC08.001_SRB3_doy2019178_aid0001.tif'
    b4 = r'27juin2019/CU_LC08.001_SRB4_doy2019178_aid0001.tif'
    b5 = r'27juin2019/CU_LC08.001_SRB5_doy2019178_aid0001.tif'
    b6 = r'27juin2019/CU_LC08.001_SRB6_doy2019178_aid0001.tif'
    b7 = r'27juin2019/CU_LC08.001_SRB7_doy2019178_aid0001.tif'
    qa = r'27juin2019/CU_LC08.001_PIXELQA_doy2019178_aid0001.tif'
    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa)

    lst = r'27juin2019/MOD11A1.006_LST_Day_1km_doy2019178_aid0001.tif'
    qa = r'27juin2019/MOD11A1.006_QC_Day_doy2019178_aid0001.tif'
    # test)
    modis = Modis(lst, qa)

    mnt = r'data/ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif'
    qa = r'data/ASTGTM_NUMNC.003_ASTER_GDEM_NUM_doy2000061_aid0001.tif'  # aussi un test (ne semble pas valide)

    aster = Aster(mnt, qa)

    rfr = Secteur(modis, landsat, aster)
    rfr.prepareData()

    predictors = ['NDVI', 'NDWI', 'NDBI', 'B1']
    df = rfr.getDf(predictors)

    print(df)
    print(df.drop('LST', axis=1))
    print(df['LST'])

    a = df['LST']
    print(a.ravel())


if __name__ == '__main__':
    main()
