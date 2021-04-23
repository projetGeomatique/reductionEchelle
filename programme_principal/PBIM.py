import numpy as np
from modis import Modis
from image import Image
from landsat import Landsat
import numpy.ma as ma


class PBIM():
    """ Classe regroupant les méthodes nécessaires pour effectuer la réduction d'échelle à l'aide de l'algorithme
        PBIM (Pixel Block Intensity Modulation).

        Attributes:
            modis_image (Modis): Images MODIS utilisées pour la réduction d'échelle
            landsat_image (Landsat): Images Landsat utilisées pour la réduction d'échelle
    """
    def __init__(self, modis_image, landsat_image):
        self.modis_image = modis_image
        self.landsat_image = landsat_image

    def calcEmissivityFVC(self):
        """ Méthode permettant de calculer l'émissivité effective selon la fraction de couverture végétale.
                Returns:
                    array (Numpy.array): Array des valeurs d'émissivité effective pour la collection d'images.
        """
        # récupérer le NDVI à partir de la méthode getNdvi de la classe Landsat
        ndvi_image = self.landsat_image.getNdvi()

        # calculer les valeurs minimales et maximales de NDVI pour toute l'image
        ndvi_min = np.amin(ndvi_image)
        ndvi_max = np.amax(ndvi_image)

        # copie de l'image de NDVI pour venir remplir les valeurs de fraction de couverture végétale
        fvc_image = np.copy(ndvi_image)  # fvc = fractional vegetation cover

        # calculer la fraction de couverture végétale à partir du NDVI
        for i in range(self.landsat_image.ysize):
            for j in range(self.landsat_image.xsize):
                temp_value = (ndvi_image[i][j] - ndvi_min)/(ndvi_max - ndvi_min)
                fvc_image[i][j] = np.power(temp_value, 2)  # remplir dans l'image fvc_image

        # calculer l'émissivité à partir de la fraction de couverture végétale
        return np.add(np.multiply(0.98, np.subtract(1, fvc_image)), np.multiply(0.93, fvc_image))

    def calcEmissivitySobrino(self):
        """ Méthode permettant de calculer l'émissivité effective selon la méthode de Sobrino.
                Returns:
                    array (Numpy.array): Array des valeurs d'émissivité effective pour la collection d'images.
        """
        # récupérer le NDVI à partir de la méthode getNdvi de la classe Landsat
        ndvi_image = self.landsat_image.getNdvi()

        # calculer les valeurs minimales et maximales de NDVI pour toute l'image
        ndvi_min = np.amin(ndvi_image)
        ndvi_max = np.amax(ndvi_image)

        # copie de l'image de NDVI pour venir remplir les valeurs d'émissivité
        emissivity_image = np.copy(ndvi_image)

        # remplir les valeurs d'émissivité effective dans emissivity_image
        i = 0
        while i < ndvi_image.shape[0]:
            j = 0
            while j < ndvi_image.shape[1]:

                # NDVI < 0.2 = sol (émissivité = valeur de réflectance dans la bande rouge)
                if ndvi_image[i][j] < 0.2:
                    emissivity_image[i][j] = self.landsat_image.getBand(4)[i][j]

                # NDVI > 0.5 = couverture végétale totale (émissivité = 0.99)
                elif ndvi_image[i][j] > 0.5:
                    emissivity_image[i][j] = 0.99

                # 0.2 <= NDVI <= 0.5 = composition mixte de sol nu et de couverture végétale
                else:

                    ndvi_min = 0.2  # peut utiliser les valeur min/max de l'image telle quelle ?
                    ndvi_max = 0.5

                    temp_value = (ndvi_image[i][j] - ndvi_min) / (ndvi_max - ndvi_min)
                    fvc = np.power(temp_value, 2)

                    # valeurs moyennes suggérées dans l'article
                    emis_vegetation = 0.985
                    emis_sol = 0.96

                    #F = 0.55
                    #d_emis = (1-emis_sol)*(1-fvc)*F*emis_vegetation

                    #emissivity = emis_vegetation*fvc + emis_sol*(1-fvc) + d_emis  # d_emis nécessaire si la surface est hétérogène et rugueuse
                    emissivity = emis_vegetation * fvc + emis_sol * (1 - fvc)

                    emissivity_image[i][j] = emissivity

                j = j + 1
            i = i + 1

        return emissivity_image

    def calcMeanEmissivity(self, emissivity_image_100m, mean_emissivity_output):
        """ Méthode permettant de calculer l'émissivité moyenne selon une fenêtre de 10x10 (pixels à 100m jusqu'à des
            pixels à 1000m). Le résultat est stocké dans le fichier de la variable 'mean_emissivity_output'.
                Args:
                    emissivity_image_100m (str): Path vers le fichier d'émissivité effective à 100m.
                    mean_emissivity_output (str): Path vers le fichier d'émissivité moyenne calculé à 1000m, puis ramené
                                                  à 100m.
        """
        # rééchantillonnage de l'image d'émissivité de 100m à 1000m avec l'algorithme de moyenne
        emissivity_image = Image(emissivity_image_100m)
        emissivity_image.reproject(emissivity_image_100m, emissivity_image_100m.replace(".tif", "_reproject.tif"),
                                   'EPSG:32618', '-9999.0', '1000', 'average')

        # rééchantillonnage (subdivision d'un pixel de 1000m avec la moyenne d'émissivité à 100m en conservant les mêmes
        # valeurs d'émissivité moyenne)
        emissivity_image.reproject(emissivity_image_100m.replace(".tif", "_reproject.tif"),
                                   mean_emissivity_output, 'EPSG:32618', '0.0', '100', 'near')

    def applyDownscaling(self, emissivity_image_100m, mean_emissivity_100m, pbim_result):
        """ Méthode permettant d'effectuer la réduction d'échelle avec la méthode PBIM.
                Args:
                    emissivity_image_100m (str): Path vers le fichier d'émissivité effective à 100m.
                    mean_emissivity_output (str): Path vers le fichier d'émissivité moyenne calculé à 1000m, puis ramené
                                                  à 100m.
                    pbim_result (str): Path vers le fichier à utiliser pour stocker le résultat de la réduction d'échelle
        """
        # ******** MODIS LST (1km) ***********

        # utiliser l'image Modis avec les pixels de 1km subdivisés à 100m (avec les mêmes valeurs de pixels)
        lst_image = Image(self.modis_image.lst.split(".")[0] + '_subdivided_100m.tif')

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

        # ********** Émissivité (100m) *********
        emissivity = Image(emissivity_image_100m).getArray(masked=False)

        mean_emissivity = Image(mean_emissivity_100m).getArray(masked=False)

        # PBIM formula (T_high = T_low * emissivity_high / emissivity_avg) for each pixel
        t_high = ma.divide(ma.multiply(lst_celsius_array, emissivity), mean_emissivity)

        # sauvegarder le résultat sous forme d'image
        Image(emissivity_image_100m).save_band(t_high, pbim_result)


if __name__ == "__main__":


    #######################################################################
    ###################### paramètres du programme ########################
    #######################################################################

    # Instructions :
    # - Les données doivent toutes être directement dans le même dossier
    # - Modifier seulement les paramètres indiqués ci-dessous

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

    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa, src)

    # données Modis
    lst = r'data/MOD11A1.006_LST_Day_1km_doy2020133_aid0001.tif'
    qa = r'data/MOD11A1.006_QC_Day_doy2020133_aid0001.tif'
    modis = Modis(lst, qa)

    # fichier pour sauvegarder l'émissivité calculée
    image_emissivite = r'data/PBIM_emissivity_Sobrino.tif'

    # fichier pour sauvegarder l'émissivité moyenne calculée
    image_emissivite_moyenne = r'data/PBIM_mean_emissivity_100m.tif'

    # fichier pour sauvegarder le résultat final de la réduction d'échelle
    resultat_downscaling_pbim = r'data/PBIM_100m_result.tif'

    #######################################################################
    ######################### fin des paramètres ##########################
    #######################################################################


    # ne plus modifier le code à partir de cette ligne




    # la chaîne de traitements débute ici

    # reprojection de l'image MODIS de départ en UTM18 (EPSG:32618)
    modis.reprojectModisSystem('EPSG:32618', '-9999.0', '1000.0', 'average')

    # subdivise en pixels de 100m
    modis.subdividePixel(10, "file", modis.lst.split(".")[0] + '_subdivided_100m.tif')

    # images landsat à 100m
    landsat.reprojectLandsat(modis.lst.split(".")[0] + '_subdivided_100m.tif', False)

    # construction d'un objet de la classe PBIM
    pbim = PBIM(modis, landsat)

    # calcul de l'émissivité effective
    emissivity = pbim.calcEmissivitySobrino()

    # sauvegarder le résultat avec une autre image (de mêmes dimensions et avec la même référence spatiale)
    # comme référence
    reference_image = Image(landsat.b4)  # Landsat B4 reprojetée à 100m
    reference_image.save_band(emissivity, image_emissivite)

    pbim.calcMeanEmissivity(image_emissivite, image_emissivite_moyenne)

    pbim.applyDownscaling(image_emissivite, image_emissivite_moyenne, resultat_downscaling_pbim)
