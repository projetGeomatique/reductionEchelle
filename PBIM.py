import numpy as np
import rasterio as rasterio
from osgeo import gdal, gdal_array
from modis import Modis
from image import Image
from landsat import Landsat
import numpy.ma as ma
from secteur import Secteur
from aster import Aster
import pandas as pd


class PBIM():
    def __init__(self, modis_image, landsat_image):
        self.modis_image = modis_image
        self.landsat_image = landsat_image

    def calcEmissivityFVC(self):

        # get ndvi array from landsat image
        ndvi_image = self.landsat_image.getNdvi()

        # calculate min and max ndvi value in whole image
        ndvi_min = np.amin(ndvi_image)
        ndvi_max = np.amax(ndvi_image)

        fvc_image = np.copy(ndvi_image)  # fvc = fractional vegetation cover

        # calculate fractional vegetation cover from ndvi
        for i in range(self.landsat_image.ysize):
            for j in range(self.landsat_image.xsize):
                temp_value = (ndvi_image[i][j] - ndvi_min)/(ndvi_max - ndvi_min)
                fvc_image[i][j] = np.power(temp_value, 2)

        # calculate and return emissivity array from fvc array
        return np.add(np.multiply(0.98, np.subtract(1, fvc_image)), np.multiply(0.93, fvc_image))

    def calcEmissivitySobrino(self):

        # get ndvi array from landsat image
        ndvi_image = self.landsat_image.getNdvi()

        # calculate min and max ndvi value in whole image
        ndvi_min = np.amin(ndvi_image)
        ndvi_max = np.amax(ndvi_image)

        # copy ndvi array as template for emissivity array
        emissivity_image = np.copy(ndvi_image)

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

                    temp_value = (ndvi_image[i][j] - ndvi_min) / (ndvi_max - ndvi_min) # ndvi_min = 0.2 & ndvi_max = 0.5 ?
                    fvc = np.power(temp_value, 2)

                    # valeurs moyennes suggérées dans l'article
                    emis_vegetation = 0.985
                    emis_sol = 0.96

                    #F = 0.55
                    #d_emis = (1-emis_sol)*(1-fvc)*F*emis_vegetation

                    #emissivity = emis_vegetation*fvc + emis_sol*(1-fvc) + d_emis   # d_emis nécessaire si la surface est hétérogène et rugueuse
                    emissivity = emis_vegetation * fvc + emis_sol * (1 - fvc)

                    emissivity_image[i][j] = emissivity

                j = j + 1
            i = i + 1

        return emissivity_image

    def calcMeanEmissivity(self, emissivity_image_100m, mean_emissivity_output):

        # calculate averaged emissivity for all high resolution emissivity pixels included in a low-res LST pixel
        #highEmissivity = self.calcEmissivitySobrino()

        # resample to 1km (with average algorithm)
        emissivity_image = Image(emissivity_image_100m)
        emissivity_image.reproject(emissivity_image_100m, emissivity_image_100m.replace(".tif", "_reproject.tif"),
                                   'EPSG:32618', '-9999.0', '1000', 'average')

        emissivity_image.reproject(emissivity_image_100m.replace(".tif", "_reproject.tif"),
                                   mean_emissivity_output, 'EPSG:32618', '0.0', '100', 'near')

    def applyDownscaling(self, emissivity_image_100m, mean_emissivity_100m):

        #emissivity_image = self.calcEmissivitySobrino()

        # ******** MODIS LST (1km) ***********

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

        # apply PBIM formula (T_high = T_low * emissivity_high / emissivity_avg) for each pixel

        # ********** Émissivité (100m) *********
        emissivity = Image(emissivity_image_100m).getArray(masked=False)

        mean_emissivity = Image(mean_emissivity_100m).getArray(masked=False)

        # PBIM formula
        t_high = ma.divide(ma.multiply(lst_celsius_array,emissivity),mean_emissivity)

        Image(emissivity_image_100m).save_band(t_high, r'secteur3/PBIM_100m_result.tif')


if __name__ == "__main__":

    # load landsat image (bands 1-7 + qa)
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

    # subdivise en pixels de 100m
    modis.subdividePixel(10, "file", modis.lst.split(".")[0] + '_subdivided_100m.tif')

    mnt = r'secteur3/aster/ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif'
    qa = r'secteur3/aster/ASTGTM_NUMNC.003_ASTER_GDEM_NUM_doy2000061_aid0001.tif'
    aster = Aster(mnt, qa)

    #secteur3 = Secteur(modis, landsat, aster)  # on devrait peut-être faire un constructeur pour secteur qui ne
                                               # nécessite pas ASTER (vu que c'est pas nécessaire du tout ici)

    #secteur3.prepareData()

    # images landsat à 100m
    landsat.reprojectLandsat(modis.lst.split(".")[0] + '_subdivided_100m.tif', False)

    pbim = PBIM(modis, landsat)

    emissivity = pbim.calcEmissivitySobrino()

    # sauvegarder le résultat avec une autre image (de mêmes dimensions et avec la même référence spatiale)
    # comme référence
    reference_image = Image(landsat.b4)  # Landsat B4 reprojetée à 100m
    reference_image.save_band(emissivity, r'secteur3/PBIM_emissivity_Sobrino.tif')

    pbim.calcMeanEmissivity(r'secteur3/PBIM_emissivity_Sobrino.tif', r'secteur3/PBIM_mean_emissivity_100m.tif')

    pbim.applyDownscaling(r'secteur3/PBIM_emissivity_Sobrino.tif', r'secteur3/PBIM_mean_emissivity_100m.tif')








    """
    # test
    test = np.random.randint(1, 10, size=(10,10))
    print(test)

    print(test[0:5,0:5])


    #print(window_mean)
    """






