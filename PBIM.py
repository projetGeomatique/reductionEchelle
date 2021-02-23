import numpy as np
import rasterio as rasterio
from osgeo import gdal, gdal_array
from modis import Modis
from image import Image
from landsat import Landsat
import numpy.ma as ma


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
                    emissivity_image[i][j] = self.landsat_image.b4[i][j]

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

    def calcMeanEmissivity(self):

        # calculate averaged emissivity for all high resolution emissivity pixels included in a low-res LST pixel
        highEmissivity = self.calcEmissivityFVC()

        # how many emissivity pixels included in a LST pixel?
        # ..........



    def applyDownscaling(self, modis_lst, number_of_MODIS_subdivisions):

        emissivity_image = self.calcEmissivitySobrino()

        emissivity_avg = np.copy(emissivity_image)
        emissivity_avg = emissivity_avg.reshape(-1, 1)

        # copy emissivity array as template for T_high array
        t_high = np.copy(emissivity_image)
        t_high = t_high.reshape(-1, 1)

        shape_0 = emissivity_image.shape[0]  # number of lines
        shape_1 = emissivity_image.shape[1]  # number of columns

        emissivity_image = emissivity_image.reshape(-1, 1)

        emis_array = []

        i = 0
        while i < emissivity_image.shape[0]:
            j = 0

            while j < number_of_MODIS_subdivisions:
                emis_array.append(emissivity_image[i+j][0])
                j = j + 1

                if j == number_of_MODIS_subdivisions:
                    i = i + shape_1
                    j = 0

                if i > emissivity_image.shape[0]:
                    emis_ma_array = ma.masked_array(emis_array)
                    #emissivity_avg[i][j] = emis_ma_array.mean()  # placer la moyenne dans les 100 pixels parcourus précédemment




        # apply PBIM formula (T_high = T_low * emissivity_high / emissivity_avg) for each pixel






if __name__ == "__main__":

    # load landsat image (bands 1-7 + qa)
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

    #modis.subdividePixel(10, "file", r'test_data/MOD11A1.006_Clear_day_cov_doy2020229_aid0001_subdivided_100m.tif')
    #landsat.reprojectLandsat(r'test_data/MOD11A1.006_Clear_day_cov_doy2020229_aid0001_subdivided_100m.tif')
    landsat.reprojectLandsat(r'fonctionReprojectAjoutees/data/MOD11A1.006_Clear_day_cov_doy2020229_aid0001.tif')

    #pbim = PBIM(modis, landsat)


    #pbim.calcEmissivitySobrino()


    #