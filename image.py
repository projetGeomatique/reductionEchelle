import gdal, gdalconst
import numpy as np

class Image():
    def __init__(self, filename):
        self.filename = filename
        #self.__initialisation()
        dataset = gdal.Open(self.filename, gdal.GA_ReadOnly)

        if (not dataset):
            print("Erreur ouverture fichier")
        else:
            self.xsize = dataset.RasterXSize
            self.ysize = dataset.RasterXSize
            self.proj = dataset.GetProjection()
            self.gt = dataset.GetGeoTransform()


    def __initialisation(self):
        dataset = gdal.Open(self.filename, gdal.GA_ReadOnly)

        if (not dataset):
            print("Erreur ouverture fichier")
        else:
            self.xsize = dataset.RasterXSize
            self.ysize = dataset.RasterXSize
            self.proj = dataset.GetProjection()
            self.gt = dataset.GetGeoTransform()
            

    def getArray(self, filename):       
        return gdal.Open(filename).GetRasterBand(1).ReadAsArray()


    def save_band(self, array, filename):
        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        outds = driver.Create(filename, xsize=self.xsize, ysize=self.ysize, bands=1, eType=gdal.GDT_Float32)
        outds.SetGeoTransform(self.gt)
        outds.SetProjection(self.proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(array)
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()
        outband = None
        outds = None

    """
    referenceFile: path de l'image de reference sur laquelle la reprojection se basera
    Fonction permettant de reprojeter et reechantilloner une image a partir d'une image de reference
    """
    def reprojectImage(self, referenceFile):
        #Ouvrir l'image a reprojeter
        inputFile = self.filename
        input = gdal.Open(inputFile, gdalconst.GA_ReadOnly)

        #Ouvrir l'image de reference et obtenir sa projection ses parametres de transformation affine
        reference = gdal.Open(referenceFile, gdalconst.GA_ReadOnly)
        referenceProj = reference.GetProjection()
        referenceTrans = reference.GetGeoTransform()
        bandreference = reference.GetRasterBand(1)

        #Transformer les parametres de la transformation de tuples vers list afin de pouvoir les modifier
        #On additionne la resolution des pixels aux coordonnees du coin en haut a gauche du pixel en haut a gauche
        #afin d'avoir une zone de reference plus petite que la zone de l'input file
        referenceTrans = list(referenceTrans)
        referenceTrans[0] = referenceTrans[0] + referenceTrans[1]
        referenceTrans[3] = referenceTrans[3] + referenceTrans[5]
        referenceTrans = tuple(referenceTrans)
        x = reference.RasterXSize - 2
        y = reference.RasterYSize - 2

        #Creer le outputfile avec le format de l'image de reference
        outputfile = inputFile.replace(".tif", "_reproject.tif")
        driver = gdal.GetDriverByName('GTiff')
        output = driver.Create(outputfile, x, y, 1, bandreference.DataType)
        output.SetGeoTransform(referenceTrans)
        output.SetProjection(referenceProj)

        #Reprojeter l'image
        gdal.ReprojectImage(input, output, self.proj, referenceProj, gdalconst.GRA_Average)

        del output
        return outputfile


def main():
    b1 = r'data/CU_LC08.001_SRB1_doy2020229_aid0001.tif'
    image1 = Image(b1)
    file = image1.reprojectImage(r"MOD11A1.006_Clear_day_cov_doy2020229_aid0001.tif")
    print(file)


if __name__ == '__main__':
    main()