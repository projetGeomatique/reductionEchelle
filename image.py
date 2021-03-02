import gdal, gdalconst
import numpy as np
from osgeo import osr

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
    def reprojectMatch(self, referenceFile):
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

    """
    Fonction permettant de reprojeter l<image qui sera utilisee en reference. Dans ce cas precis, elle reprojete en UTM 
    zone 18
    """
    def reprojectUTM18(self):
        inputFile = self.filename
        input = gdal.Open(inputFile, 1)

        srs = osr.SpatialReference()
        srs.ImportFromEPSG(32618)
        srsWkt = srs.ExportToWkt()

        input.SetProjection(srsWkt)
        input.FlushCache()
        input = None

    """
    Fonction permettant de reprojeter l'image qui sera utilisée en référence dans reprojectMatch.
        outFilename: nom du fichier virtuel en sortie (sans l'extension)
        outCRS: système de référence dans lequel on souhaite projeter le jeu de données
        x_resolution: résolution en X souhaitée dans le résultat de la projection
        y_resolution: résolution en Y souhaitée dans le résultat de la projection
        resample_alg: algorithme de rééchantillonnage utilisé dans la reprojection (par défaut: plus proche voisin)
    """
    def reproject(self, outFilename, outCRS, x_resolution, y_resolution, resample_alg="near"):
        outVrt = outFilename + '.vrt'
        warp = gdal.Warp(outVrt, self.filename, format="vrt", dstSRS=outCRS, xRes=x_resolution, yRes=y_resolution,
                         options=gdal.WarpOptions(resampleAlg=resample_alg))
        warp = None

        self.filename = outVrt  # la classe Image pointe maintenant sur ce fichier reprojeté


def main():
    b1 = r'data/CU_LC08.001_SRB1_doy2020229_aid0001_test2.tif'
    image1 = Image(b1)
    #file = image1.reprojectMatch(r"MOD11A1.006_Clear_day_cov_doy2020229_aid0001.tif")
    #image1.reprojectUTM18()
    image1.reproject("reprojected", "EPSG:32618", 30, 30, resample_alg="average")

    #modis_image = r'data/MOD11A1.006_Clear_day_cov_doy2020229_aid0001_clipped.tif'
    #modis_image1 = Image(modis_image)
    #modis_image1.reproject("reprojected2", "EPSG:32618", 250, 250)

    #image1.reprojectMatch(modis_image1.filename)
    #print(file)


if __name__ == '__main__':
    main()