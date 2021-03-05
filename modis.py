from image import Image
import gdal
import os


class Modis:
    """ Classe modélisant une collection d'images MODIS. Plus précisément, elle modélise les collections MOD11A1 de
        température de surface MODIS. Cette classe est composée d'objets de type Image, mais ne les appelle pas dans
        son constructeur afin d'économiser de l'espace mémoire et minimiser le temps d'exécution.

        Attributes:
            lst (str): Path vers le fichier .tiff du produit de température de surface de la collection d'images MODIS
                       (MOD11A1)
            qa (str): Path vers le fichier .tiff de la bande de qualité de la collection d'images MODIS (MOD11A1)
    """
    def __init__(self, lst, qa):
        self.lst = lst
        self.qa = qa

    def setBands(self, lst, qa):
        """ Méthode permettant d'assigner des nouveaux fichiers aux bandes de la collection.
                Args:
                    lst (str): Nouveau path vers le fichier .tiff du produit de température de surface de la collection
                               d'images MODIS (MOD11A1)
                    qa (str): Nouveau path vers le fichier .tiff de la bande de qualité de la collection d'images MODIS
                              (MOD11A1)
        """
        self.lst = lst
        self.qa = qa

    def reprojectModis(self, referenceFile):
        """ Permet de reprojeter, découper, aligner et rééchantillonner une image à partir d'une image de référence.
            Cette méthode effectue ces traitements pour l'ensemble des bandes de la collection MODIS. Elle fait
            appel à la méthode reprojectMatch() de la classe Image.
                Args:
                    referenceFile (string): Path du fichier de l'image de référence à utiliser.
        """
        bandPaths = [self.lst, self.qa]

        newBandsPaths = []

        for band in bandPaths:
            image = Image(band)
            newBandsPaths.append(image.reprojectMatch(referenceFile))

        self.lst = newBandsPaths[0]
        self.qa = newBandsPaths[1]

        print("          Reprojection termine")

    def reprojectModisSystem(self, outCRS, noDataVal, resolution, resample_alg):
        """ Permet de reprojeter une image d'un système de référence à un autre. Cette méthode effectue ces traitements
            pour l'ensemble des bandes de la collection MODIS. Elle fait appel à la méthode reproject() de la classe
            Image.
            Exemples d'arguments: reprojectModisSystem('EPSG:32618', '-9999.0', '1000.0', 'average')
                Args:
                    outCRS (string): Code EPSG du système de référence auquel on souhaite reprojeter (ex: EPSG:31618).
                    noDataVal (string): Valeur à fixer pour les valeurs nulles.
                    resolution (string): Résolution spatiale (taille de pixel) voulue.
                    resample_alg (string): Algorithme de rééchantillonnage à utiliser (ex: average, near).
        """
        bandPaths = [self.lst, self.qa]
        newBandsPaths = []

        for band in bandPaths:
            image = Image(band)

            image.reproject(band, band.replace(".tif", "_reprojUTM18.tif"), outCRS, noDataVal, resolution, resample_alg)
            newBandsPaths.append(band.replace(".tif", "_reprojUTM18.tif"))

        self.lst = newBandsPaths[0]
        self.qa = newBandsPaths[1]

    def subdividePixel(self, subdivisions, outputType, outFile=""):
        """ Permet de subdiviser une image MODIS en un certain nombre de subdivisions pour chaque pixel. On conserve
            la même valeur de pixel pour chacune des subdivisions (utilisation de 'near' comme resampleAlg).
                Args:
                    subdivisions (int ou float): nombre de subdivisions voulues (ex: pour une valeur de 10, on subdivise
                                                 chaque pixel de 1km en 100 (10²) pixels de 100m.
                    outputType (string): type de output qu'on veut que la fonction produise. Si "array", produit un
                                         array numpy. Si "file" produit un fichier vrt qu'on peut réutiliser à l'externe
                                         de la classe.
                    outFile (string): nom du fichier à sauvegarder si on choisit l'option outputType = "file"
                Returns:
                    warped (Numpy.array): Array Numpy de l'image avec subdivisions de pixels (si l'option "array" est
                                          choisie pour le outputType)
        """

        # ******* à ajuster pour inclure le qa aussi *******
        lst_image = Image(self.lst)

        if outputType == "array":

            subdividedImage = gdal.Warp("warped.vrt", lst_image.filename, format="vrt",
                           options=gdal.WarpOptions(xRes=lst_image.gt[1] / subdivisions, yRes=-lst_image.gt[5] / subdivisions,
                                                    resampleAlg='near'))
            subdividedImage = None

            warpedImage = Image("warped.vrt")
            warped = warpedImage.getArray()
            os.remove("warped.vrt")

            return warped

        elif outputType == "file":
            subdividedImage = gdal.Warp(outFile, lst_image.filename, format="vrt",
                           options=gdal.WarpOptions(xRes=lst_image.gt[1] / subdivisions, yRes=-lst_image.gt[5] / subdivisions,
                                                    resampleAlg='near'))
            subdividedImage = None


def main():
    """ Tests de la classe et de ses méthodes.
    """
    lst = r'data/MOD11A1.006_Clear_day_cov_doy2020229_aid0001.tif'
    qa = r'data/MOD11A1.006_QC_Day_doy2020229_aid0001.tif'
    modis = Modis(lst, qa)

    print(modis.lst[0])

    print("!!!!!!!!!!!")

    warped = modis.subdividePixel(4, "array")

    print(warped[0])


if __name__ == '__main__':
    main()

