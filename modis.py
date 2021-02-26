from image import Image
import gdal
import os


class Modis(Image):
    def __init__(self, lst, qa):
        super().__init__(lst)
        self.lst = lst
        self.qa = qa

    def reprojectModis(self, referenceFile):
        bandPaths = [self.lst, self.qa]

        newBandsPaths = []

        for band in bandPaths:
            image = Image(band)
            newBandsPaths.append(image.reprojectImage(referenceFile))

        self.lst = newBandsPaths[0]
        self.qa = newBandsPaths[1]

        print("          Reprojection termine")

    def subdividePixel(self, subdivisions, outputType, outFile=""):
        """ Permet de subdiviser une image MODIS en un certain nombre de subdivisions pour chaque pixel. On conserve
            la même valeur de pixel pour chacune des subdivisions (utilisation de 'near' comme resampleAlg).
                Args:
                    subdivisions (int ou float): nombre de subdivisions voulues (ex: pour une valeur de 10, on subdivise
                    chaque pixel de 1km en 100 (10²) pixels de 100m.

                    outputType (string): type de output qu'on veut que la fonction produise. Si "array", produit un
                    array numpy. Si "file" produit un fichier vrt qu'on peut réutiliser à l'externe de la classe.

                    outFile (string): nom du fichier à sauvegarder si on choisit l'option outputType = "file"
        """

        # *** à ajuster pour inclure le qa aussi ***

        if outputType == "array":
            subdividedImage = gdal.Warp("warped.vrt", self.filename, format="vrt",
                           options=gdal.WarpOptions(xRes=self.gt[1] / subdivisions, yRes=-self.gt[5] / subdivisions,
                                                    resampleAlg='near'))
            subdividedImage = None
            warped = self.getArray("warped.vrt")
            os.remove("warped.vrt")

            return warped

        elif outputType == "file":
            subdividedImage = gdal.Warp(outFile, self.filename, format="vrt",
                           options=gdal.WarpOptions(xRes=self.gt[1] / subdivisions, yRes=-self.gt[5] / subdivisions,
                                                    resampleAlg='near'))
            subdividedImage = None


def main():
    lst = r'data/MOD11A1.006_Clear_day_cov_doy2020229_aid0001.tif'
    qa = r'data/MOD11A1.006_QC_Day_doy2020229_aid0001.tif'
    modis = Modis(lst, qa)

    print(modis.lst[0])

    print("!!!!!!!!!!!")

    warped = modis.subdividePixel(4, "array")

    print(warped[0])


if __name__ == '__main__':
    main()

