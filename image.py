import gdal, gdalconst
import numpy as np
import numpy.ma as ma
from osgeo import osr
import os


class Image:
    """ Classe modélisant une image (généralement un fichier .tif)

        Attributes:
            filename (str): Path vers le fichier .tiff de l'image
            dataset (osgeo.gdal.Dataset): Le dataset GDAL de l'image
            xsize (int): Nombre de pixels en X de l'image
            ysize (int): Nombre de pixels en Y de l'image
            proj (string): Informations sur le système de référence et la projection de l'image géoréférencée
            gt (tuple): Paramètres de GeoTransform de l'image
    """
    def __init__(self, filename):
        self.filename = filename
        self.dataset = gdal.Open(self.filename, gdal.GA_ReadOnly)

        if not self.dataset:
            print("Erreur ouverture fichier")
        else:
            self.xsize = self.dataset.RasterXSize
            self.ysize = self.dataset.RasterYSize
            self.proj = self.dataset.GetProjection()
            self.gt = self.dataset.GetGeoTransform()

    def getArray(self, masked=False, lower_valid_range=None, upper_valid_range=None):
        """ Permet de récupérer un array Numpy contenant l'ensemble des valeurs de pixels de l'image.

            Les paramètres d'entrée offrent deux options:
                1) On récupère le array sans masquage
                2) On récupère le array en masquant les valeur hors intervalle valide

                Args:
                    masked (bool): Paramètre pour masquer le array ou non (par défaut, on ne masque pas)
                    lower_valid_range (int ou float): Valeur minimale acceptée comme une valeur valide à ne pas masquer
                    upper_valid_range (int ou float): Valeur maximale acceptée comme une valeur valide à ne pas masquer
                Returns:
                    array (Numpy.array ou Numpy.ma.masked_array): Array des valeurs de pixels de l'image.
        """
        # On masque seulement si on a les 3 paramètres optionnels à une valeur autre que celle par défaut
        if masked and lower_valid_range is not None and upper_valid_range is not None:
            band = self.dataset.GetRasterBand(1)
            in_array = band.ReadAsArray().astype(np.float32)

            # on remplace les valeurs à l'extérieur de l'intervalle de validité par -9999
            out_array = np.where(np.logical_or(in_array > upper_valid_range, in_array < lower_valid_range), -9999, in_array)
            noDataIndex = np.where(out_array < 0, 1, 0)
            array = ma.masked_array(out_array, noDataIndex)  # on masque le  array original

        else:
            band = self.dataset.GetRasterBand(1)
            array = band.ReadAsArray().astype(np.float32)

        band.FlushCache()
        band = None  # fermeture du fichier
        return array

    def save_band(self, array, filename):
        """ Permet de sauvegarder un array Numpy dans un fichier TIFF avec le même géoréférencement que l'objet de la
            classe Image (mêmes dimensions et même système de référence).
                Args:
                    array (Numpy.array): Array des valeurs de pixels de l'image.
                    filename (string): Fichier TIFF dans lequel on veut sauvegarder l'information.
        """
        driver = gdal.GetDriverByName('GTiff')
        driver.Register()
        outds = driver.Create(filename, xsize=self.xsize, ysize=self.ysize, bands=1, eType=gdal.GDT_Float32)
        outds.SetGeoTransform(self.gt)
        outds.SetProjection(self.proj)
        outband = outds.GetRasterBand(1)
        outband.WriteArray(array)
        outband.SetNoDataValue(np.nan)
        outband.FlushCache()
        outband = None  # fermeture des fichiers
        outds = None

    def getMetadata(self):
        """ Permet de récupérer les métadonnées de l'image.
                Returns:
                    meta (dict): Dictionnaire contenant les informations comprises dans les métadonnées de l'image.
        """
        inputFile = self.filename

        if "_reproject" in inputFile:
            inputFile = inputFile.replace("_reproject", "")  # les métadonnées sont intactes dans le fichier original,
                                                             # mais pas dans les fichiers reprojetés
        metadata_json = gdal.Info(inputFile, format='json')
        meta = metadata_json['metadata']['']

        return meta

    def reprojectMatch(self, referenceFile, reduce_zone=True):
        """ Permet de reprojeter, découper, aligner et rééchantillonner une image à partir d'une image de référence.
                Args:
                    referenceFile (string): Path du fichier de l'image de référence à utiliser.
                    reduce_zone (bool): Indicateur permettant de choisir si on souhaite réduire la zone d'étude
                                        sur laquelle les images sont "matchées". Ceci est utile pour éviter des
                                        problèmes avec des valeurs nulles sur les bords des images qui s'alignent
                                        sur le referenceFile. Par défaut, cette option est égale à True (donc, on
                                        effectue le rétrécissement de zone).
                Returns:
                    outputfile (string): Path du fichier de l'image reprojetée.
        """
        inputFile = self.filename  # path de l'image à reprojeter
        # input = gdal.Open(inputFile, gdalconst.GA_ReadOnly)

        # Ouvrir l'image de référence et obtenir sa projection ses paramètres de transformation affine
        reference = gdal.Open(referenceFile, gdalconst.GA_ReadOnly)
        referenceProj = reference.GetProjection()
        referenceTrans = reference.GetGeoTransform()
        bandreference = reference.GetRasterBand(1)

        # Transformer les paramètres de la transformation de tuples vers list afin de pouvoir les modifier
        # On additionne la resolution des pixels aux coordonnées du coin en haut a gauche du pixel en haut a gauche
        # afin d'avoir une zone de référence plus petite que la zone de l'input file
        referenceTrans = list(referenceTrans)
        referenceTrans[0] = referenceTrans[0] + referenceTrans[1]
        referenceTrans[3] = referenceTrans[3] + referenceTrans[5]
        referenceTrans = tuple(referenceTrans)

        if reduce_zone:
            x = reference.RasterXSize - 2
            y = reference.RasterYSize - 2
        else:
            x = reference.RasterXSize
            y = reference.RasterYSize

        # Créer le outputfile avec le format de l'image de référence
        outputfile = inputFile.replace(".tif", "_reproject.tif")
        driver = gdal.GetDriverByName('GTiff')
        output = driver.Create(outputfile, x, y, 1, bandreference.DataType)
        output.SetGeoTransform(referenceTrans)
        output.SetProjection(referenceProj)

        # Reprojeter l'image
        gdal.ReprojectImage(self.dataset, output, self.proj, referenceProj, gdalconst.GRA_Average)

        del output
        return outputfile

    """
    Fonction permettant de reprojeter l<image qui sera utilisee en reference. Dans ce cas precis, elle reprojete en UTM 
    zone 18
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

    """
    Fonction permettant de reprojeter l'image qui sera utilisée en référence dans reprojectMatch.
        outFilename: nom du fichier virtuel en sortie (sans l'extension)
        outCRS: système de référence dans lequel on souhaite projeter le jeu de données
        x_resolution: résolution en X souhaitée dans le résultat de la projection
        y_resolution: résolution en Y souhaitée dans le résultat de la projection
        resample_alg: algorithme de rééchantillonnage utilisé dans la reprojection (par défaut: plus proche voisin)
    """
    """
    def reproject(self, outFilename, outCRS, x_resolution, y_resolution, resample_alg="near"):
        outVrt = outFilename + '.vrt'
        warp = gdal.Warp(outVrt, self.filename, format="VRT", dstSRS=outCRS, xRes=x_resolution, yRes=y_resolution,
                         options=gdal.WarpOptions(resampleAlg=resample_alg))
        warp = None

        self.filename = outVrt  # la classe Image pointe maintenant sur ce fichier reprojeté
    """

    """
    *** Semble marcher mieux que les autres fonctions ***
    """
    def reproject(self, infile, outfile, outSRS, noDataVal, resolution, resample_alg):
        """ Permet de reprojeter une image d'un système de référence à un autre.
            Exemples d'arguments: reproject('data/infile.tif', 'data/outfile.tif', 'EPSG:32618', '-9999.0', '1000.0', 'average')
                Args:
                    infile (string): Path du fichier de l'image en entrée à reprojeter.
                    outfile (string): Path du fichier de l'image en sortie lorsqu'elle sera reprojetée
                    outSRS (string): Code EPSG du système de référence auquel on souhaite reprojeter (ex: EPSG:31618).
                    noDataVal (string): Valeur à fixer pour les valeurs nulles.
                    resolution (string): Résolution spatiale (taille de pixel) voulue.
                    resample_alg (string): Algorithme de rééchantillonnage à utiliser (ex: average, near).
        """
        if os.path.isfile(outfile):
            os.remove(outfile)  # si le fichier output est déjà existant, on le supprime

        string = 'gdalwarp ' + infile + ' ' + outfile + ' -t_srs ' + outSRS + ' -dstnodata ' \
                 + noDataVal + ' -tr ' + resolution + ' ' + resolution + ' -r ' + resample_alg
        os.system(string)

        self.filename = outfile  # remplacement du filename de l'image par le nouveau fichier reprojeté


def main():
    """ Tests de la classe et de ses méthodes.
    """
    b1 = r'data/CU_LC08.001_SRB1_doy2020229_aid0001_test2.tif'
    image1 = Image(b1)

    print(type(image1.dataset))

    #image1.reprojectUTM18()
    #image1.reproject(b1, 'outfile5_test.tif', 'EPSG:32618', '-9999.0', '30.0', 'average')


if __name__ == '__main__':
    main()
