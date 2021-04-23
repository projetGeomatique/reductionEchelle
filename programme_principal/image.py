import gdal
import gdalconst
import numpy as np
import numpy.ma as ma
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

    def setNewFile(self, filename):
        """ Permet d'assigner un nouveau fichier pour l'objet d'image déjà instancié (on réassigne des nouvelles
            valeurs pour les attributs de la classe).
                Args:
                    filename (str): Path vers le novueau fichier .tiff de l'image
        """
        self.filename = filename
        self.dataset = gdal.Open(self.filename, gdal.GA_ReadOnly)

        if not self.dataset:
            print("Erreur ouverture fichier")
        else:
            self.xsize = self.dataset.RasterXSize
            self.ysize = self.dataset.RasterYSize
            self.proj = self.dataset.GetProjection()
            self.gt = self.dataset.GetGeoTransform()

    def getArray(self, masked=False, lower_valid_range=None, upper_valid_range=None, qa_filename=None,
                 cloud_overlay_filename=None, data_source=None):
        """ Permet de récupérer un array Numpy contenant l'ensemble des valeurs de pixels de l'image.

            Les paramètres d'entrée offrent quatres options:
                1) On récupère le array en masquant les valeurs hors intervalle valide
                2) On récupère le array en masquant les valeurs hors intervalle valide et on masque les nuages selon
                   les codes de la bande de qualité (qa)
                3) On récupère le array en masquant les valeurs hors intervalle valide et on masque les nuages selon
                   le overlay des pixels à 1000m à masquer (voir la méthode cloudOverlay)
                4) On récupère le array sans masquage

                Args:
                    masked (bool): Paramètre pour masquer le array ou non (par défaut, on ne masque pas)
                    lower_valid_range (int ou float): Valeur minimale acceptée comme une valeur valide à ne pas masquer
                    upper_valid_range (int ou float): Valeur maximale acceptée comme une valeur valide à ne pas masquer
                    qa_filename (str): Path vers le fichier de la bande de qualité à utiliser pour le masquage des nuages
                    cloud_overlay_filename (str): Path vers le fichier pour les pixels à masquer selon le pourcentage
                                                  de nuages obtenu du overlay
                    data_source (str): Source des données (appeears ou earthdata). Les codes des nuages à masquer ne sont
                                       pas les mêmes d'une source de données à l'autre.
                Returns:
                    array (Numpy.array ou Numpy.ma.masked_array): Array des valeurs de pixels de l'image.
        """

        # on ouvre le fichier de l'image en array Numpy
        band = self.dataset.GetRasterBand(1)
        in_array = band.ReadAsArray().astype(np.float32)

        # Option 1: on masque les valeurs à l'extérieur de l'intervalle de validité, mais on ne masque pas les nuages
        if masked and lower_valid_range is not None and upper_valid_range is not None and qa_filename is None and cloud_overlay_filename is None:

            # on remplace les valeurs à l'extérieur de l'intervalle de validité par -9999
            out_array = np.where(np.logical_or(in_array > upper_valid_range, in_array < lower_valid_range), -9999,
                                 in_array)
            noDataIndex = np.where(out_array < 0, 1, 0)
            array = ma.masked_array(out_array, noDataIndex)  # on masque le  array original

        # Option 2: Masquage appliqué sur les images Landsat (à 30m) pour les nuages selon la bande de qualité
        elif masked and qa_filename is not None:

            # masquage standard de l'option 1
            out_array = np.where(np.logical_or(in_array > upper_valid_range, in_array < lower_valid_range), -9999,
                                 in_array)

            # codes pour les nuages selon les bandes de qualité de appeears ou earthdata
            if data_source == "appeears":
                clouds = [352, 368, 416, 432, 480, 864, 880, 928, 944, 992, 328, 392, 840, 904, 1350, 834, 836, 840, 848,
                          864, 880, 898, 900, 904, 912]
            elif data_source == "earthdata":
                clouds = [2800, 2804, 2808, 2812, 2752, 2756, 2760, 2764, 3008, 3012, 3016, 3020, 3776, 3784, 3788,
                          3780, 6848, 6852, 6856, 6860, 6896, 6900, 6904, 6908, 7104, 7108, 7112, 7116, 7872, 7876, 7880,
                          7884]

            # ouvrir la bande de qualité
            qa = gdal.Open(qa_filename)
            band_qa = qa.GetRasterBand(1)
            qa_array = band_qa.ReadAsArray().astype(np.float32)

            # masque tous les pixels qui ont un des codes précédents dans la bande de qualité (nuages)
            masked_array = np.isin(qa_array, clouds)
            final_array = np.where(masked_array == True, -9999, out_array)

            noDataIndex = np.where(final_array < 0, 1, 0)
            array = ma.masked_array(final_array, noDataIndex)  # on masque le  array original

        # Option 3: Masquage appliqué sur les images Landsat pour les nuages, mais avec l'overlay à 1000m
        elif masked and cloud_overlay_filename is not None:

            # on ouvre l'image de overlay
            cloud_overlay = gdal.Open(cloud_overlay_filename)
            cloud_overlay_band1 = cloud_overlay.GetRasterBand(1)
            cloud_overlay_band1_array = cloud_overlay_band1.ReadAsArray().astype(np.float32)

            out_array = np.where(
                np.logical_or(cloud_overlay_band1_array > upper_valid_range, cloud_overlay_band1_array == -9999), -9999,
                in_array)
            out_array = np.where(in_array == 0, -9999, out_array)

            noDataIndex = np.where(out_array < 0, 1, 0)
            array = ma.masked_array(out_array, noDataIndex)

        # Option 4: on ne masque pas et on obtient le array numpy de l'image sans masque
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
                    filename (str): Fichier TIFF dans lequel on veut sauvegarder l'information.
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
                    referenceFile (str): Path du fichier de l'image de référence à utiliser.
                    reduce_zone (bool): Indicateur permettant de choisir si on souhaite réduire la zone d'étude
                                        sur laquelle les images sont "matchées". Ceci est utile pour éviter des
                                        problèmes avec des valeurs nulles sur les bords des images qui s'alignent
                                        sur le referenceFile. Par défaut, cette option est égale à True (donc, on
                                        effectue le rétrécissement de zone).
                Returns:
                    outputfile (str): Path du fichier de l'image reprojetée.
        """
        inputFile = self.filename  # path de l'image à reprojeter

        # Ouvrir l'image de référence et obtenir sa projection ses paramètres de transformation affine
        reference = gdal.Open(referenceFile, gdalconst.GA_ReadOnly)
        referenceProj = reference.GetProjection()
        referenceTrans = reference.GetGeoTransform()
        bandreference = reference.GetRasterBand(1)

        # Transformer les paramètres de la transformation de tuples vers list afin de pouvoir les modifier.
        # On additionne la résolution des pixels aux coordonnées du coin en haut à gauche du pixel en haut à gauche
        # afin d'avoir une zone de référence plus petite que la zone de l'input file
        referenceTrans = list(referenceTrans)
        referenceTrans[0] = referenceTrans[0] + referenceTrans[1]
        referenceTrans[3] = referenceTrans[3] + referenceTrans[5]
        referenceTrans = tuple(referenceTrans)

        # on réduit la zone de 2 lignes et 2 colonnes (au bords de l'image) si reduce_zone = True
        if reduce_zone:
            x = reference.RasterXSize - 2
            y = reference.RasterYSize - 2
        else:
            x = reference.RasterXSize
            y = reference.RasterYSize

        # Créer le outputfile avec le format de l'image de référence
        if inputFile.endswith(".TIF"):  # .TIF si l'image provient de earthdata
            outputfile = inputFile.replace(".TIF", "_reproject.TIF")
        else:
            outputfile = inputFile.replace(".tif", "_reproject.tif")

        driver = gdal.GetDriverByName('GTiff')
        output = driver.Create(outputfile, x, y, 1, gdal.GDT_Float32)  # originalement bandreference.DataType au lieu de GDT_Float32
        output.SetGeoTransform(referenceTrans)
        output.SetProjection(referenceProj)

        # Reprojeter l'image
        gdal.ReprojectImage(self.dataset, output, self.proj, referenceProj, gdalconst.GRA_Average)

        del output
        return outputfile

    def reproject(self, infile, outfile, outSRS, noDataVal, resolution, resample_alg):
        """ Permet de reprojeter une image d'un système de référence à un autre.
            Exemples d'arguments: reproject('data/infile.tif', 'data/outfile.tif', 'EPSG:32618', '-9999.0', '1000.0', 'average')
                Args:
                    infile (str): Path du fichier de l'image en entrée à reprojeter.
                    outfile (str): Path du fichier de l'image en sortie lorsqu'elle sera reprojetée
                    outSRS (str): Code EPSG du système de référence auquel on souhaite reprojeter (ex: EPSG:32618).
                    noDataVal (str): Valeur à fixer pour les valeurs nulles.
                    resolution (str): Résolution spatiale (taille de pixel) voulue.
                    resample_alg (str): Algorithme de rééchantillonnage à utiliser (ex: average, near).
        """
        if os.path.isfile(outfile):
            os.remove(outfile)  # si le fichier output est déjà existant, on le supprime

        # construction de la chaîne de caractères qu'on passe à os.system() pour exécuter les fonctions gdal
        string = 'gdalwarp ' + infile + ' ' + outfile + ' -t_srs ' + outSRS + ' -dstnodata ' \
                 + noDataVal + ' -tr ' + resolution + ' ' + resolution + ' -r ' + resample_alg
        os.system(string)  # command line utility de GDAL

        self.filename = outfile  # remplacement du filename de l'image par le nouveau fichier reprojeté

    def cloudOverlay(self, fileLowRes, reduce_zone=True, data_source=None):
        """ Permet d'effectuer un overlay entre la bande de qualité (qa) à 30m et une image à basse résolution (1km).
                Args:
                    fileLowRes (str): Path du fichier de basse résolution spatiale dans lequel on veut savoir chaque
                                      pixel contient quel pourcentage de nuages à haute résolution.
                    reduce_zone (bool): Indicateur permettant de choisir si on souhaite réduire la zone d'étude
                                        sur laquelle les images sont "matchées". Ceci est utile pour éviter des
                                        problèmes avec des valeurs nulles sur les bords des images qui s'alignent
                                        sur le referenceFile. Par défaut, cette option est égale à True (donc, on
                                        effectue le rétrécissement de zone).
                    data_source (str): Source des données (appeears ou earthdata). Les codes des nuages à masquer ne sont
                                       pas les mêmes d'une source de données à l'autre.
                Returns:
                    dst_ds_name (str): Path du fichier de overlay avec le pourcentage de nuages inclus dans chaque pixel
                                       de basse résolution.
        """
        highRes = gdal.Open(self.filename)  # on ouvre l'image Landsat QA (30m)
        band = highRes.GetRasterBand(1)
        pj = self.proj
        gt = self.gt

        highResArray = highRes.ReadAsArray()

        # codes pour les nuages et l'eau dépendamment de la source des images (appeears ou earthdata)
        if data_source == "appeears":
            clouds = [352, 368, 416, 432, 480, 864, 880, 928, 944, 992, 328, 392, 840, 904, 1350, 834, 836, 840, 848, 864,
                      880, 898, 900, 904, 912, 928, 944, 992]

            water = [324, 388, 836, 900, 1348]  # codes pour la présence d'eau
        elif data_source == "earthdata":
            clouds = [2800, 2804, 2808, 2812, 2752, 2756, 2760, 2764, 3008, 3012, 3016, 3020, 3776, 3784, 3788, 3780,
                      6848, 6852, 6856, 6860, 6896, 6900, 6904, 6908, 7104, 7108, 7112, 7116, 7872, 7876, 7880, 7884]
            water = []

        # on remplace les valeurs du highResArray à 0 quand elles sont égales à 1 (dans la bande QA)
        highResArray = np.where(highResArray == 1, 0, highResArray)

        # on remplace les valeurs du highResArray à 1 quand elles sont égales aux codes des nuages (dans la bande QA)
        for i in clouds:
            highResArray = np.where(highResArray == i, 1, highResArray)

        # on remplace les valeurs du highResArray à 2 quand elles sont égales aux codes de l'eau (dans la bande QA)
        for i in water:
            highResArray = np.where(highResArray == i, 2, highResArray)

        # on remplace les valeurs du highResArray à 3 quand elles ne sont pas égales à 1 ni à 2
        highResArray = np.where(np.logical_and(highResArray != 1, highResArray != 2), 3, highResArray)

        highRes = band = None  # fermeture du fichier

        classIds = (np.arange(3) + 1).tolist()  # classIds: [1, 2, 3]

        # Construction d'un bit raster (3 bandes, binaire 0 ou 1)
        # Bande 1 = 0 -> pas de nuage, Bande 1 = 1 -> nuage
        # Bande 2 = 0 -> pas d'eau, Bande 2 = 1 -> eau
        # Bande 3 = 0 -> objet à masquer, Bande 3 = 1 -> rien à masquer
        drv = gdal.GetDriverByName('GTiff')
        ds = drv.Create('data/bit_raster.tif', highResArray.shape[1], highResArray.shape[0],
                        len(classIds), gdal.GDT_Byte,
                        ['NBITS=1', 'COMPRESS=LZW', 'INTERLEAVE=BAND'])
        ds.SetGeoTransform(gt)
        ds.SetProjection(pj)
        for bidx in range(ds.RasterCount):
            band = ds.GetRasterBand(bidx + 1)
            selection = (highResArray == classIds[bidx])
            band.WriteArray(selection.astype('B'))
        ds = band = None  # fermeture des fichiers

        # On ouvre le bit raster qu'on vient de créer
        src_ds = gdal.Open('data/bit_raster.tif')

        # On ouvre une copie, pour les dimensions et le masque de NoData
        cpy_ds = gdal.Open(fileLowRes)
        band = cpy_ds.GetRasterBand(1)
        cpy_mask = (band.ReadAsArray() == band.GetNoDataValue())

        # Image résultante de même résolution et position que la copie
        referenceTrans = cpy_ds.GetGeoTransform()
        referenceTrans = list(referenceTrans)
        referenceTrans[0] = referenceTrans[0] + referenceTrans[1]
        referenceTrans[3] = referenceTrans[3] + referenceTrans[5]
        referenceTrans = tuple(referenceTrans)

        # gestion des images provenant de earthdata (.TIF) ou appeears (.tif)
        if self.filename.endswith(".TIF"):
            dst_ds_name = self.filename.replace('.TIF', 'pourcentagenuage_1km.TIF')
        else:
            dst_ds_name = self.filename.replace('.tif', 'pourcentagenuage_1km.tif')

        # réduire la zone de 2 lignes et 2 colonnes si on a choisi cette option en input
        if reduce_zone:
            dst_ds = drv.Create(dst_ds_name, cpy_ds.RasterXSize - 2, cpy_ds.RasterYSize - 2,
                                len(classIds), gdal.GDT_Float32, ['INTERLEAVE=BAND'])
        else:
            dst_ds = drv.Create(dst_ds_name, cpy_ds.RasterXSize, cpy_ds.RasterYSize,
                                len(classIds), gdal.GDT_Float32, ['INTERLEAVE=BAND'])

        dst_ds.SetGeoTransform(referenceTrans)
        dst_ds.SetProjection(cpy_ds.GetProjection())

        # Rééchantillonnage par moyenne
        gdal.ReprojectImage(src_ds, dst_ds, None, None, gdal.GRA_Average)

        # Convertir les fractions en pourcentages et appliquer le masque de NoData à la copie
        btest = dst_ds.GetRasterBand(bidx + 1)
        artest = btest.ReadAsArray() * 100.0
        if cpy_mask.size != artest.size:
            cpy_mask = cpy_mask[1:-1, 1:-1]
        cpyshape = cpy_mask.size
        arshape = artest.size
        NODATA = -9999
        for bidx in range(dst_ds.RasterCount):
            band = dst_ds.GetRasterBand(bidx + 1)
            ar = band.ReadAsArray() * 100.0
            ar[cpy_mask] = NODATA
            band.WriteArray(ar)
            band.SetNoDataValue(NODATA)

        src_ds = cpy_ds = dst_ds = band = None  # sauvegarder et fermer les fichiers

        return dst_ds_name


def main():
    """ Tests de la classe et de ses méthodes.
    """
    b1 = r'data/CU_LC08.001_SRB1_doy2020229_aid0001_test2.tif'
    image1 = Image(b1)

    print(type(image1.dataset))

    # image1.reprojectUTM18()
    # image1.reproject(b1, 'outfile5_test.tif', 'EPSG:32618', '-9999.0', '30.0', 'average')


if __name__ == '__main__':
    main()
