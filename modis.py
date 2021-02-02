from image import Image


class Modis(Image):
    def __init__(self, lst, qa):
        super().__init__(lst)
        self.lst = Image.getArray(lst)
        self.qa = Image.getArray(qa)

def main():
    lst = 'MOD11A1.006_LST_Day_1km_doy2020217_aid0001.tif'
    qa = 'MOD11A1.006_LST_Day_1km_doy2020217_aid0001.tif'
    modis = Modis(lst, qa)

    print(modis.lst)




if __name__ == '__main__':
    main()

