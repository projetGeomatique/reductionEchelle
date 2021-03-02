from landsat import Landsat
from modis import Modis
from aster import Aster
from secteur import Secteur


def main():
    # secteur 1
    b1 = r'data/CU_LC08.001_SRB1_doy2020229_aid0001.tif'
    b2 = r'data/CU_LC08.001_SRB2_doy2020229_aid0001.tif'
    b3 = r'data/CU_LC08.001_SRB3_doy2020229_aid0001.tif'
    b4 = r'data/CU_LC08.001_SRB4_doy2020229_aid0001.tif'
    b5 = r'data/CU_LC08.001_SRB5_doy2020229_aid0001.tif'
    b6 = r'data/CU_LC08.001_SRB6_doy2020229_aid0001.tif'
    b7 = r'data/CU_LC08.001_SRB7_doy2020229_aid0001.tif'
    qa = r'data/CU_LC08.001_PIXELQA_doy2020229_aid0001.tif'
    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa)

    lst = r'data/MOD11A1.006_Clear_day_cov_doy2020229_aid0001.tif'
    qa = r'data/MOD11A1.006_QC_Day_doy2020229_aid0001.tif'
    modis = Modis(lst, qa)

    mnt = r'data\ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif'
    qa = r'data\ASTGTM_NUMNC.003_ASTER_GDEM_NUM_doy2000061_aid0001.tif'
    aster = Aster(mnt, qa)

    s1 = Secteur(modis, landsat, aster)
    s1.prepareData()

    df1 = s1.getDf()

    print(df1)
    
    # secteur 2
    b1 = r'data/secteur2/CU_LC08.001_SRB1_doy2020229_aid0001.tif'
    b2 = r'data/secteur2/CU_LC08.001_SRB2_doy2020229_aid0001.tif'
    b3 = r'data/secteur2/CU_LC08.001_SRB3_doy2020229_aid0001.tif'
    b4 = r'data/secteur2/CU_LC08.001_SRB4_doy2020229_aid0001.tif'
    b5 = r'data/secteur2/CU_LC08.001_SRB5_doy2020229_aid0001.tif'
    b6 = r'data/secteur2/CU_LC08.001_SRB6_doy2020229_aid0001.tif'
    b7 = r'data/secteur2/CU_LC08.001_SRB7_doy2020229_aid0001.tif'
    qa = r'data/secteur2/CU_LC08.001_PIXELQA_doy2020229_aid0001.tif'
    landsat = Landsat(b1, b2, b3, b4, b5, b6, b7, qa)


    lst = r'data/secteur2/MOD11A1.006_LST_Day_1km_doy2020229_aid0001.tif'
    qa = r'data/secteur2/MOD11A1.006_QC_Day_doy2020229_aid0001.tif'
    modis = Modis(lst, qa)

    mnt = r'data\secteur2\ASTGTM_NC.003_ASTER_GDEM_DEM_doy2000061_aid0001.tif'
    qa = r'data\secteur2\ASTGTM_NUMNC.003_ASTER_GDEM_NUM_doy2000061_aid0001.tif'
    aster = Aster(mnt, qa)


    s2 = Secteur(modis, landsat, aster)
    s2.prepareData()

    df2 = s2.getDf()

    print(df2)

    df = df1.append([df2], ignore_index=True)

    print(df)

if __name__ == '__main__':
    main()