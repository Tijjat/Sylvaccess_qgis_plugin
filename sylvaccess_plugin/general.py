



import datetime,gc
import numpy as np
from osgeo import gdal, osr, ogr

#########################################################################
#  _______  _______ .__   __.  _______ .______        ___       __      #
# /  _____||   ____||  \ |  | |   ____||   _  \      /   \     |  |     #
#|  |  __  |  |__   |   \|  | |  |__   |  |_)  |    /  ^  \    |  |     #
#|  | |_ | |   __|  |  . `  | |   __|  |      /    /  /_\  \   |  |     #
#|  |__| | |  |____ |  |\   | |  |____ |  |\  \--./  _____  \  |  `----.#
# \______| |_______||__| \__| |_______|| _| `.___/__/     \__\ |_______|#
#########################################################################
    

def heures(Hdebut):
    Hfin = datetime.datetime.now()
    duree = Hfin - Hdebut
    str_duree = str(duree).split('.')[0]
    str_duree = str_duree.split(':')[0] + 'h ' + str_duree.split(':')[1] + 'm ' + str_duree.split(':')[2] + 's'
    str_debut = Hdebut.strftime('%d/%m/%Y %H:%M:%S')
    str_fin = Hfin.strftime('%d/%m/%Y %H:%M:%S')

    return str_duree, str_fin, str_debut


def save_integer_ascii(file_name,head_text,data):
    np.savetxt(file_name, data, fmt='%i', delimiter=' ')
    with open(file_name, "r+") as f:
        old = f.read()
        f.seek(0)
        f.write(head_text + old)
        f.close()


def clear_big_nparray():    
    """clear all globals over 100 Mo size and their associated memory space"""
    for uniquevar in [var for var in dir() if isinstance(globals()[var],np.ndarray)]:
        if globals()[uniquevar].nbytes/1000000>50:
            del globals()[uniquevar]
    gc.collect()


def read_info(info_file):
    names = np.genfromtxt(info_file, dtype=None,usecols=(0),encoding ='latin1')
    values = np.genfromtxt(info_file, dtype=None,usecols=(1),encoding ='latin1')  
    return list(names),list(values)


def raster_get_info(in_file_name):
    source_ds = gdal.Open(in_file_name)    
    src_proj = osr.SpatialReference(wkt=source_ds.GetProjection())
    src_ncols = source_ds.RasterXSize
    src_nrows = source_ds.RasterYSize
    xmin,Csize_x,_,ymax,_,Csize_y = source_ds.GetGeoTransform()
    ymin = ymax+src_nrows*Csize_y
    nodata = source_ds.GetRasterBand(1).GetNoDataValue()
    names = ['ncols', 'nrows', 'xllcorner', 'yllcorner', 'cellsize','NODATA_value']
    values = [src_ncols,src_nrows,xmin,ymin,Csize_x,nodata]
    Extent = [xmin,xmin+src_ncols*Csize_x,ymin,ymax]
    return names,values,src_proj,Extent


def read_raster(file_name):
    source_ds = gdal.Open(file_name)
    source_ds.FlushCache() # Flush 
    Array = source_ds.GetRasterBand(1).ReadAsArray()
    Array[Array==0]=-9999
    return Array


def from_az_to_arr(xmin,xmax,ymin,ymax,Csize,Lmax,az):    
    X1 = np.sin(np.radians(az))*Lmax
    Y1 = np.cos(np.radians(az))*Lmax
    #Initialize raster info
    nrows,ncols = int((ymax-ymin)/float(Csize)+0.5),int((xmax-xmin)/float(Csize)+0.5)    
    #create polygon object:
    driver = ogr.GetDriverByName('Memory')
    datasource = driver.CreateDataSource('')
    source_srs=osr.SpatialReference()
    source_srs.ImportFromEPSG(2154)
    layer = datasource.CreateLayer('layerName',source_srs,geom_type=ogr.wkbLineString)
    layerDefinition = layer.GetLayerDefn()
    new_field = ogr.FieldDefn('ID', ogr.OFTInteger)
    layer.CreateField(new_field)
    line = ogr.Geometry(ogr.wkbLineString)
    line.AddPoint(0,0)
    line.AddPoint(X1,Y1)
    feature = ogr.Feature(layerDefinition)
    feature.SetGeometry(line)
    feature.SetFID(az)
    feature.SetField('ID',1)
    layer.CreateFeature(feature)    
    feature.Destroy()     
    # Initialize the new memory raster      
    maskvalue = 1    
    xres=float(Csize)
    yres=float(Csize)
    geotransform=(xmin,xres,0,ymax,0, -yres)    
    target_ds = gdal.GetDriverByName('MEM').Create('', int(ncols), int(nrows), 1, gdal.GDT_Int16)
    target_ds.SetGeoTransform(geotransform)
    if source_srs:
        # Make the target raster have the same projection as the source
        target_ds.SetProjection(source_srs.ExportToWkt())
    else:
        # Source has no projection (needs GDAL >= 1.7.0 to work)
        target_ds.SetProjection('LOCAL_CS["arbitrary"]')
    # Rasterize
    err = gdal.RasterizeLayer(target_ds, [maskvalue], layer,options=["ATTRIBUTE=ID","ALL_TOUCHED=TRUE"])
    if err != 0:
        raise Exception("error rasterizing layer: %s" % err)
    else:
        target_ds.FlushCache()
        mask_arr = target_ds.GetRasterBand(1).ReadAsArray()
    datasource.Destroy()
    return X1,Y1,mask_arr
