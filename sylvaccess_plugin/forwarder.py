"""
/***************************************************************************
 forwarder
                                 A QGIS plugin
 This plugin is the Sylvaccess app made in qgis
 Generated by Plugin Builder: http://g-sherman.github.io/Qgis-Plugin-Builder/
                             -------------------
        begin                : 2024-01-19
        git sha              : $Format:%H$
        copyright            : (C) 2024 by Cosylval
        email                : yoann.zenner@viacesi.fr
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

import os,gc,datetime
import numpy as np
from PyQt5.QtCore import QCoreApplication
from .console import console_info
from .gis import shapefile_to_np_array,load_float_raster,load_float_raster_simple,ArrayToGtiff,get_proj_from_road_network, get_source_src
from .function_np import pente, exposition, focal_stat_max, focal_stat_nb, calcul_distance_de_cout, Link_RF_res_pub, Link_tracks_res_pub, Dfwd_flat_forest_tracks, Dfwd_flat_forest_road, fwd_azimuts_contour, Fwd_add_contour, fill_Link
from .general import clear_big_nparray, raster_get_info, read_info, heures
from .skidder import prepa_obstacle_skidder, create_buffer_skidder, create_arrays_from_roads, make_summary_surface_vol, create_access_shapefile


###########################################################################
#.______     ______   .______  .___________. _______  __    __  .______   #
#|   _  \   /  __  \  |   _  \ |           ||   ____||  |  |  | |   _  \  #
#|  |_)  | |  |  |  | |  |_)  |`---|  |----`|  |__   |  |  |  | |  |_)  | #
#|   ___/  |  |  |  | |      /     |  |     |   __|  |  |  |  | |      /  #
#|  |      |  `--'  | |  |\  \-.   |  |     |  |____ |  `--'  | |  |\  \-.#
#| _|       \______/  | _| `.__|   |__|     |_______| \______/  | _| `.__|#
# #########################################################################                                                                               

  
def prepa_data_fwd(Wspace,Rspace,file_MNT,file_shp_Foret,file_shp_Desserte,Dir_Obs_forwarder):
    console_info(QCoreApplication.translate("MainWindow","Pre-processing of the inputs for forwarder model"))
    ### Make directory for temporary files
    Dir_temp = Wspace+"Temp/"
    try:os.mkdir(Dir_temp)
    except:pass 
    Rspace_f = Rspace+"Forwarder/"
    try:os.mkdir(Rspace_f)
    except:pass
    ##############################################################################################################################################
    ### Initialization
    ##############################################################################################################################################
    _,values,_,Extent = raster_get_info(file_MNT)
    Csize,ncols,nrows = values[4],int(values[0]),int(values[1])  
    road_network_proj=get_proj_from_road_network(file_shp_Desserte)
    ##############################################################################################################################################
    ### Forest : shapefile to raster
    ##############################################################################################################################################
    Foret = shapefile_to_np_array(file_shp_Foret,Extent,Csize,"FORET")
    np.save(Dir_temp+"Foret",np.int8(Foret))    
    del Foret
    console_info(QCoreApplication.translate("MainWindow","    - Forest raster processe"))
    ##############################################################################################################################################
    ### Calculation of a slope raster and a cost raster of slope
    ##############################################################################################################################################
    # Slope raster
    MNT,Extent,Csize,_ = load_float_raster(file_MNT,Dir_temp)
    np.save(Dir_temp+"MNT",np.float32(MNT))      
    Pente = pente(MNT,Csize,-9999)
    np.save(Dir_temp+"Pente",np.float32(Pente))    
    Exposition = np.int16(exposition(MNT,Csize,-9999)+0.5)
    Exposition[Pente==-9999] = -9999
    np.save(Dir_temp+"Aspect",Exposition) 
    # Cost raster of slope
    Pond_pente = np.sqrt((Pente*0.01*Csize)**2+Csize**2)/float(Csize)
    Pond_pente[Pente==-9999] = 10000
    np.save(Dir_temp+"Pond_pente",np.float32(Pond_pente))
    # Report a success message   
    del Pente,MNT,Exposition
    console_info(QCoreApplication.translate("MainWindow","    - Slope and aspects rasters processed"))  
    ##############################################################################################################################################
    ### Road network processing
    ##############################################################################################################################################
    Res_pub,Route_for,Piste= create_arrays_from_roads(file_shp_Desserte,Extent,Csize)
    np.save(Dir_temp+"Res_pub",Res_pub)  
    ##############################################################################################################################################
    ### Forest road network processing
    ##############################################################################################################################################
    pixels = np.argwhere(Res_pub==1) 
    # Give an identifiant to each public network pixel    
    ID = 1    
    Tab_res_pub = np.zeros((pixels.shape[0]+1,2),dtype=np.int32) 
    for pixel in pixels:
        Tab_res_pub[ID,0],Tab_res_pub[ID,1]=pixel[0],pixel[1]
        ID +=1         
    np.save(Dir_temp+"Tab_res_pub",Tab_res_pub)
    pixels = np.argwhere(Route_for==1)
    #num_ligne = id_RF, Y, X, Dtransp,Lien_Respub
    Lien_RF = np.zeros((pixels.shape[0]+1,5),dtype=np.float32)     
    ID = 1
    for pixel in pixels:
        Lien_RF[ID,0],Lien_RF[ID,1]=pixel[0],pixel[1]
        Lien_RF[ID,3]=-9999
        if Pond_pente[pixel[0],pixel[1]]==10000:
            Lien_RF[ID,2]=-9999
        else:
            Lien_RF[ID,2]=100001
        ID +=1 
    # Link RF with res_pub and calculate transportation distance
    Lien_RF=Link_RF_res_pub(Tab_res_pub,Pond_pente,Route_for,Res_pub, Lien_RF,Csize) 
    Lien_RF[:,2]=np.int_(Lien_RF[:,2]+0.5)
    Lien_RF=Lien_RF.astype('int')
    Temp = (Lien_RF[:,3]>0)*(Lien_RF[:,2]==0)
    Lien_RF=Lien_RF[Temp==0]    
    np.save(Dir_temp+"Lien_RF",Lien_RF)
    # Check if all Forest road are linked to public network    
    if np.max(Lien_RF[:,2])==100001:
        RF_bad = np.zeros((nrows,ncols),dtype=np.int8)
        pixels = np.argwhere(Lien_RF[:,2]==100001)
        for pixel in pixels:
            ind = pixel[0]            
            RF_bad[Lien_RF[ind,0],Lien_RF[ind,1]]=1        
        ArrayToGtiff(RF_bad,Rspace_f+'Forest_road_not_connected',Extent,nrows,ncols,road_network_proj,0,'UINT8')
        console_info(QCoreApplication.translate("MainWindow","    - Some forest road are not connected to public network. To see where, check raster ")+Rspace_f+"Forest_road_not_connected.tif")
    else:
        console_info(QCoreApplication.translate("MainWindow","    - Forest road processed")) 
             
    ##############################################################################################################################################
    ### Forest tracks network processing
    ##############################################################################################################################################
    pixels = np.argwhere(Piste==1)
    #num_ligne = id_piste, Y, X, Dpiste,Dtransp,Lien_RF, Lien_Respub,
    Lien_piste = np.zeros((pixels.shape[0]+1,7),dtype=np.float32)    
    ID = 1
    for pixel in pixels:
        Lien_piste[ID,0],Lien_piste[ID,1]=pixel[0],pixel[1]
        Lien_piste[ID,4]=-9999
        if Pond_pente[pixel[0],pixel[1]]==10000:
            Lien_piste[ID,2]=-9999
        else:
            Lien_piste[ID,2]=100001
        ID +=1
    Lien_piste=Link_tracks_res_pub(Tab_res_pub,Lien_RF,Pond_pente,Piste,Route_for,Res_pub,Lien_piste,Csize)
    Lien_piste[:,2]=np.int_(Lien_piste[:,2]+0.5)
    Lien_piste=Lien_piste.astype('int')
    Temp = (Lien_piste[:,5]>0)*(Lien_piste[:,2]==0)
    Lien_piste=Lien_piste[Temp==0]    
    ind = np.lexsort((Lien_piste[:,1],Lien_piste[:,2]))
    Lien_piste=Lien_piste[ind]
    np.save(Dir_temp+"Lien_piste",Lien_piste) 
    if np.max(Lien_piste[:,2])==100001:
        RF_bad = np.zeros((nrows,ncols),dtype=np.int8)
        pixels = np.argwhere(Lien_piste[:,2]==100001)
        for pixel in pixels:
            ind = pixel[0]            
            RF_bad[Lien_piste[ind,0],Lien_piste[ind,1]]=1   
            Piste[Lien_piste[ind,0],Lien_piste[ind,1]]=0
            ArrayToGtiff(RF_bad,Rspace_f+'Forest_tracks_not_connected',Extent,nrows,ncols,Csize,road_network_proj,0,'UINT8')
            console_info(QCoreApplication.translate("MainWindow","    - Some forest tracks are not connected to public network or forest road."))
            console_info(QCoreApplication.translate("MainWindow","      To see where, check raster "+Rspace_f+"Forest_tracks_not_connected.tif"))
    else:
        console_info(QCoreApplication.translate("MainWindow","    - Forest road processed"))  
    Route_for[Res_pub==1]=0
    Piste[Res_pub==1]=0
    np.save(Dir_temp+"Route_for",Route_for) 
    np.save(Dir_temp+"Piste",np.int8(Piste))
    del Tab_res_pub,Lien_RF,Lien_piste,Res_pub  
    gc.collect() 
    ##############################################################################################################################################
    ### Create a raster of total obstacle for forwarder
    ##############################################################################################################################################
    if Dir_Obs_forwarder!="":
        Obstacles_forwarder = prepa_obstacle_skidder(Dir_Obs_forwarder,Extent,Csize,ncols,nrows,((Route_for>0)*1+(Piste>0)*1))
    else:
        Obstacles_forwarder = np.zeros((nrows,ncols),dtype=np.int8)
    np.save(Dir_temp+"Obstacles_forwarder",np.int8(Obstacles_forwarder))    
    console_info(QCoreApplication.translate("MainWindow","    - Forwarder obstacles raster processed")) 
    console_info(QCoreApplication.translate("MainWindow","Input data processing achieved"))
    ##############################################################################################################################################
    ### Close the script
    ##############################################################################################################################################
    clear_big_nparray()


def process_forwarder(Wspace,Rspace,file_MNT,file_shp_Foret,file_shp_Desserte,Dir_Obs_forwarder,file_Vol_ha,Pente_max_bucheron,
                        Forw_angle_incl,Forw_angle_up,Forw_angle_down,Forw_Lmax,Forw_Dmax_out_for,Forw_portee,Forw_Debclass):


    console_info(QCoreApplication.translate("MainWindow","Sylvaccess - Forwarder starts"))

    ###############################################################################################################################################
    ### Initialisation
    ###############################################################################################################################################
    Hdebut = datetime.datetime.now()
    
    # Create a folder for process result
    Rspace_s = Rspace+"Forwarder/"
    try:os.mkdir(Rspace_s)
    except:pass
    Dir_temp = Wspace+"Temp/"
    
    # Check if temporary files have been generated and have the same extent
    try:
        _,values,_,Extent = raster_get_info(file_MNT)
    except:
        console_info(QCoreApplication.translate("MainWindow","Error: please define a projection for the DTM raster"))
        return ""    
    try: 
        _,v1=read_info(Dir_temp+'info_extent.txt')
        for i,item in enumerate(values):
            if v1[i]!=round(item,2):
                prepa_data_fwd(Wspace,Rspace,file_MNT,file_shp_Foret,file_shp_Desserte,Dir_Obs_forwarder)
            if i+1>4:break
    except:
        prepa_data_fwd(Wspace,Rspace,file_MNT,file_shp_Foret,file_shp_Desserte,Dir_Obs_forwarder)
    
    Csize = values[4]
    # Inputs
    try:
        Foret = np.int8(np.load(Dir_temp+"Foret.npy"))
        Piste = np.int8(np.load(Dir_temp+"Piste.npy"))
        Route_for = np.int8(np.load(Dir_temp+"Route_for.npy"))        
        Lien_piste = np.load(Dir_temp+"Lien_piste.npy")
        Res_pub = np.int8(np.load(Dir_temp+"Res_pub.npy"))
        Lien_RF = np.load(Dir_temp+"Lien_RF.npy")
        Pente = np.load(Dir_temp+"Pente.npy")
        Pond_pente = np.load(Dir_temp+"Pond_pente.npy")
        MNT = np.load(Dir_temp+"MNT.npy")
        try:
            Aspect = np.load(Dir_temp+"Aspect.npy")    
        except:
            Aspect = np.int16(exposition(np.float_(MNT),Csize,-9999))
            Aspect[Pente==-9999] = -9999
            np.save(Dir_temp+"Aspect",Aspect)             
        try:
            Obstacles_forwarder = np.int8(np.load(Dir_temp+"Obstacles_forwarder.npy"))
        except:
            nrows,ncols = MNT.shape[0],MNT.shape[1]
            if Dir_Obs_forwarder!="":
                Obstacles_forwarder = np.int8(prepa_obstacle_skidder(Dir_Obs_forwarder,Extent,Csize,ncols,nrows,((Route_for>0)*1+(Piste>0)*1)))
            else:
                Obstacles_forwarder = np.zeros((nrows,ncols),dtype=np.int8)
                np.save(Dir_temp+"Obstacles_forwarder",np.int8(Obstacles_forwarder)) 
    except: 
        prepa_data_fwd(Wspace,Rspace,file_MNT,file_shp_Foret,file_shp_Desserte,Dir_Obs_forwarder)
        Foret = np.int8(np.load(Dir_temp+"Foret.npy"))
        Piste = np.int8(np.load(Dir_temp+"Piste.npy"))
        Res_pub = np.int8(np.load(Dir_temp+"Res_pub.npy"))
        Route_for = np.int8(np.load(Dir_temp+"Route_for.npy"))        
        Lien_piste = np.load(Dir_temp+"Lien_piste.npy")
        Lien_RF = np.load(Dir_temp+"Lien_RF.npy")
        Pente = np.load(Dir_temp+"Pente.npy")
        Pond_pente = np.load(Dir_temp+"Pond_pente.npy")
        MNT = np.load(Dir_temp+"MNT.npy")
        Aspect = np.load(Dir_temp+"Aspect.npy")
        Obstacles_forwarder = np.int8(np.load(Dir_temp+"Obstacles_forwarder.npy"))
    
    # Generate useful variable for the process
    nrows,ncols = MNT.shape[0],MNT.shape[1]
    road_network_proj=get_proj_from_road_network(file_shp_Desserte)
    Fwd_max_inc = np.degrees(np.arctan(Forw_angle_incl*0.01))
    Fwd_max_up = np.degrees(np.arctan(Forw_angle_up*0.01))
    Fwd_max_down = np.degrees(np.arctan(Forw_angle_down*0.01))
    Pond_pente[Obstacles_forwarder==1] = 1000
    Manual_harvesting = np.int8((focal_stat_max(np.float_(Pente),-9999,1)<=Pente_max_bucheron))
    MNT_OK = np.int8((MNT!=values[5]))
    Pente_deg = np.degrees(np.arctan(Pente*0.01))
    Pente_deg[Pente==-9999]=-9999        
        
    Pente_ok_forw = np.int8((Pente_deg<=min(Fwd_max_inc,Fwd_max_up,Fwd_max_down))*(Pente_deg > -9999))    
    
    Surf_foret = np.sum((Foret==1)*MNT_OK)*Csize*Csize*0.0001
    Surf_foret_non_access = int(np.sum((Manual_harvesting==0)*(Foret==1)*MNT_OK*Csize*Csize*0.0001)+0.5)
    
    Row_line,Col_line,D_line,Nbpix_line=create_buffer_skidder(Csize,Forw_Lmax,Forw_Lmax)
    
    if file_Vol_ha != "":
        Vol_ha = load_float_raster_simple(file_Vol_ha)
        Vol_ha[np.isnan(Vol_ha)]=0
        Temp = ((Vol_ha>0)*(Foret==1)*MNT_OK)>0
        Vtot = np.mean(Vol_ha[Temp])*np.sum(Temp)*Csize*Csize*0.0001
        Temp = ((Vol_ha>0)*(Manual_harvesting==0)*(Foret==1)*MNT_OK)>0
        Vtot_non_buch = np.mean(Vol_ha[Temp])*np.sum(Temp)*Csize*Csize*0.0001
        del Vol_ha,Temp
    else:
        Vtot=0    
        Vtot_non_buch =0
       
        ArrayToGtiff(Manual_harvesting,Rspace_s+'Manual_harvesting',Extent,nrows,ncols,road_network_proj,0,'UINT8')
        console_info(QCoreApplication.translate("MainWindow","    - Initialization achieved"))   
    del Pente
    gc.collect()     
    
    ###############################################################################################################################################    
    ### Calculation of skidding distance inside the forest stands
    ###############################################################################################################################################                  
    # Identify the forest area that may be run through by the skidder
    zone_rast = Pente_ok_forw*(Foret==1)
    zone_rast[Obstacles_forwarder==1]=0
    zone_rast[Res_pub==1]=0   
    zone_rast[MNT_OK==0]=0   
    from_rast = np.int8(((Piste==1)+(Route_for==1))>0)
    from_rast[Res_pub==1]=0
    Zone_for,_ = calcul_distance_de_cout(from_rast,Pond_pente,zone_rast,Csize) 
    Zone_for[Zone_for>=0]=1
    Zone_for[from_rast==1]=1
    Zone_for=np.int8(Zone_for)
    
    # Create a buffer of Dmax_out_forest around these area taking into account slope and obstacles
    from_rast = focal_stat_nb(np.float_(Zone_for==1),0,1)
    from_rast = np.int8((from_rast<9)*(from_rast>0))
    zone_rast = np.copy(Pente_ok_forw)
    zone_rast[Obstacles_forwarder==1]=0
    zone_rast[Res_pub==1]=0   
    zone_rast[MNT_OK==0]=0   
    Zone_for2,Out_alloc = calcul_distance_de_cout(from_rast,Pond_pente,zone_rast,Csize,Forw_Dmax_out_for) 
    Pente_ok_forwarder = np.int8(Zone_for2>0)
    Pente_ok_forwarder[Zone_for==1]=1    
    
    del Zone_for,Zone_for2,Out_alloc
    gc.collect()
    
    #Stick all forest with pente_ok_skidder to the area
    from_rast = focal_stat_nb(np.float_(Pente_ok_forwarder==1),0,1)
    from_rast = np.int8((from_rast<9)*(from_rast>0))
    zone_rast = Pente_ok_forw*(Foret==1)
    zone_rast[Obstacles_forwarder==1]=0
    zone_rast[Res_pub==1]=0   
    zone_rast[MNT_OK==0]=0   
    Zone_for,Out_alloc = calcul_distance_de_cout(from_rast,Pond_pente,zone_rast,Csize) 
    Pente_ok_forwarder[Zone_for>=0]=1  
    
    # Create a buffer of Dmax_out_forest around forest
    from_rast = focal_stat_nb(np.float_(Foret==1),0,1)
    from_rast = np.int8((from_rast<9)*(from_rast>0))
    zone_rast = np.int8(Pente_deg<=max(Fwd_max_inc,Fwd_max_up,Fwd_max_down))
    zone_rast[Obstacles_forwarder==1]=0
    zone_rast[Foret==1]=0
    zone_rast[Res_pub==1]=0   
    zone_rast[MNT_OK==0]=0   
    Zone_for,Out_alloc = calcul_distance_de_cout(from_rast,Pond_pente,zone_rast,Csize,Forw_Dmax_out_for) 
    BufForest = np.int8(Zone_for>0)
           
    del Zone_for,from_rast,zone_rast,Out_alloc,Pente_ok_forw
    gc.collect()     
    
    ###############################################################################################################################################
    ### Get directly passable area from forest tracks PAFT (forwarder can reach throught the forest)
    ###############################################################################################################################################
    D_foret,L_Piste,D_piste=Dfwd_flat_forest_tracks(Lien_piste, Pond_pente,Pente_ok_forwarder*(Route_for==0), Csize)
    
    D_foret[(Foret+BufForest)==0] = -9999
    L_Piste[(Foret+BufForest)==0] = -9999
    D_piste[(Foret+BufForest)==0] = -9999
    
    ###############################################################################################################################################
    ### Get directly passable area from forest roads PAFR (forwarder can reach throught the forest)
    ###############################################################################################################################################
    RF_D,RF_L_forRF = Dfwd_flat_forest_road(Lien_RF,Pond_pente,Pente_ok_forwarder*(Piste==0),Csize)

    RF_D[(Foret+BufForest)==0] = -9999
    RF_L_forRF[(Foret+BufForest)==0] = -9999    
    
    del BufForest,Pente_ok_forwarder
    gc.collect()
    
    ###############################################################################################################################################
    ### Check forwarder inclination and terrain slope conditions from forest road network
    ###############################################################################################################################################
    from_rast = focal_stat_nb(np.float_(Foret==1),0,1)
    from_rast = np.int8((from_rast<9)*(from_rast>0))
    zone_rast = np.copy(Manual_harvesting)
    zone_rast[Obstacles_forwarder==1]=0
    zone_rast[Foret==1]=0
    zone_rast[Res_pub==1]=0   
    zone_rast[MNT_OK==0]=0   
    Zone_OK,Out_alloc = calcul_distance_de_cout(from_rast,Pond_pente,zone_rast,Csize,Forw_Dmax_out_for) 
    Zone_OK[Zone_OK>=0]=1
    Zone_OK[Zone_OK<0]=0
    Zone_OK=np.int8(Zone_OK)
    Zone_OK[Foret==1]=1
    Zone_OK[MNT_OK==0]=0
    Zone_OK[Obstacles_forwarder==1]=0
    Zone_OK[Manual_harvesting==0]=0
    
    del Out_alloc
    
    contour=np.int8((Piste+Route_for)>0)
    contour[Obstacles_forwarder==1]=0
    contour[Res_pub==1]=0   
    contour[MNT_OK==0]=0     
    pixels=np.argwhere(contour>0)
    del contour    
    
    #line=ID_contour, Y, X,Dpis,Dfor,L_RF,L_Piste    
    Lien_contour = np.zeros((pixels.shape[0]+1,6),dtype=np.int16)    
    ID = 1
    Dpis=np.zeros_like(MNT,np.int32)
    Lpis=np.zeros_like(MNT,np.int32)
    LRF=np.zeros_like(MNT,np.int32)
    for i,p in enumerate(Lien_piste[1:]):
        Dpis[p[0],p[1]]=p[2]
        Lpis[p[0],p[1]]=i+1
        LRF[p[0],p[1]]=p[4]
    for i,p in enumerate(Lien_RF[1:]):        
        LRF[p[0],p[1]]=i+1    
    
    for pixel in pixels:        
        Lien_contour[ID,0],Lien_contour[ID,1]=pixel[0],pixel[1]
        Lien_contour[ID,2],Lien_contour[ID,3]=Dpis[pixel[0],pixel[1]],0
        Lien_contour[ID,4],Lien_contour[ID,5]=LRF[pixel[0],pixel[1]],Lpis[pixel[0],pixel[1]]
        ID +=1   
    
    del Dpis,Lpis,LRF
    gc.collect()
    
    zone_rast = np.int8(Zone_OK*(Pente_deg<=max(Fwd_max_inc,Fwd_max_up,Fwd_max_down)))
        
    Dpente,L_RF,L_pis,Dpis,Dfor=fwd_azimuts_contour(Lien_contour,MNT,Aspect,Pente_deg,Row_line,Col_line,D_line,Nbpix_line,
                                                       Fwd_max_up, Fwd_max_down,Fwd_max_inc, Forw_Lmax, nrows,ncols,zone_rast)    
        
    del Dfor
    gc.collect()
    
    ###############################################################################################################################################
    ### Concatenate for passable area
    ###############################################################################################################################################   
    DTot = np.ones((nrows,ncols),dtype=np.int32)*100001
    Dforet = np.ones((nrows,ncols),dtype=np.int32)*-9999    
    Dpiste = np.ones((nrows,ncols),dtype=np.int32)*-9999
    Lien_foret_piste = np.ones((nrows,ncols),dtype=np.int32)*-9999     
    Lien_foret_RF = np.ones((nrows,ncols),dtype=np.int32)*-9999
    
    ### Get flat area from forest tracks
    Temp = (D_piste>=0)
    DTot[Temp] = D_foret[Temp]+D_piste[Temp]
    Dforet[Temp] = D_foret[Temp] 
    Dpiste[Temp] = D_piste[Temp]
    Lien_foret_piste[Temp] = L_Piste[Temp] 
       
    ### Get flat area from forest roads
    Temp =  (DTot<100001)*(RF_D>=0)
    Temp[RF_D>(Dforet+0.1*Dpiste)]=0
    DTot[Temp] = RF_D[Temp]
    Dforet[Temp] = RF_D[Temp] 
    Dpiste[Temp] = 0
    Lien_foret_RF[Temp] = RF_L_forRF[Temp]
    Temp=(DTot==100001)*(RF_D>=0)
    DTot[Temp] = RF_D[Temp]
    Dforet[Temp] = RF_D[Temp] 
    Dpiste[Temp] = 0
    Lien_foret_RF[Temp] = RF_L_forRF[Temp]
    
    contour = focal_stat_nb(np.float_(Dforet),-9999,1)
    contour = ((contour<9)*(contour>0))>0
    
    ### Get slope area from tracks and roads
    Temp =  (DTot<100001)*(Dpente>=0)
    Temp[Dpente<(Dforet+0.1*Dpiste)]=0
    DTot[Temp] = Dpente[Temp]+Dpis[Temp]
    Dforet[Temp] = 0
    Dpiste[Temp] = Dpis[Temp]
    Lien_foret_RF[Temp] = L_RF[Temp]
    Lien_foret_piste[Temp] = L_pis[Temp] 
    Temp =  (DTot==100001)*(Dpente>=0)
    DTot[Temp] = Dpente[Temp]+Dpis[Temp]
    Dforet[Temp] = 0
    Dpiste[Temp] = Dpis[Temp]
    Lien_foret_RF[Temp] = L_RF[Temp]
    Lien_foret_piste[Temp] = L_pis[Temp] 
    
    del RF_D,RF_L_forRF,Temp,D_foret,L_Piste,D_piste
    gc.collect()    
    
    console_info(QCoreApplication.translate("MainWindow","    - Directly passable area identified")) 
    
    ###############################################################################################################################################
    ### Get contour of passable area (check forwarder inclination and terrain slope conditions)
    ###############################################################################################################################################
    #Identify zone_ok taking into account non forest area    
    contour[Obstacles_forwarder==1]=0
    contour[Res_pub==1]=0  
    contour[Route_for==1]=0  
    contour[Piste==1]=0  
    contour[MNT_OK==0]=0 
    
    Temp = (Dforet>=0)*(contour==0)
    pixels=np.argwhere(contour>0)
    del contour
    
    #line=ID_contour, Y, X,Dpis,Dfor,L_RF,L_Piste    
    Lien_contour = np.zeros((pixels.shape[0]+1,6),dtype=np.int16)    
    ID = 1
    for pixel in pixels:
        Lien_contour[ID,0],Lien_contour[ID,1]=pixel[0],pixel[1]
        Lien_contour[ID,2],Lien_contour[ID,3]=Dpiste[pixel[0],pixel[1]],Dforet[pixel[0],pixel[1]]
        Lien_contour[ID,4],Lien_contour[ID,5]=Lien_foret_RF[pixel[0],pixel[1]],Lien_foret_piste[pixel[0],pixel[1]]
        ID +=1   
        
    zone_rast = np.int8(Zone_OK*(Pente_deg<=max(Fwd_max_inc,Fwd_max_up,Fwd_max_down))*(Temp==0))
        
    Dpente,L_RF,L_pis,Dpis,Dfor=fwd_azimuts_contour(Lien_contour,MNT,Aspect,Pente_deg,Row_line,Col_line,D_line,Nbpix_line,
                                                       Fwd_max_up, Fwd_max_down,Fwd_max_inc, Forw_Lmax, nrows,ncols,zone_rast)
    
    del MNT,Aspect,Pente_deg
    gc.collect()   
    
    ###############################################################################################################################################
    ### Concatenate for in slope area 
    ###############################################################################################################################################      
     
    Temp = (DTot==100001)*(Dpente+Dfor+Dpis>=0)
    DTot[Temp] = Dfor[Temp]+Dpis[Temp]+Dpente[Temp]
    Dforet[Temp] = Dfor[Temp]+Dpente[Temp]
    Dpiste[Temp] = Dpis[Temp]
    Lien_foret_piste[Temp] = L_pis[Temp] 
    Lien_foret_RF[Temp] = L_RF[Temp] 
    
    Temp = (Foret==0)
    DTot[Temp] = 100001
    Dforet[Temp] = -9999
    Dpiste[Temp] = -9999
    Lien_foret_piste[Temp] = -9999
    Lien_foret_RF[Temp] = -9999   
    
    del Dpente,Dfor,L_pis,Dpis,Temp,Lien_contour,pixels,L_RF,zone_rast
    gc.collect()    
    
    console_info(QCoreApplication.translate("MainWindow","    - Accessible area in slope identified")) 
    
    ################################################################################################################################################
    ### Calculation of area reachable with the grap
    ################################################################################################################################################                   
    # Get the contour of traversable area
    contour = focal_stat_nb(np.float_(Dforet),-9999,1)
    contour = (contour<9)*(contour>0) 
    contour[Obstacles_forwarder==1]=0
    contour[Res_pub==1]=0   
    contour[MNT_OK==0]=0 
    pixels=np.argwhere(contour>0)
    
    del contour,MNT_OK,Res_pub,Obstacles_forwarder
    gc.collect()
    
    Temp = (DTot<100001)    
    Lien_contour = np.zeros((pixels.shape[0]+1,6),dtype=np.int16)    
    ID = 1
    for pixel in pixels:
        Lien_contour[ID,0],Lien_contour[ID,1]=pixel[0],pixel[1]
        Lien_contour[ID,2],Lien_contour[ID,3]=Dforet[pixel[0],pixel[1]],Lien_foret_RF[pixel[0],pixel[1]]
        Lien_contour[ID,4],Lien_contour[ID,5]=Dpiste[pixel[0],pixel[1]],Lien_foret_piste[pixel[0],pixel[1]]
        ID +=1   
    
    zone_rast = Zone_OK*(Temp==0)   
    Dbras,Lien_RF2,Lien_piste2,Dpiste2,Dforet2=Fwd_add_contour(Lien_contour, Pond_pente,zone_rast,Forw_portee, Csize)

    ###############################################################################################################################################                                                                                    
    ### Concatenation of the resultats number 2
    ###############################################################################################################################################
    Temp =  (DTot==100001)*(Dbras+Dpiste2+Dforet2>=0)
    DTot[Temp] = Dbras[Temp]+Dpiste2[Temp]+Dforet2[Temp]
    Dforet[Temp] = Dbras[Temp] + Dforet2[Temp]
    Dpiste[Temp] = Dpiste2[Temp]
    Lien_foret_piste[Temp] = Lien_piste2[Temp]
    Lien_foret_RF[Temp] = Lien_RF2[Temp] 
    
    # Keep only results in Forest area
    Temp = (DTot==100001)
    DTot[Temp] = -9999
    Temp = (Foret==0)
    DTot[Temp] = -9999
    Dforet[Temp] = -9999
    Dpiste[Temp] = -9999
    Lien_foret_piste[Temp] = -9999
    Lien_foret_RF[Temp] = -9999
    
    del zone_rast,Zone_OK,Temp,Lien_contour,Dbras,Dpiste2,Dforet2,Lien_piste2,Lien_RF2,Pond_pente
    gc.collect()
    
    # Fill Lien foret respub and Lien foret RF
    Lien_foret_Res_pub,Lien_foret_RF,Keep=fill_Link(Lien_foret_piste,Lien_piste,Lien_RF, Lien_foret_RF, nrows,ncols)
    
    Temp = (Keep<1)*((Piste==1)+(Route_for==1))>0
    DTot[Temp] = -9999
    Dforet[Temp] = -9999
    Dpiste[Temp] = -9999
    Lien_foret_piste[Temp] = -9999
    Lien_foret_RF[Temp] = -9999
       
    Zone_accessible = np.int8(1*(DTot>=0))    
    
    del Keep,Piste,Route_for
    gc.collect()    
    console_info(QCoreApplication.translate("MainWindow","    - Area reachable with the boom added")) 
    model_name = QCoreApplication.translate("MainWindow","Forwarder")
    
    
    ###############################################################################################################################################                                                                                    
    ### CREATE SUMMARY TABLE
    ###############################################################################################################################################
    make_summary_surface_vol(Forw_Debclass,file_Vol_ha,Surf_foret,Surf_foret_non_access,Csize,DTot,Vtot,Vtot_non_buch,Rspace_s,model_name)
            
    ###############################################################################################################################################                                                                                    
    ### SAVE RASTER
    ###############################################################################################################################################    
    console_info("    - Saving output files") 
    ArrayToGtiff(DTot,Rspace_s+'Total_yarding_distance',Extent,nrows,ncols,road_network_proj,-9999,'INT16')
    ArrayToGtiff(Lien_foret_piste,Rspace_s+'Link_forest_forest_tracks',Extent,nrows,ncols,road_network_proj,-9999,'INT32')
    ArrayToGtiff(Lien_foret_RF,Rspace_s+'Link_forest_forest_road',Extent,nrows,ncols,road_network_proj,-9999,'INT32')
    ArrayToGtiff(Lien_foret_Res_pub,Rspace_s+'Link_forest_public_network',Extent,nrows,ncols,road_network_proj,-9999,'INT32')
    ArrayToGtiff(Dforet,Rspace_s+'Distance_in_forest',Extent,nrows,ncols,road_network_proj,-9999,'INT16')
    ArrayToGtiff(Dpiste,Rspace_s+'Distance_on_forest_tracks',Extent,nrows,ncols,road_network_proj,-9999,'INT16')
    ArrayToGtiff(Zone_accessible,Rspace_s+'Accessible_area',Extent,nrows,ncols,road_network_proj,0,'UINT8')
    
    layer_name = 'Forwarder_recap_accessibility'
    source_src=get_source_src(file_shp_Desserte)  
    create_access_shapefile(DTot,Rspace_s,Foret,Forw_Debclass.split(";"),road_network_proj,source_src, Dir_temp,Extent,nrows,ncols,layer_name)
       
    ###############################################################################################################################################                                                                                    
    ### SAVE PARAMETERS
    ###############################################################################################################################################    
    
    str_duree,str_fin,str_debut=heures(Hdebut)
    ### Genere le fichier avec le resume des parametres de simulation
    file_name = str(Rspace_s)+"Parameters_of_simulation.txt"
    resume_texte = QCoreApplication.translate("MainWindow","Sylvaccess : AUTOMATIC MAPPING OF FOREST ACCESSIBILITY WITH FORWARDER\n\n\n")
    ver = "0.2"
    date = "2024/02"
    resume_texte += QCoreApplication.translate("MainWindow","Software version :") + ver + QCoreApplication.translate("MainWindow"," - ", "skidder_results") + date + "\n\n"
    resume_texte += QCoreApplication.translate("MainWindow","Resolution       : ")+str(Csize)+" m\n\n"
    resume_texte += "" .join (["_"]*80)+"\n\n"
    resume_texte += QCoreApplication.translate("MainWindow","Date and time when launching the script:              ")+str_debut+"\n"
    resume_texte += QCoreApplication.translate("MainWindow","Date and time at the end of execution of the script:  ")+str_fin+"\n"
    resume_texte += QCoreApplication.translate("MainWindow","Total execution time of the script:                   ")+str_duree+"\n\n"
    resume_texte += "" .join (["_"]*80)+"\n\n"
    resume_texte += QCoreApplication.translate("MainWindow","PARAMETERS USED FOR THE MODELING:\n\n")
    resume_texte += QCoreApplication.translate("MainWindow","   - Maximum perpendicular lateral inclination (MPLI):            ")+str(Forw_angle_incl)+" %\n"
    resume_texte += QCoreApplication.translate("MainWindow","   - Maximum slope for an uphill yarding:                         ")+str(Forw_angle_up)+" %\n"
    resume_texte += QCoreApplication.translate("MainWindow","   - Maximum slope for an downhill yarding:                       ")+str(Forw_angle_down)+" %\n"
    resume_texte += QCoreApplication.translate("MainWindow","   - Boom reach:                                                  ")+str(Forw_portee)+" m\n"
    resume_texte += QCoreApplication.translate("MainWindow","   - Maximum yarding distance when terrain slope > MPLI:          ")+str(Forw_Lmax)+" m\n"
    resume_texte += QCoreApplication.translate("MainWindow","   - Maximum slope for a free access of the parcels with skidder: ")+str(Forw_angle_incl)+" %\n"
    resume_texte += QCoreApplication.translate("MainWindow","   - Maximum slope for manual felling of the trees:               ")+str(Pente_max_bucheron)+" %\n"       
    
    if os.path.exists(Rspace_s+"Forest_tracks_not_connected.tif"):
        resume_texte += "\n\n"
        resume_texte += "" .join (["-"]*80)+"\n\n"
        resume_texte += QCoreApplication.translate("MainWindow","      !!! Warning !!! Some forest tracks are not connected to public network.\n" ) 
        resume_texte += QCoreApplication.translate("MainWindow","      They were removed from the analysis.\n")  
    if os.path.exists(Rspace_s+"Forest_road_not_connected.tif"):
        resume_texte += "" .join (["-"]*80)
        resume_texte += QCoreApplication.translate("MainWindow","\n\n      !!! Warning !!! Some forest roads are not connected to public network.\n")      
    
    fichier = open(file_name, "w")
    fichier.write(resume_texte)
    fichier.close()
    console_info(QCoreApplication.translate("MainWindow","Forwarder accessibility processed"))

    ##############################################################################################################################################
    ### Close the script
    ##############################################################################################################################################
    clear_big_nparray()