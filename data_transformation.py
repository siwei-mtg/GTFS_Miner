import pandas as pd
import numpy as np
import os
import datetime
import logging
from .baseline_functions import *
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.vq import kmeans, kmeans2
from scipy import cluster
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

logger = logging.getLogger('GTFS_miner')

# transformation 1.1
def service_date_generate(calendar,calendar_dates,dates):
    calendar['start_date'] = pd.to_datetime(calendar['start_date'],format="%Y%m%d")
    calendar['end_date'] = pd.to_datetime(calendar['end_date'],format="%Y%m%d")
    calendar_dates['date'] = pd.to_datetime(calendar_dates['date'],format="%Y%m%d")
    dates['Date_GTFS'] = pd.to_datetime(dates['Date_GTFS'],format="%Y%m%d")
    dates_small = dates[['Date_GTFS','Type_Jour']]

    if calendar is None:
        logger.warning("Le fichier calendar n'est pas présent dans le jeu de données")
        result = calendar_dates.merge(dates, left_on = 'date', right_on = 'Date_GTFS',how = 'left')[cal_cols].sort_values(['service_id','Date_GTFS']).reset_index(drop = True)
    elif sum(calendar['monday']) + sum(calendar['tuesday']) + sum(calendar['wednesday']) + sum(calendar['thursday']) + sum(calendar['friday']) + sum(calendar['saturday']) + sum(calendar['sunday'])  ==0:
        logger.warning("Le fichier calendar n'a pas de jour de semaine valide.")
        result = calendar_dates.merge(dates, left_on = 'date', right_on = 'Date_GTFS',how = 'left')[cal_cols].sort_values(['service_id','Date_GTFS']).reset_index(drop = True)
    else :
        logger.info("Les fichiers calendar et calendar_dates sont bien présents.")
        date_range_per_service = []
        for _, row in calendar.iterrows():
            dates_service = dates_small.loc[(dates_small['Date_GTFS'] >= row['start_date']) & (dates_small['Date_GTFS'] <= row['end_date']) ].copy()
            dates_service['service_id'] = row['service_id']
            date_range_per_service.append(dates_service)

        date_range_per_service = pd.concat(date_range_per_service, ignore_index=True)
        calendar.drop(['start_date','end_date'], axis= 1, inplace=True)
        calendar_range_date = pd.merge(calendar,date_range_per_service, on = 'service_id' )

        # List of days and their corresponding Type_Jour values
        days_of_week = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        day_type_jour_mapping = {1: 'monday', 2: 'tuesday', 3: 'wednesday', 4: 'thursday', 5: 'friday', 6: 'saturday', 7: 'sunday'}

        # Initialize the mask
        mask = False
        # Loop through each day and create the mask
        for type_jour, day in day_type_jour_mapping.items():
            mask |= (calendar_range_date['Type_Jour'] == type_jour) & (calendar_range_date[day] == 1)

        # Apply the mask to filter the DataFrame
        calendar_range_clean = calendar_range_date.loc[mask]
        calendar_range_clean.drop(days_of_week,axis=1,inplace=True)
        calendar_full_join = pd.merge(calendar_range_clean, calendar_dates, left_on=['service_id','Date_GTFS'],right_on=['service_id','date'], how='outer')
        calendar_full_join.loc[calendar_full_join['exception_type']==1,'Date_GTFS'] = calendar_full_join.loc[calendar_full_join['exception_type']==1,'date'] 
        calendar_with_exceptions = calendar_full_join.loc[calendar_full_join['exception_type'] != 2]
        calendar_with_exceptions.drop(['Type_Jour','date','exception_type'],axis=1,inplace=True)
        calendar_with_exceptions.sort_values(['service_id','Date_GTFS'],inplace=True)
        service_dates = calendar_with_exceptions.merge(dates, on = 'Date_GTFS')
        cal_cols = ['service_id','Date_GTFS','Type_Jour','Semaine','Mois','Annee',
                    'Type_Jour_Vacances_A','Type_Jour_Vacances_B','Type_Jour_Vacances_C']
        result = service_dates[cal_cols]
        result.rename({'service_id':'id_service'}, axis=1,inplace=True)
    return result

# transformation 1.2
def ag_ap_generate_hcluster(raw_stops):
    '''In cases of non-existance of parent station, or incomplete parent station grouping, create AG by hcluster'''
    AP = raw_stops.loc[raw_stops.location_type == 0,:].reset_index(drop=True)
    AP_coor = AP.loc[:,['stop_lon','stop_lat']].to_numpy()

    distmatrix = pdist(AP_coor, lambda u, v: getDistanceByHaversine(u,v))
    cut = cluster.hierarchy.cut_tree(linkage(distmatrix, method='complete'), height = 100)
    AP['id_ag'] = cut + 1
    # AP['id_ag_num'] =  AP['id_ag']+10000
    AP['id_ag'] = AP['id_ag'].astype(str)
    # AP['id_ap_num'] = np.arange(1, len(AP) + 1)+100000

    AG = AP.groupby(
                 ['id_ag'], as_index = False).agg(
                 {'stop_name':'first',
                  'stop_lat' : 'mean',
                  'stop_lon' : 'mean'}, as_index = False).reset_index(drop=True)
    return AP,AG

def ag_ap_generate_asit(raw_stops):
    '''When parent station exist in good form, take the existing AG'''
    AG = raw_stops.loc[raw_stops.location_type == 1].drop(
        ['parent_station','location_type'], axis = 1).rename(
        {'stop_id':'id_ag'}, axis = 1).groupby(
        ['id_ag'], as_index = False).agg(
        {'stop_name':'first',
         'stop_lat' : 'mean',
         'stop_lon' : 'mean'}, as_index = False).reset_index(drop=True)
    AP = raw_stops.loc[raw_stops.location_type == 0].drop(
        ['location_type'], axis = 1).reset_index(drop=True).rename({'stop_id':'id_ap', 'parent_station':'id_ag'}, axis = 1)
    # AP['id_ap_num'] = np.arange(1, len(AP) + 1)+100000
    # AG['id_ag_num']= np.arange(1, len(AG) + 1)+10000
    AP = AP.merge(AG[['id_ag']], how = 'left',
                  on='id_ag', suffixes=('','_y'))
    return AP, AG

def ag_ap_generate_bigvolume(rawstops):
    AP = rawstops.loc[rawstops.location_type ==0,:].reset_index(drop=True)
    ap_coor = AP.loc[:,['stop_lon','stop_lat']].to_numpy()
    kcentroids = round(len(ap_coor)/500)
    id_centroid = kmeans2(ap_coor, kcentroids, minit = 'points')[1]
    AP['kmean_id'] = id_centroid
    for i in range(kcentroids):
        AP_kmeaned_coor = AP.loc[AP.kmean_id==i,['stop_lat','stop_lon']]
        AP_kmeaned_coornp = AP_kmeaned_coor.to_numpy()
        distmat = distmatrice(AP_kmeaned_coornp)
        distmat_cutree = cluster.hierarchy.cut_tree(linkage(distmat, method='complete'), height = 100)
        AP.loc[AP.kmean_id==i,'clust_id'] = distmat_cutree
    AP['id_ag'] = AP['kmean_id'].astype(str) + '_' + AP['clust_id'].astype(int).astype(str)
    # AP['id_ap_num'] = np.arange(1, len(AP) + 1)+100000
    AG = AP.groupby(
                 ['id_ag'], as_index = False).agg(
                 {'stop_name':'first',
                  'stop_lat' : 'mean',
                  'stop_lon' : 'mean'}, as_index = False).reset_index(drop=True)
    # AG['id_ag_num'] = np.arange(1, len(AG) + 1)+100000
    AP = AP.merge(AG[['id_ag']])
    return AP, AG

def ag_ap_generate_reshape(raw_stops):
    nb_location_type = len(raw_stops.location_type.unique())
    ap_not_in_any_ag = raw_stops[raw_stops['location_type'] == 0]['parent_station'].isnull().sum()
    if nb_location_type == 1:
        AP,AG = ag_ap_generate_hcluster(raw_stops)
        marker = 'cluster méthode'

    elif nb_location_type >= 2:
        if ap_not_in_any_ag == 0 :
            AP,AG = ag_ap_generate_asit(raw_stops)
            marker = 'original parent station'
        elif ap_not_in_any_ag > 0:
            ap_potentiel = len(raw_stops.loc[raw_stops['location_type'] == 0,:])
            if ap_potentiel <5000:
                AP,AG = ag_ap_generate_hcluster(raw_stops)
                marker = 'cluster méthode'
            else:
                AP, AG = ag_ap_generate_bigvolume(raw_stops)
                marker = 'cluster méthode pour grand volume'
    AP = AP.rename({'stop_id':'id_ap'},axis = 1)
    AP.dropna(axis = 'columns', how = 'all', inplace = True)
    AG.dropna(axis = 'columns', how = 'all', inplace = True)
    return AP,AG,marker

def itineraire_generate(raw_stoptimes, AP, raw_trips):
    stop_times = raw_stoptimes.rename({'stop_id':'id_ap'},axis = 1).dropna(axis = 'columns', how = 'all')
    stop_times['TH'] = np.vectorize(str_time_hms_hour)(stop_times.arrival_time)
    stop_times['TH'] = np.where(stop_times['TH']>=24, stop_times['TH']-24,stop_times['TH'])
    stop_times['arrival_time'] = np.vectorize(str_time_hms)(stop_times.arrival_time)
    stop_times['departure_time'] = np.vectorize(str_time_hms)(stop_times.departure_time)
    itnry_1 = stop_times.merge(AP[['id_ap','id_ag','stop_lon','stop_lat']], how = 'left', on = 'id_ap')
    itnry_2 = itnry_1.merge(raw_trips[['trip_id', 'route_id', 'service_id','direction_id']], how = 'right', on ='trip_id')
    itineraire = itnry_2[['trip_id','route_id','service_id','direction_id','stop_sequence',
                          'id_ap','id_ag','stop_lon','stop_lat','arrival_time','departure_time','TH']]
    itineraire['stop_sequence'] = itineraire.groupby(['trip_id']).cumcount()+1
    if itineraire['direction_id'].isnull().sum() >0:
        itineraire['direction_id'] = 999
    return itineraire

def itiarc_generate(itineraire,AG):
    iti_arc_R = itineraire.copy().drop(['departure_time','service_id','route_id','stop_lon','stop_lat'], axis = 1)
    iti_arc_L = itineraire.copy().drop(['arrival_time','stop_lon','stop_lat'], axis = 1)
    iti_arc_L['ordre_b'] = iti_arc_R['stop_sequence'] +1
    iti_arc = iti_arc_L.merge(iti_arc_R, how ='left', left_on = ['trip_id','ordre_b','direction_id'], right_on = ['trip_id', 'stop_sequence','direction_id'], suffixes = ('_a','_b')).dropna(subset=['id_ag_b']).reset_index(drop = True)
    iti_arc_dist = iti_arc.merge(AG[['id_ag','stop_lon','stop_lat']], left_on = 'id_ag_a',right_on = 'id_ag',how = 'left').merge(AG[['id_ag','stop_lon','stop_lat']], left_on = 'id_ag_b',right_on = 'id_ag',how = 'left')
    cols = ['trip_id','route_id','service_id','direction_id','stop_sequence_a','departure_time','id_ap_a','id_ag_a','TH_a','stop_sequence_b','arrival_time','id_ap_b', 'id_ag_b','TH_b']
    iti_arc_f = iti_arc_dist[cols].rename({'stop_sequence_a':'ordre_a','arrival_time':'heure_arrive',
                                      'stop_sequence_b':'ordre_b','departure_time':'heure_depart'},axis =1)
    return iti_arc_f

def course_generate(itineraire):
    course = itineraire.groupby(
    ['route_id', 'service_id','trip_id','direction_id'], as_index=False).agg(
    {'arrival_time': 'min',
    'departure_time': 'max',
    'id_ap': ['first', 'last'],
    'id_ag': ['first', 'last'],
    'stop_sequence': 'max'})
    course.columns = course.columns.map(''.join)
    course.rename({'arrival_timemin':'heure_depart', 'departure_timemax':'heure_arrive',
                'id_apfirst':'id_ap_origine', 'id_aplast':'id_ap_destination',
                'id_agfirst':'id_ag_origine', 'id_aglast':'id_ag_destination',
                'stop_sequencemax':'nb_arrets'}, axis = 1, inplace = True)
    return course

# Create tracés
def shape_exist_condition(gtfs_norm_dict, course,itineraire):
    iti_geom = itineraire_geometry_generate(itineraire)
    course = add_iti_geom_to_course(iti_geom, course)
    course['Dist_Vol_Oiseau'] = course['Dist_Vol_Oiseau'].round(0)
    course = course_enrich_sl_th(course)
    # if shape exist
    if gtfs_norm_dict['shapes'] is not None:
        logger.info("Le fichier shapes est présent.")
        shapes_with_dist = shapes_geometry_generate(gtfs_norm_dict['shapes'])
        # shape info added to course
        course = add_shape_info_to_courses(shapes_with_dist, course, gtfs_norm_dict['trips'])
        course['Dist_shape'] = course['Dist_shape'].round(0)
    else:
        logger.info("Le fichier shapes n'est pas présent.")
    return course

def shapes_geometry_generate(shapes):
    # Group by shape_id and create LineString geometry
    shapes_geom = shapes.groupby('shape_id').apply(lambda x: create_shapes_linestring_by_group(x)).reset_index()
    shapes_geom = shapes_geom.rename(columns={0: 'geom_shape'})  # Rename the geometry column  
    # Create GeoDataFrame
    shapes_gdf = gpd.GeoDataFrame(shapes_geom, geometry='geom_shape')   
    # Set CRS (WGS84)
    shapes_gdf.crs = "EPSG:4326"    
    # Project to Lambert-93 (EPSG:2154)
    shapes_gdf_projected = shapes_gdf.to_crs(epsg=2154)   
    # Calculate shape length
    shapes_gdf_projected['Dist_shape'] = shapes_gdf_projected.length   
    return shapes_gdf_projected

def create_shapes_linestring_by_group(group):
    group = group.sort_values(by='shape_pt_sequence')
    return LineString(zip(group['shape_pt_lon'],group['shape_pt_lat']))

def add_shape_id_to_course(course, rawtrip):
    trip_shape_corr= rawtrip[['trip_id','shape_id']]
    result = course.merge(trip_shape_corr, on = 'trip_id',how= 'left')
    nb_course_sans_shape = len(result.loc[result['shape_id'].isnull(),'trip_id'])
    if nb_course_sans_shape>0:
        logger.warning(f"{nb_course_sans_shape} courses n'ont pas de shape_id.")
    return result

def add_shape_info_to_courses(shapes_with_dist, course, rawtrip):
    course = add_shape_id_to_course(course,rawtrip)
    course['shape_id'] = course['shape_id'].astype(str)
    shapes_with_dist['shape_id'] = shapes_with_dist['shape_id'].astype(str)
    result = pd.merge(course,shapes_with_dist, on = 'shape_id', how='left')
    return result

# Create tracé si shape n'existe pas
def create_itineraire_linestring_by_group(group):
    group = group.sort_values(by='stop_sequence')
    return LineString(zip(group['stop_lon'],group['stop_lat']))

def itineraire_geometry_generate(itineraire):
    # Group by shape_id and create LineString geometry
    iti_geom = itineraire.groupby('trip_id').apply(lambda x: create_itineraire_linestring_by_group(x)).reset_index()
    iti_geom = iti_geom.rename(columns={0: 'geom_vo'})  # Rename the geometry column  
    # Create GeoDataFrame
    iti_gdf = gpd.GeoDataFrame(iti_geom, geometry='geom_vo')   
    # Set CRS (WGS84)
    iti_gdf.crs = "EPSG:4326"    
    # Project to Lambert-93 (EPSG:2154)
    iti_gdf_projected = iti_gdf.to_crs(epsg=2154)   
    iti_gdf_projected['Dist_Vol_Oiseau'] = iti_gdf_projected.length
    return iti_gdf_projected

def add_iti_geom_to_course(iti_geom, course):
    iti_geom[['trip_id','Dist_Vol_Oiseau','geom_vo']]
    return pd.merge(course,iti_geom, on = 'trip_id', how='left')

def course_enrich_sl_th(course):
    course['sous_ligne'] = course[['route_id', 'direction_id', 'id_ag_origine', 'id_ag_destination', 'nb_arrets', 'Dist_Vol_Oiseau']].astype(str).agg('_'.join, axis=1)  
    course['TH'] = np.floor(course['heure_depart']*24)
    cols = ['trip_id','route_id','sous_ligne', 'service_id',
            'direction_id',	'id_ap_origine','id_ag_origine', 
     'heure_depart','id_ap_destination','id_ag_destination','heure_arrive',	'nb_arrets','TH',
     'Dist_Vol_Oiseau','geom_vo']  
    course = course[cols]
    return course

def sl_generate(course,AG):
    AG = AG[['id_ag','stop_name']]
    course = course.merge(AG, left_on = 'id_ag_origine', right_on = 'id_ag').merge(AG, left_on = 'id_ag_destination', right_on = 'id_ag')
    if course.get('geom_shape') is None:
        gdf_course = gpd.GeoDataFrame(course, geometry = 'geom_vo')
        cols = ['sous_ligne', 'route_id', 'direction_id','id_ag_origine','stop_name_x','id_ag_destination','stop_name_y','geom_vo']  
    else: 
        gdf_course = gpd.GeoDataFrame(course, geometry = 'geom_shape')
        cols = ['sous_ligne', 'route_id', 'direction_id','id_ag_origine','stop_name_x','id_ag_destination','stop_name_y', 'geom_shape']  
    group_col = ['sous_ligne', 'route_id','id_ag_origine','stop_name_x', 'id_ag_destination','stop_name_y', 'direction_id','nb_arrets']
    gdf_sl = gdf_course.dissolve(by = group_col).reset_index()
    gdf_sl = gdf_sl[cols]
    gdf_sl.rename({'route_id':'id_ligne',
                   'stop_name_x':'ag_origin_name',
                   'stop_name_y':'ag_destination_name'}, axis = 1, inplace = True)
    return gdf_sl

def ligne_generate(sl_gdf,raw_routes ,route_type):
    if sl_gdf.get('geom_shape') is None:
        col = ['id_ligne','geom_vo']
        ligne_gdf = sl_gdf[col].dissolve(by = 'id_ligne').reset_index()
        ligne_gdf.rename({'geom_vo':'geom'},axis =1,inplace = True)
    else:
        col = ['id_ligne','geom_shape']
        ligne_gdf = sl_gdf[col].dissolve(by = 'id_ligne').reset_index()
        ligne_gdf.rename({'geom_shape':'geom'},axis =1,inplace = True)
    raw_routes.rename({'route_id':'id_ligne'}, axis = 1, inplace = True)
    lignes = pd.merge(raw_routes,ligne_gdf, on = 'id_ligne')
    lignes = pd.merge(lignes, route_type[['Code','Description']], how='left',left_on='route_type', right_on='Code')
    lignes.drop(['Code'], axis = 1,inplace=True)
    lignes.dropna(axis = 1, how = 'all', inplace = True)
    return lignes


