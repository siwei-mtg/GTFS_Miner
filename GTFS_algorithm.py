import pandas as pd
import sys
import pandas as pd
import numpy as np
import os
import chardet
import datetime as dt
from datetime import date, datetime
import time
import scipy as sp
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.cluster.vq import kmeans, kmeans2
from scipy import cluster
import xlrd
from zipfile import ZipFile
import matplotlib.pyplot as plt
import shutil
import pyodbc
import re
import math

def norm_upper_str(pd_series):
    normed = pd_series.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').str.upper()
    return normed

def str_time_hms_hour(hms):
    h = int(hms.split(':')[0])
    return h

def str_time_hms(hms):
    h,m,s = hms.split(':')
    time = int(h)/24 + int(m)/24/60 + int(s)/24/3600
    return time

def get_sec(input_timedelta):
    sec_list = [input_timedelta[i].total_seconds() for i in range(len(input_timedelta))]
    return sec_list

def get_time_now(datetime_now):
    time_now = '{}:{}:{}'.format(datetime_now.hour, datetime_now.minute, datetime_now.second)
    return time_now

def encoding_guess(acces):
    '''
    detect and return file encoding
    '''
    with open(acces, 'rb') as rawdata:
        encod = chardet.detect(rawdata.read(10000))
    return encod

def rawgtfs(dirpath) :
    '''
    Read raw GTFS data from given directory
    '''
    listfile = os.listdir(dirpath)
    filename = [os.path.splitext(i)[0] for i in listfile]
    filepath = [dirpath +'/' + f for f in listfile]
    rawgtfs = {a:0 for a in filename}
    for i in range(len(filepath)):
        df = pd.read_csv(filepath[i])
        rawgtfs[filename[i]] = df
    return rawgtfs

def raw_from_zip(zippath):
    with ZipFile(zippath, "r") as zfile:
        list_file = [name for name in zfile.namelist()]
    filename = [i.split('.')[0] for i in list_file]
    rawgtfs = {a:0 for a in filename}
    for num, name in enumerate(list_file):
        df = pd.read_csv(zf.open(name))
        rawgtfs[filename[num]] = df
    return rawgtfs

def read_date(plugin_path):
    '''
    Read calendar info for plugin
    '''
    Dates = pd.read_excel((plugin_path+"/Resources/Calendrier.xls"),
        sheet_name='Dates', parse_dates=['Date_Num'], dtype={'Type_Jour': 'int32'})
    #encoding='utf-8'
    Dates.drop(['Date_Num','Date_Opendata', 'Ferié', 'Vacances_A', 'Vacances_B', 'Vacances_C',
            'Concat_Select_Type_A', 'Concat_Select_Type_B', 'Concat_Select_Type_C','Type_Jour_IDF','Annee_Scolaire'],axis = 1 , inplace = True)
    return Dates

def read_validite(plugin_path):
    '''
    Read validite for plugin
    '''
    validite = pd.read_csv((plugin_path+ "/Resources/Correspondance_Validite.txt"),
        encoding=encoding_guess(plugin_path+"/Resources/Correspondance_Validite.txt")['encoding'],
                        sep = ';', dtype={'valid_01': str, 'valid': 'int32'})
    return validite

def read_input(dirpath,plugin_path):
    '''Combinasion of read_calendar and read_validité'''
    rawGTFS = rawgtfs(dirpath)
    GTFS_norm = gtfs_normalize(rawGTFS)
    Dates = read_date(plugin_path)
    validite = read_validite(plugin_path)
    return GTFS_norm, Dates, validite

def getDistanceByHaversine(loc1, loc2):
    '''Haversine formula - give coordinates as a 2D numpy array of
    (lat_denter link description hereecimal,lon_decimal) pairs'''
    #
    # "unpack" our numpy array, this extracts column wise arrays
    EARTHRADIUS = 6371000.0

    lat1 = loc1[1]
    lon1 = loc1[0]
    lat2 = loc2[1]
    lon2 = loc2[0]
    #
    # convert to radians ##### Completely identical
    lon1 = lon1 * np.pi / 180.0
    lon2 = lon2 * np.pi / 180.0
    lat1 = lat1 * np.pi / 180.0
    lat2 = lat2 * np.pi / 180.0
    #
    # haversine formula #### Same, but atan2 named arctan2 in numpy
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2.0))**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
    m = EARTHRADIUS * c
    return m

def getDistHaversine(lat1,lon1,lat2,lon2):
    '''Haversine formula - give coordinates as a 2D numpy array of
    (lat_denter link description hereecimal,lon_decimal) pairs'''
    EARTHRADIUS = 6371000.0
    lon1 = lon1* np.pi / 180.0
    lon2 = lon2* np.pi / 180.0
    lat1 = lat1* np.pi / 180.0
    lat2 = lat2* np.pi / 180.0
    # haversine formula #### Same, but atan2 named arctan2 in numpy
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2.0))**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
    m = (EARTHRADIUS * c)
    return m

def getDistHaversine2(lat1,lon1,lat2,lon2):
    '''Haversine formula - give coordinates as a 2D numpy array of
    (lat_denter link description hereecimal,lon_decimal) pairs'''
    EARTHRADIUS = 6371000.0
    lon1 = lon1* np.pi / 180.0
    lon2 = lon2* np.pi / 180.0
    lat1 = lat1* np.pi / 180.0
    lat2 = lat2* np.pi / 180.0
    # haversine formula #### Same, but atan2 named arctan2 in numpy
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (np.sin(dlat/2))**2 + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon/2.0))**2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0-a))
    m = EARTHRADIUS * c
    return m

def distmatrice(nparray):
    distmatrix = pdist(nparray, lambda u, v: getDistanceByHaversine(u,v))
    return distmatrix

def heure_goal(horaire_excel):
    horaire = f'H{int(math.modf(horaire_excel*24)[1]):02}{int(math.modf(horaire_excel*24)[0]*60):02}'
    return horaire

def agency_norm(raw_agency):
    agency_v =  pd.DataFrame(columns = ['agency_id', 'agency_name', 'agency_url', 'agency_timezone',
            'agency_lang', 'agency_phone', 'agency_fare_url', 'agency_email'], index = None)
    agency = agency_v.append(raw_agency)
    agency.dropna(axis = 1, how = 'all', inplace =True)
    return agency

def stops_norm(raw_stops):
    stops_v =  pd.DataFrame(columns = ['stop_id', 'stop_code', 'stop_name', 'stop_desc',
        'stop_lat', 'stop_lon', 'zone_id', 'stop_url','location_type', 'parent_station',
        'stop_timezone','wheelchair_boarding','level_id','platform_code'], index = None)
    stops = stops_v.append(raw_stops)
    stops.stop_id = stops.stop_id.astype(str)
    try:
        stops.stop_lat = stops.stop_lat.astype(np.float32)
        stops.stop_lon = stops.stop_lon.astype(np.float32)
    except ValueError:
        #Le cas de Mulhouse
        stops.stop_lat = stops.stop_lat.str.strip().replace('', '0')
        stops.stop_lon = stops.stop_lat.str.strip().replace('', '0')
        stops.stop_lat = stops.stop_lat.astype(np.float32)
        stops.stop_lon = stops.stop_lon.astype(np.float32)
    stops.stop_name = norm_upper_str(stops.stop_name)
    any_nan = pd.isnull(stops.location_type.unique()).any()
    if any_nan:
        stops.location_type = stops.location_type.fillna(0).astype(np.int8)
    else:
        stops.location_type = stops.location_type.astype(np.int8)
    try:
        stops.parent_station = nan_in_col_workaround(stops.parent_station)
    except ValueError:
        stops.parent_station = stops.parent_station
    stops_essentials = ['stop_id','stop_lat','stop_lon','stop_name','location_type','parent_station']
    stops = stops[stops_essentials]
    return stops

def routes_norm(raw_routes):
    routes_v = pd.DataFrame(columns = ['route_id', 'agency_id', 'route_short_name', 'route_long_name',
        'route_desc', 'route_type', 'route_url', 'route_color','route_text_color', 'route_sort_order',
        'continuous_pickup','continuous_drop_off'], index = None)
    routes = routes_v.append(raw_routes)
    routes.drop(['route_url', 'route_sort_order',
                 'continuous_pickup','continuous_drop_off'],axis = 1, inplace =True)
    routes.dropna(axis = 1, inplace = True, how = 'all')
    routes.route_id = routes.route_id.astype(str)
    routes.route_type = routes.route_type.astype(np.int8)
    #routes.agency_id = routes.agency_id.astype(str)
    #routes.route_short_name = routes.route_short_name.astype(str)
    #routes.route_long_name = routes.route_long_name.astype(str)
    #routes.route_color = routes.route_color.astype(str)
    #routes.route_text_color = routes.route_text_color.astype(str)
    routes['id_ligne_num'] = np.arange(1, len(routes) + 1)
    return routes

def trips_norm(raw_trips):
    trips_v =  pd.DataFrame(
        columns = ['route_id', 'service_id', 'trip_id', 'trip_headsign','trip_short_name','direction_id',
                  'block_id','shape_id','wheelchair_accessible' ,'bikes_allowed'], index = None)
    trips = trips_v.append(raw_trips)
    trips.drop(['trip_headsign','trip_short_name',
                'block_id','wheelchair_accessible' ,'bikes_allowed'],axis = 1, inplace =True)
    tps_cols = ['route_id','service_id','trip_id']
    trips[tps_cols] = trips[tps_cols].apply(lambda x:x.astype(str))
    if pd.isnull(trips.shape_id).all():
        trips.drop('shape_id',axis = 1, inplace = True)
    else:
        trips.shape_id = trips.shape_id.astype(str)
    trips['id_course_num'] = np.arange(1, len(trips) + 1)
    return trips

def stop_times_norm(raw_stoptimes):
    stop_times_v =  pd.DataFrame(columns = ['trip_id', 'arrival_time', 'departure_time','stop_id', 'stop_sequence',
        'stop_headsign', 'pickup_type', 'drop_off_type', 'continuous_pickup',
        'continuous_drop_off', 'shape_dist_traveled','timepoint'], index = None)
    stop_times = stop_times_v.append(raw_stoptimes)
    stop_times.drop(['stop_headsign', 'pickup_type', 'drop_off_type', 'continuous_pickup',
                     'continuous_drop_off', 'timepoint'], axis = 1,  inplace =True)
    stp_t_cols = ['trip_id','arrival_time','departure_time','stop_id']
    stop_times[stp_t_cols] = stop_times[stp_t_cols].apply(lambda x:x.astype(str))
    stop_times.stop_sequence = stop_times.stop_sequence.astype(np.int8)
    stop_times.shape_dist_traveled = stop_times.shape_dist_traveled.astype(np.float32)
    stop_times.dropna(how = 'all', axis = 1, inplace = True)
    return stop_times

def calendar_norm(raw_cal):
    calendar_v = pd.DataFrame(
        columns =['service_id', 'monday', 'tuesday', 'wednesday',
                  'thursday', 'friday','saturday', 'sunday', 'start_date', 'end_date'], index = None)
    calendar = calendar_v.append(raw_cal)
    calendar.service_id = calendar.service_id.astype(str)
    week_cols = ['monday', 'tuesday','wednesday','thursday','friday','saturday', 'sunday']
    calendar[week_cols] = calendar[week_cols].apply(lambda x: x.astype(np.bool8))
    calendar.start_date = calendar.start_date.astype(np.int32)
    calendar.end_date = calendar.end_date.astype(np.int32)
    return calendar

def cal_dates_norm(raw_caldates):
    calendar_dates_v = pd.DataFrame(columns =['service_id', 'date', 'exception_type'], index = None)
    calendar_dates = calendar_dates_v.append(raw_caldates)
    calendar_dates.date = calendar_dates.date.astype(np.int32)
    calendar_dates.service_id = calendar_dates.service_id.astype(str)
    calendar_dates.exception_type = calendar_dates.exception_type.astype(np.int8)
    return calendar_dates

def gtfs_normalize(rawgtfs):
    '''
    adapt gtfs data to standard form. Give a dict with normalized gtfs raw data
    '''
    agency = agency_norm(rawgtfs['agency'])
    routes = routes_norm(rawgtfs['routes'])
    route_id_coor = routes[['route_id','id_ligne_num']]
    stops = stops_norm(rawgtfs['stops'])
    trips = trips_norm(rawgtfs['trips'])
    trip_id_coor = trips[['trip_id','id_course_num']]
    trips = trips.merge(route_id_coor, on = 'route_id').drop('route_id',axis =1)
    service_id = trips.service_id.unique()
    ser_id_coor = pd.DataFrame(service_id,columns = ['service_id'])
    ser_id_coor['id_service_num'] = np.arange(1, len(ser_id_coor) + 1)
    trips = trips.merge(ser_id_coor, on ='service_id').drop(['service_id'],axis = 1)
    stop_times = stop_times_norm(rawgtfs['stop_times'])
    # id num stop_times
    stop_times = stop_times.merge(trip_id_coor, on = 'trip_id').drop('trip_id',axis =1)
    trips = trips.drop(['trip_id'],axis = 1)
    try :
        calendar = calendar_norm(rawgtfs['calendar'])
        if len(calendar)==0:
            calendar = None
    except KeyError:
        calendar = None
    calendar = calendar.merge(ser_id_coor, on ='service_id').drop(['service_id'],axis = 1)
    calendar_dates = cal_dates_norm(rawgtfs['calendar_dates'])
    calendar_dates = calendar_dates.merge(ser_id_coor, on ='service_id').drop(['service_id'],axis = 1)
    return {'agency' :agency, 'stops' :stops, 'stop_times':stop_times,'routes': routes,'trips': trips, 'calendar': calendar,'calendar_dates': calendar_dates,'route_id_coor':route_id_coor,'trip_id_coor':trip_id_coor,'ser_id_coor':ser_id_coor}

def ag_ap_generate_hcluster(raw_stops):
    '''In cases of non-existance of parent station, or incomplete parent station grouping, create AG by hcluster'''
    AP = raw_stops.loc[raw_stops.location_type == 0,:].reset_index(drop=True)
    AP_coor = AP.loc[:,['stop_lon','stop_lat']].to_numpy()

    distmatrix = pdist(AP_coor, lambda u, v: getDistanceByHaversine(u,v))
    cut = cluster.hierarchy.cut_tree(linkage(distmatrix, method='complete'), height = 100)
    AP['id_ag'] = cut + 1
    AP['id_ag_num'] =  AP['id_ag']+10000
    AP['id_ag'] = AP['id_ag'].astype(str)
    AP['id_ap_num'] = np.arange(1, len(AP) + 1)+100000

    AG = AP.groupby(
                 ['id_ag', 'id_ag_num'], as_index = False).agg(
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
    AP['id_ap_num'] = np.arange(1, len(AP) + 1)+100000
    AG['id_ag_num']= np.arange(1, len(AG) + 1)+10000
    AP = AP.merge(AG[['id_ag', 'id_ag_num']], how = 'left',
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
    AP['id_ap_num'] = np.arange(1, len(AP) + 1)+100000
    AG = AP.groupby(
                 ['id_ag'], as_index = False).agg(
                 {'stop_name':'first',
                  'stop_lat' : 'mean',
                  'stop_lon' : 'mean'}, as_index = False).reset_index(drop=True)
    AG['id_ag_num'] = np.arange(1, len(AG) + 1)+100000
    AP = AP.merge(AG[['id_ag','id_ag_num']])
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

def ag_ap_generate_reshape_sncf(raw_stops):

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
    AP.loc[:,'id_ag_num'] =AP.loc[:,'id_ag'].str.replace('StopArea:OCE','').astype(np.int64)
    AG.loc[:,'id_ag_num'] =AG.loc[:,'id_ag'].str.replace('StopArea:OCE','').astype(np.int64)
    AP.dropna(axis = 'columns', how = 'all', inplace = True)
    AG.dropna(axis = 'columns', how = 'all', inplace = True)
    return AP,AG,marker

def ligne_generate(raw_routes):
    route_type = pd.DataFrame({'route_type' : pd.Series([0,1,2,3,4,5,6,7,11,12]),
                               'mode' : pd.Series(["tramway", "metro", "train", "bus","ferry", "tramway par cable",
                                                   "téléphérique", "funiculaire", "trolleybus", "monorail"])})
    lignes = raw_routes.merge(route_type, on = 'route_type', how = 'left')
    lignes.dropna(axis = 'columns', how = 'all', inplace = True)
    return lignes

def itineraire_generate(raw_stoptimes, AP, raw_trips):
    stop_times = raw_stoptimes.rename({'stop_id':'id_ap'},axis = 1).dropna(axis = 'columns', how = 'all')
    stop_times['TH'] = np.vectorize(str_time_hms_hour)(stop_times.arrival_time)
    stop_times['TH'] = np.where(stop_times['TH']>=24, stop_times['TH']-24,stop_times['TH'])
    stop_times['arrival_time'] = np.vectorize(str_time_hms)(stop_times.arrival_time)
    stop_times['departure_time'] = np.vectorize(str_time_hms)(stop_times.departure_time)
    itnry_1 = stop_times.merge(AP[['id_ap','id_ap_num', 'id_ag_num']], how = 'left', on = 'id_ap')
    itnry_2 = itnry_1.merge(raw_trips[['id_course_num', 'id_ligne_num', 'id_service_num','direction_id']], how = 'left', on ='id_course_num')
    itineraire = itnry_2[['id_course_num','id_ligne_num','id_service_num','direction_id','stop_sequence','id_ap_num','id_ag_num','arrival_time','departure_time','TH']]
    return itineraire

def itiarc_generate(itineraire,AG):
    iti_arc_R = itineraire.copy().drop(['departure_time','id_service_num','id_ligne_num'], axis = 1)
    iti_arc_L = itineraire.copy().drop(['arrival_time'], axis = 1)
    iti_arc_L['ordre_b'] = iti_arc_R['stop_sequence'] +1
    iti_arc = iti_arc_L.merge(iti_arc_R, how ='left', left_on = ['id_course_num','ordre_b','direction_id'], right_on = ['id_course_num', 'stop_sequence','direction_id'], suffixes = ('_a','_b')).dropna().reset_index(drop = True)
    iti_arc_dist = iti_arc.merge(AG[['id_ag_num','stop_lon','stop_lat']], left_on = 'id_ag_num_a',right_on = 'id_ag_num',how = 'left').merge(AG[['id_ag_num','stop_lon','stop_lat']], left_on = 'id_ag_num_b',right_on = 'id_ag_num',how = 'left')
    iti_arc_dist['DISTANCE'] = np.vectorize(getDistHaversine)(iti_arc_dist.stop_lat_x,iti_arc_dist.stop_lon_x,iti_arc_dist.stop_lat_y,iti_arc_dist.stop_lon_y )
    cols = ['id_course_num','id_ligne_num','id_service_num','direction_id','stop_sequence_a','departure_time','id_ap_num_a','id_ag_num_a','TH_a','stop_sequence_b','arrival_time','id_ap_num_b', 'id_ag_num_b','TH_b','DISTANCE']
    iti_arc_f = iti_arc_dist[cols].rename({'stop_sequence_a':'ordre_a','arrival_time':'heure_arrive',
                                      'stop_sequence_b':'ordre_b','departure_time':'heure_depart'},axis =1)
    return iti_arc_f

def course_generate(itineraire,itineraire_arc):
    course = itineraire.groupby(
    ['id_ligne_num', 'id_service_num','id_course_num','direction_id'], as_index=False).agg(
    {'arrival_time': 'min',
    'departure_time': 'max',
    'id_ap_num': ['first', 'last'],
    'id_ag_num': ['first', 'last'],
    'stop_sequence': 'max'})
    course.columns = course.columns.map(''.join)
    dist = itineraire_arc.groupby('id_course_num',as_index = False)['DISTANCE'].sum()
    course = course.merge(dist, on = 'id_course_num', how = 'left')
    course['sous_ligne'] = course['id_ligne_num'].astype('str') + '_' + course['direction_id'].astype('str')+ '_' + course['id_ag_numfirst'].astype('str') + '_' + course['id_ag_numlast'].astype('str') + '_' + course['stop_sequencemax'].astype('str')
    course.rename({'arrival_timemin':'heure_depart', 'departure_timemax':'heure_arrive',
                'id_ap_numfirst':'id_ap_num_debut', 'id_ap_numlast':'id_ap_num_terminus',
                'id_ag_numfirst':'id_ag_num_debut', 'id_ag_numlast':'id_ag_num_terminus',
                'stop_sequencemax':'nb_arrets'}, axis = 1, inplace = True)
    return course

def sl_generate(course):
    sous_ligne = course.groupby(
    ['sous_ligne', 'id_ligne_num','id_ag_num_debut', 'id_ag_num_terminus', 'direction_id','nb_arrets'], as_index=False).id_course_num.count()
    sous_ligne.drop('id_course_num',axis = 1, inplace = True)
    return sous_ligne

def service_date_generate(calendar,calendar_dates,validite,Dates):
    if calendar is None:
        service_date_redu = calendar_dates.merge(Dates, left_on = 'date', right_on = 'Date_GTFS')
    else :
        calendar['validite'] = calendar.monday + calendar.tuesday*2 + calendar.wednesday*4 + calendar.thursday*8 + calendar.friday*16 + calendar.saturday*32 + calendar.sunday*64
        calendar =calendar[['id_service_num', 'validite', 'start_date', 'end_date']].merge(
        validite, how = 'left', left_on='validite', right_on='valid')
        calendar.drop('valid', axis=1, inplace=True)
        list_dates = Dates[(Dates.Date_GTFS >= min(calendar.start_date)) & (Dates.Date_GTFS <= max(calendar.start_date))]
        calendar['key'] = 0
        list_dates['key'] = 0
        service_date_CJ = pd.merge(calendar, list_dates, on = 'key')
        service_date_redu1 = service_date_CJ[
        (service_date_CJ.Date_GTFS >= service_date_CJ.start_date) &
        (service_date_CJ.Date_GTFS <= service_date_CJ.end_date)].reset_index(drop = True)
        service_date_redu1['triage'] = [service_date_redu1['valid_01'][a][service_date_redu1['Type_Jour'][a]-1] for a in range(len(service_date_redu1['valid_01']))]
        service_date_redu2 = service_date_redu1[service_date_redu1['triage'] == "1"]
        service_date_redu2.drop(['start_date', 'end_date','valid_01','key','validite','triage'], axis = 1, inplace = True)
        cal_dates = calendar_dates.merge(list_dates, left_on = 'date', right_on = 'Date_GTFS')
        service_date_redu3 = service_date_redu2.append(cal_dates[cal_dates['exception_type']==1]).drop(
            ['date', 'exception_type', 'key'], axis = 1).reset_index(drop = True)
        cal_date_suppr = cal_dates[cal_dates['exception_type']==2].reset_index(drop = True)
        service_date_redu4 = service_date_redu3.merge(calendar_dates[calendar_dates['exception_type']==2],
                                                      how = 'left', left_on=['id_service_num', 'Date_GTFS'] ,
                                                      right_on=['id_service_num', 'date'])
        service_date_redu = service_date_redu4[service_date_redu4['exception_type'] != 2].drop(
            ['date', 'exception_type'], axis = 1)
    mindate = min(service_date_redu['Date_GTFS'])
    maxdate = max(service_date_redu['Date_GTFS'])
    msg_date = f"Le présent jeu de donnée a une validité de {mindate} à {maxdate}"
    return service_date_redu, msg_date

def service_jour_type_generate(service_date,course, type_vacances):
    crs_simplifie = course[['id_course_num', 'id_ligne_num', 'id_service_num']]
    crs_et_plage_date = pd.merge(crs_simplifie, service_date, how = 'left', on = 'id_service_num')
    nb_crs_parligne_parjtype = crs_et_plage_date.groupby(
    ['Date_GTFS','id_ligne_num',type_vacances], as_index = False)['id_course_num'].count().rename(
    {'id_course_num':'count'}, axis=1)
    nb_jr_per_nb_crs = nb_crs_parligne_parjtype.groupby(
    ['id_ligne_num', type_vacances, 'count'], as_index=False)['Date_GTFS'].count().rename(
    {'Date_GTFS':'count_days'}, axis=1)
    max_nb_jr_zb = nb_jr_per_nb_crs.groupby(
    ['id_ligne_num', type_vacances], as_index=False)['count_days'].max().rename(
    {'count_days':'max_days'}, axis=1)
    ncours_jtype = nb_jr_per_nb_crs.merge(max_nb_jr_zb, how = 'left', on = ['id_ligne_num', type_vacances])
    choix_jtype_1 = nb_crs_parligne_parjtype.merge(
    ncours_jtype, how = 'left', on = ['id_ligne_num',type_vacances,'count'])
    choix_jtype = choix_jtype_1[
    choix_jtype_1['count_days']==choix_jtype_1['max_days']].groupby(
    ['id_ligne_num',type_vacances],as_index=False ).agg(
    {'Date_GTFS':'first','count':'max'},as_index=False )
    service_jtype_1 = crs_et_plage_date.merge(
    choix_jtype, how = 'left', on=['id_ligne_num',type_vacances ,'Date_GTFS']).dropna().reset_index(drop = True)
    service_jtype = service_jtype_1.groupby(
    ['id_ligne_num', 'id_service_num', type_vacances], as_index = False).agg({'Date_GTFS':'first'}, as_index = False)
    return service_jtype

def nb_passage_ag(service_jour_type_export, itineraire_export,AG,type_vac):
    iti_typejour = itineraire_export.merge(service_jour_type_export, on =['id_ligne','service_id'])
    nb_passage = iti_typejour.groupby(
        ['id_ag_num', type_vac], as_index = False).id_course.count().sort_values(['id_ag_num'])
    nb_passage_ag = nb_passage.merge(AG)
    nb_passage_ag = nb_passage_ag.reset_index(drop=True).rename({'id_course':'nb_passage'},axis = 1)
    nb_passage_ag_pv = pd.pivot_table(nb_passage_ag,values = 'nb_passage', index = ['id_ag_num','stop_name', 'stop_lat','stop_lon' ], columns = type_vac, fill_value = 0,  aggfunc=np.sum).reset_index()
    return nb_passage_ag_pv

def nb_course_ligne(service_jour_type_export, courses_export,type_vac):
    courses_jtype =  courses_export.merge(service_jour_type_export, on =['id_ligne','service_id'])
    nb_courses_par_ligne = courses_jtype.groupby(
        ['id_ligne',type_vac], as_index = False)['id_course'].count().sort_values(
        ['id_ligne']).rename({'id_course':'nb_courses'},axis = 1)
    nb_courses_par_ligne_pv = pd.pivot_table(nb_courses_par_ligne,values = 'nb_courses', index = ['id_ligne'], columns = type_vac, fill_value = 0,  aggfunc=np.sum).reset_index()
    return nb_courses_par_ligne_pv

def nb_course_sl(service_jour_type_export, courses_export,type_vac):
    courses_jtype =  courses_export.merge(service_jour_type_export, on =['id_ligne','service_id'])
    nb_courses_par_sl = courses_jtype.groupby(
        ['sous_ligne',type_vac], as_index = False)['id_course'].count().sort_values(
        ['sous_ligne']).rename({'id_course':'nb_courses'},axis = 1)
    nb_courses_par_sl_pv = pd.pivot_table(nb_courses_par_sl,values = 'nb_courses', index = ['sous_ligne'], columns = type_vac, fill_value = 0,  aggfunc=np.sum).reset_index()
    return nb_courses_par_sl_pv

def calcul_headway(service_jour_type_export,courses_export,debut_HPM , fin_HPM, debut_HPS ,fin_HPS, type_vac):

    courses_jtype =  courses_export.merge(service_jour_type_export, on =['id_ligne','service_id'])

    mask_FM = courses_jtype['heure_depart'] < debut_HPM
    mask_HPM = (courses_jtype['heure_depart'] >= debut_HPM) & (courses_jtype['heure_depart'] < fin_HPM)
    mask_HC = (courses_jtype['heure_depart'] >= fin_HPM) & (courses_jtype['heure_depart'] < debut_HPS)
    mask_HPS = (courses_jtype['heure_depart'] >= debut_HPS) & (courses_jtype['heure_depart'] < fin_HPS)
    mask_FS = (courses_jtype['heure_depart'] >= fin_HPS)

    courses_jtype.loc[mask_FM,'periode'] = 'FM'
    courses_jtype.loc[mask_HPM,'periode'] = 'HPM'
    courses_jtype.loc[mask_HC,'periode'] = 'HC'
    courses_jtype.loc[mask_HPS,'periode'] = 'HPS'
    courses_jtype.loc[mask_FS,'periode'] = 'FS'

    headway = courses_jtype.groupby([type_vac, 'sous_ligne','periode'],as_index = False)['id_course'].count().rename({'id_course':'nb_courses'},axis = 1)
    headway_pv = pd.pivot_table(headway,values = 'nb_courses', index = [type_vac,'sous_ligne'], columns = 'periode', fill_value = 0,  aggfunc=np.sum).reset_index()
    duration_FM = (debut_HPM - min(courses_jtype['heure_depart']))*24*60
    duration_HPM = (fin_HPM - debut_HPM)*24*60
    duration_HC = (debut_HPS - fin_HPM)*24*60
    duration_HPS = (fin_HPS - debut_HPS)*24*60
    duration_FS = (max(courses_jtype['heure_depart']) - fin_HPS)*24*60
    headway_pv.loc[:,'HEADWAY_FM'] = duration_FM/headway_pv.loc[:,'FM']
    headway_pv.loc[:,'HEADWAY_HPM'] = duration_HPM/headway_pv.loc[:,'HPM']
    headway_pv.loc[:,'HEADWAY_HC'] = duration_HC/headway_pv.loc[:,'HC']
    headway_pv.loc[:,'HEADWAY_HPS'] = duration_HPS/headway_pv.loc[:,'HPS']
    headway_pv.loc[:,'HEADWAY_FS'] = duration_FS/headway_pv.loc[:,'FS']
    headway_pv = headway_pv.replace(np.inf, np.nan).drop(['FM','HPM','HC','HPS','FS'],axis = 1)
    return headway_pv

def nan_in_col_workaround(pd_serie):
    a = pd_serie.astype('float64')
    b = a.fillna(-1)
    c = b.astype(np.int64)
    d = c.astype(str)
    e = d.replace('-1', np.nan)
    return e

def MEF_ligne(lignes):
    lignes_export = lignes.rename({'route_id':'id_ligne'},axis = 1)
    return lignes_export

def MEF_course(courses,route_id_coor, ser_id_coor,trip_id_coor):
    crs_cols = ['trip_id','id_course_num','route_id', 'sous_ligne' ,'service_id','heure_depart','heure_arrive', 'id_ap_num_debut', 'id_ap_num_terminus','id_ag_num_debut', 'id_ag_num_terminus', 'nb_arrets','DISTANCE']
    courses_export = courses.merge(
    route_id_coor).merge(
    ser_id_coor).merge(
    trip_id_coor).drop(
    ['id_ligne_num', 'id_service_num' ], axis = 1)[crs_cols].rename({'route_id':'id_ligne','trip_id':'id_course' },axis = 1)
    return courses_export

def MEF_iti(itineraire,route_id_coor, ser_id_coor,trip_id_coor):
    iti_cols = ['trip_id','route_id', 'service_id',  'id_course_num','ordre',
            'id_ap_num', 'id_ag_num', 'heure_depart', 'heure_arrivee', 'TH']
    itineraire_export = itineraire.merge(
    route_id_coor).merge(
    ser_id_coor).merge(
    trip_id_coor).rename(
    {'stop_sequence':'ordre',
     'arrival_time':'heure_depart',
     'departure_time':'heure_arrivee' },axis = 1).drop(
    ['id_ligne_num', 'id_service_num'],axis =1)[iti_cols].rename({'route_id':'id_ligne','trip_id':'id_course'},axis = 1)
    return itineraire_export

def MEF_iti_arc(itineraire_arc,route_id_coor, ser_id_coor,trip_id_coor):
    itiarc_cols = ['trip_id', 'id_course_num','route_id', 'service_id', 'ordre_a','heure_depart', 'id_ap_num_a', 'id_ag_num_a', 'TH_a', 'ordre_b',
               'heure_arrive', 'id_ap_num_b', 'id_ag_num_b', 'TH_b','DISTANCE']
    iti_arc_export = itineraire_arc.merge(
        route_id_coor).merge(
        ser_id_coor).merge(
        trip_id_coor).drop(
        ['id_ligne_num', 'id_service_num'],axis =1)[itiarc_cols].rename({'route_id':'id_ligne','trip_id':'id_course'},axis = 1)
    return iti_arc_export

def MEF_SL(sous_ligne,route_id_coor):
    sl_cols = ['sous_ligne','route_id','direction_id', 'id_ag_num_debut', 'id_ag_num_terminus','nb_arrets']
    sl_export = sous_ligne.merge(
        route_id_coor).drop(
        ['id_ligne_num'],axis =1)[sl_cols].rename({'route_id':'id_ligne'},axis = 1)
    return sl_export

def MEF_serdate(service_dates, ser_id_coor):
    ser_cols = ['service_id','Date_GTFS', 'Type_Jour', 'Mois', 'Annee','Type_Jour_Vacances_A', 'Type_Jour_Vacances_B', 'Type_Jour_Vacances_C']
    service_dates_export = service_dates.merge(
        ser_id_coor).drop(
        ['id_service_num'],axis =1)[ser_cols]
    return service_dates_export

def MEF_servjour(service_jour_type,route_id_coor,ser_id_coor,type_vac):
    servjour_cols = ['route_id', 'service_id', 'Date_GTFS', type_vac]
    service_jour_type_export = service_jour_type.merge(
        route_id_coor).merge(
        ser_id_coor).drop(
        ['id_ligne_num', 'id_service_num'],axis =1)[servjour_cols].rename({'route_id':'id_ligne'},axis = 1)
    return service_jour_type_export

def MEF_course_sncf(courses,route_id_coor, ser_id_coor,trip_id_coor):
    crs_cols = ['trip_id','id_course_num','route_id', 'sous_ligne' ,'service_id','heure_depart','heure_arrive', 'id_ap_num_debut', 'id_ap_num_terminus','id_ag_num_debut', 'id_ag_num_terminus', 'nb_arrets','id_service_num','DISTANCE']
    courses_export = courses.merge(
    route_id_coor).merge(
    ser_id_coor).merge(
    trip_id_coor).drop(
    ['id_ligne_num'], axis = 1)[crs_cols].rename({'route_id':'id_ligne','trip_id':'id_course' },axis = 1)
    courses_export.loc[:,'N_train'] = courses_export.loc[:,'id_course'].str.strip('OCE').str.strip('SN').str.strip('ZW').str[:6].astype(np.int32)
    courses_export.loc[:,'Code_Course'] = courses_export.loc[:,'id_course'].str[11:]
    return courses_export

def MEF_iti_sncf(itineraire,courses_export,route_id_coor, ser_id_coor,trip_id_coor):
    iti_cols = [ 'trip_id', 'id_course_num','route_id', 'service_id','ordre',
            'id_ap_num', 'id_ag_num', 'heure_depart', 'heure_arrivee', 'TH']
    itineraire_export = itineraire.merge(
    route_id_coor).merge(
    ser_id_coor).merge(
    trip_id_coor).rename(
    {'stop_sequence':'ordre',
     'arrival_time':'heure_depart',
     'departure_time':'heure_arrivee' },axis = 1).drop(
    ['id_ligne_num', 'id_service_num'],axis =1)[iti_cols].rename({'route_id':'id_ligne','trip_id':'id_course'},axis = 1)
    itineraire_export = itineraire_export.merge(courses_export[['id_course','N_train']])
    return itineraire_export

def MEF_iti_arc_sncf(itineraire_arc,courses_export,route_id_coor, ser_id_coor,trip_id_coor):
    itiarc_cols = [ 'trip_id', 'id_course_num','route_id', 'service_id','ordre_a','heure_depart', 'id_ap_num_a', 'id_ag_num_a', 'TH_a', 'ordre_b',
               'heure_arrive', 'id_ap_num_b', 'id_ag_num_b', 'TH_b','DISTANCE']
    iti_arc_export = itineraire_arc.merge(
        route_id_coor).merge(
        ser_id_coor).merge(
        trip_id_coor).drop(
        ['id_ligne_num', 'id_service_num'],axis =1)[itiarc_cols].rename({'route_id':'id_ligne','trip_id':'id_course'},axis = 1)
    iti_arc_export = iti_arc_export.merge(courses_export[['id_course','N_train']])
    return iti_arc_export

def GOAL_train(AG,courses,calendar,validite, lignes):
    ag_name = AG[['id_ag_num','stop_name']]
    ag_name.loc[:,'nom_gare'] = AG.loc[:,'stop_name'].str.replace('GARE DE ','')
    ag_simp = ag_name.loc[:,['id_ag_num','nom_gare']]
    cols_train = ['id_ligne','id_course','N_train','Code_Course','id_service_num','id_ag_num_debut','id_ag_num_terminus']
    crs_train = courses[cols_train].merge(
        ag_simp, left_on = 'id_ag_num_debut', right_on = 'id_ag_num').merge(
        ag_simp, left_on = 'id_ag_num_terminus', right_on = 'id_ag_num').drop(['id_ag_num_x','id_ag_num_y'],axis = 1)
    crs_train.loc[:,'DESCRIPTION'] = crs_train.loc[:,'nom_gare_x'] + ' < > ' + crs_train.loc[:,'nom_gare_y']
    crs_train_jr = crs_train.merge(calendar, how = 'left')[['id_ligne','id_course','N_train','Code_Course','DESCRIPTION','validite']]
    goal_train_1 = crs_train_jr.merge(validite, left_on = 'validite',right_on = 'valid').drop(
        ['validite','valid'],axis =1).rename(
        {'valid_01':'JOURS_CIRCULATION'},axis = 1)
    goal_train_1.loc[goal_train_1['JOURS_CIRCULATION'].isnull()] = '0000000'
    goal_train = goal_train_1.merge(lignes[['id_ligne','mode']])
    return goal_train


def base_ferro_tbls(BaseFerro_PATH='C:/Users/wei.si/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/gtfs_miner/Resources/Base_Ferroviaire_User.accdb'):
    access_driver = pyodbc.dataSources()['MS Access Database']
    path_BF = (BaseFerro_PATH)
    with pyodbc.connect(driver = access_driver, dbq = path_BF) as conn_BF:
        arc_elem_corr = pd.read_sql_query('SELECT * FROM Corr_Arcs_Elémentaires_Liaisons',conn_BF)
        arc_elem = pd.read_sql_query('SELECT * FROM Arcs_Elémentaires',conn_BF)
        gares = pd.read_sql_query('SELECT * FROM Codes_Gares',conn_BF)
    return arc_elem_corr,arc_elem,gares

def export_access(table_name, var_name,str_create_table, str_insert, access_path='./GTFS.accdb' ):
    access_driver = pyodbc.dataSources()['MS Access Database']
    con = pyodbc.connect(driver = access_driver, dbq = access_path)
    cur = con.cursor()
    try:
        cur.execute(f"DROP TABLE [{table_name}]")
        cur.execute(str_create_table)
        cur.executemany(str_insert,var_name.itertuples(index=False))
    except pyodbc.ProgrammingError:
        cur.execute(str_create_table)
        cur.executemany(str_insert,var_name.itertuples(index=False))
    con.commit()
    cur.close()
    con.close()
    del con,cur

def GOAL_trainmarche(iti_arc_export,goal_train):
    cols_trainmarche = ['id_ligne','id_course','N_train','Code_Course','id_ag_num_a','id_ag_num_b','ordre_b','heure_depart','heure_arrive']
    train_marche = iti_arc_export.merge(goal_train[['id_course','Code_Course']])[cols_trainmarche]
    train_marche.ordre_b = train_marche.ordre_b.astype(np.int8)
    train_marche.id_ag_num_b = train_marche.id_ag_num_b.astype(np.int32)
    iti_arc_goal['HEURE_SORTIE'] = np.vectorize(heure_goal)(iti_arc_goal['h_dep_2'])
    iti_arc_goal['HEURE_ARRIVEE'] = np.vectorize(heure_goal)(iti_arc_goal['h_arr_2'])
    Goal_trainmarche = train_marche.drop(['heure_depart','heure_arrive'],axis = 1).rename(
        {'id_ag_num_a':'GARE_A','id_ag_num_b':'GARE_B','ordre_b':'SEQUENCE'},axis = 1)
    return Goal_trainmarche

def visum_fer_tbls(path):
    access_driver = pyodbc.dataSources()['MS Access Database']
    path_BF = (path)
    with pyodbc.connect(driver = access_driver, dbq = path_BF) as conn_BF:
        NODE = pd.read_sql_query('SELECT * FROM NODE',conn_BF)
        LINK = pd.read_sql_query('SELECT * FROM LINK',conn_BF)
    return NODE,LINK

def iti_elem_lookup(to_do,table_arc_elem):
    debut_liaison = to_do[0]
    fin_de_liaison = to_do[1]
    # Prepare empty table
    iti_potentiel = pd.DataFrame(columns=['ID_ARC','UIC_A','UIC_B','LENGTH','order'])
    iti_potentiel['order'] = 0
    # First step : look for first iti_elem
    etape1 = table_arc_elem.loc[table_arc_elem['UIC_A']==debut_liaison]
    etape1['order'] = 1
    iti_potentiel = iti_potentiel.append(etape1)
    if (debut_liaison not in table_arc_elem['UIC_A'].values) or (fin_de_liaison not in table_arc_elem['UIC_B'].values) :
        pathnotfound = pd.DataFrame({'UIC_A_Liaison' : [debut_liaison],'UIC_B_Liaison' : [fin_de_liaison]})
        iti_arc_elem_final = pd.DataFrame(columns=['UIC_A_Liaison','UIC_B_Liaison','ID_ARC','UIC_A_Arc','UIC_B_Arc','LENGTH','order'])
    else :
        # Loop until find the end of UIC_B_LIAISON
        max_iter = 500
        for i in range(1,max_iter):
            if fin_de_liaison in iti_potentiel.loc[:,'UIC_B'].values:
                break
            etape = table_arc_elem.loc[table_arc_elem['UIC_A'].isin(iti_potentiel.loc[iti_potentiel['order']==i,'UIC_B'].values) &
                                 ~table_arc_elem['UIC_B'].isin(iti_potentiel.loc[iti_potentiel['order']==i,'UIC_A'].values)]
            etape['order'] = 1+i
            iti_potentiel = iti_potentiel.append(etape)
        # Remove unused itinerary
        iti_keep = iti_potentiel.loc[iti_potentiel['ID_ARC']==0]
        select_fin = iti_potentiel.loc[iti_potentiel['UIC_B']==fin_de_liaison]
        iti_keep = iti_keep.append(select_fin)
        max_order = max(iti_potentiel['order'])
        for i in reversed(range(1,max_order+1)):
            gare_A_to_look = iti_keep.loc[iti_keep['order']==i,'UIC_A'].values
            select = iti_potentiel.loc[iti_potentiel['UIC_B'].isin(gare_A_to_look)& (iti_potentiel['order']==i-1)]
            iti_keep = iti_keep.append(select)
        iti_arc_elem_final = iti_keep.sort_values(by='order', ascending=True)
        iti_arc_elem_final['UIC_A_Liaison'] = to_do[0]
        iti_arc_elem_final['UIC_B_Liaison'] = to_do[1]
        iti_arc_elem_final.rename({'UIC_A':'UIC_A_Arc','UIC_B':'UIC_B_Arc'},axis = 1,inplace = True)
        iti_arc_elem_final = iti_arc_elem_final[['UIC_A_Liaison','UIC_B_Liaison','ID_ARC','UIC_A_Arc','UIC_B_Arc','LENGTH','order']]
        pathnotfound = pd.DataFrame(columns=['UIC_A_Liaison' ,'UIC_B_Liaison'])
    return(iti_arc_elem_final,pathnotfound)

def duree_arc(df):
    duree = max(df['heure_arrive']) - min(df['heure_depart'])
    return duree

def heure_goal(horaire_excel):
    horaire = f'H{int(math.modf(horaire_excel*24)[1]):02}{int(math.modf(horaire_excel*24)[0]*60):02}'
    return horaire

def GOAL_trainmarche(iti_arc_export,goal_train):
    cols_trainmarche = ['id_course','id_ligne','N_train','Code_Course','UIC_A_Arc','UIC_B_Arc','order','h_dep_2','h_arr_2']
    train_marche = iti_arc_export.merge(goal_train[['id_course','Code_Course']])[cols_trainmarche]
    train_marche.order = train_marche.order.astype(np.int8)+1
    train_marche['HEURE_SORTIE'] = np.vectorize(heure_goal)(train_marche['h_dep_2'])
    train_marche['HEURE_ARRIVEE'] = np.vectorize(heure_goal)(train_marche['h_arr_2'])
    Goal_trainmarche = train_marche.drop(['h_dep_2','h_arr_2'],axis = 1).rename({'order':'SEQUENCE'},axis = 1)
    return Goal_trainmarche

def arc_elementaire_create(itineraire_export,iti_arc_export,lignes_export,AG,pth_reseau):
    node,link = visum_fer_tbls(pth_reseau)
    node = node[['NO','NAME']].rename({'NO':'UIC','NAME':'NOM_GARE'},axis = 1)
    link['ID_ARC']= link.index
    link = link[['ID_ARC','FROMNODENO','TONODENO','LENGTH']].rename({'NO':'UIC','FROMNODENO':'UIC_A','TONODENO':'UIC_B'},axis = 1)
    iti_mode = itineraire_export.merge(lignes_export[['id_ligne','mode']])
    iti_lourd = iti_mode.loc[iti_mode['mode'].isin(['train','tramway'])]
    toute_gares = iti_lourd.groupby(['id_ag_num'],as_index = False)['id_course'].count()
    toute_gares['Dans_Base'] = toute_gares['id_ag_num'].isin(node['UIC'])
    gare_non_exist = toute_gares.loc[~toute_gares['Dans_Base'],'id_ag_num']
    list_gare_a_ajouter = AG.loc[AG['id_ag_num'].isin(gare_non_exist)]
    itiarc_mode = iti_arc_export.merge(lignes_export[['id_ligne','mode']])
    itiarc_lourd = itiarc_mode.loc[itiarc_mode['mode'].isin(['train','tramway'])]
    liaison_GTFS = itiarc_lourd.groupby(['id_ag_num_a','id_ag_num_b'],as_index = False)['id_course'].count()[['id_ag_num_a','id_ag_num_b']].rename({'id_ag_num_a':'UIC_A_Liaison','id_ag_num_b':'UIC_B_Liaison'},axis = 1)
    liaison_GTFS_f =  liaison_GTFS.loc[~(liaison_GTFS['UIC_A_Liaison'].isin(gare_non_exist)|liaison_GTFS['UIC_B_Liaison'].isin(gare_non_exist))]
    list_arc_non_exist = pd.DataFrame(columns=['UIC_A_Arc','UIC_B_Arc'])
    iti_elem_to_add = pd.DataFrame(columns=['UIC_A_Liaison','UIC_B_Liaison','ID_ARC','UIC_A_Arc','UIC_B_Arc','LENGTH','order'])

    for i, row in liaison_GTFS_f.iterrows():
        liais_arc, empty_arc = iti_elem_lookup(row,link)
        list_arc_non_exist = list_arc_non_exist.append(empty_arc)
        iti_elem_to_add = iti_elem_to_add.append(liais_arc)
    # jointure avec itineraire arc
    iti_arc_elem = itiarc_lourd.merge(iti_elem_to_add, left_on=['id_ag_num_a','id_ag_num_b'], right_on=['UIC_A_Liaison','UIC_B_Liaison'],how = 'left')
    iti_arc_elem.loc[iti_arc_elem['UIC_A_Arc'].isna(),'NON_EXIST'] =1
    list_arc_non_trouve = iti_arc_elem.loc[iti_arc_elem['UIC_A_Arc'].isna(),['id_ap_num_a','id_ag_num_a','NON_EXIST']].groupby(['id_ap_num_a','id_ag_num_a'],as_index = False)['NON_EXIST'].count()
    iti_arc_elem.loc[iti_arc_elem['NON_EXIST']==1,'UIC_A_Arc' ] = iti_arc_elem.loc[iti_arc_elem['NON_EXIST']==1,'id_ag_num_a' ]
    iti_arc_elem.loc[iti_arc_elem['NON_EXIST']==1,'UIC_B_Arc' ] = iti_arc_elem.loc[iti_arc_elem['NON_EXIST']==1,'id_ag_num_b' ]
    iti_arc_elem.reset_index(drop = True, inplace = True)
    iti_arc_elem['order'] = iti_arc_elem.groupby('id_course').cumcount()
    iti_arc_elem['sum_length'] = iti_arc_elem.groupby(['id_course', 'ordre_a'])['LENGTH'].transform(np.sum)
    iti_arc_elem['ratio'] = iti_arc_elem['LENGTH']/iti_arc_elem['sum_length']
    iti_arc_elem['min_dep'] = iti_arc_elem.groupby(['id_course', 'ordre_a'])['heure_depart'].transform(min)
    iti_arc_elem['max_arr'] = iti_arc_elem.groupby(['id_course', 'ordre_a'])['heure_arrive'].transform(max)
    iti_arc_elem['duree'] = iti_arc_elem['max_arr'] - iti_arc_elem['min_dep']
    iti_arc_elem['tps_arc'] = iti_arc_elem['ratio'] * iti_arc_elem['duree']
    iti_arc_elem['tps_arc_csum'] = iti_arc_elem.groupby(['id_course', 'ordre_a'])['tps_arc'].transform(np.cumsum)
    iti_arc_elem['h_arr_2'] =  iti_arc_elem['min_dep'] + iti_arc_elem['tps_arc_csum']
    iti_arc_elem['h_arr_2'] = np.where(iti_arc_elem['h_arr_2'].isna(), iti_arc_elem['heure_arrive'], iti_arc_elem['h_arr_2'])
    iti_arc_elem['h_dep_2'] = iti_arc_elem.groupby(['id_course', 'ordre_a'])['h_arr_2'].shift(1)
    iti_arc_elem['h_dep_2'] = np.where(iti_arc_elem['h_dep_2'].isna(), iti_arc_elem['min_dep'], iti_arc_elem['h_dep_2'])
    iti_arc_elem['h_arr_2'] = np.where(iti_arc_elem['h_arr_2'].isna(), iti_arc_elem['heure_arrive'], iti_arc_elem['h_arr_2'])
    iti_arc_elem_f = iti_arc_elem[['id_course', 'id_course_num', 'id_ligne',    'service_id', 'ordre_a',  'heure_depart',
       'id_ap_num_a',   'id_ag_num_a', 'TH_a',       'ordre_b',  'heure_arrive',   'id_ap_num_b', 'id_ag_num_b',
       'TH_b',       'N_train',          'mode', 'UIC_A_Arc', 'UIC_B_Arc',        'LENGTH',         'order',
        'NON_EXIST',  'h_dep_2', 'h_arr_2']]
    return iti_arc_elem_f,list_arc_non_exist,list_gare_a_ajouter
