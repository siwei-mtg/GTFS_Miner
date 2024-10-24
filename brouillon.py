import os
from datetime import datetime
import time
import logging
import json
import pandas as pd
import math
import logging
import chardet
import numpy as np
from scipy.spatial.distance import pdist
import logging
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.vq import kmeans2
from scipy import cluster
import geopandas as gpd
from shapely.geometry import LineString

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

def heure_from_xsltime(horaire_excel):
    horaire = f'{int(math.modf(horaire_excel*24)[1]):02}:{int(math.modf(horaire_excel*24)[0]*60):02}'
    return horaire

def encoding_guess(acces):
    '''
    detect and return file encoding
    '''
    with open(acces, 'rb') as rawdata:
        encod = chardet.detect(rawdata.read(10000))
    return encod

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

def nan_in_col_workaround(pd_serie):
    a = pd_serie.astype('float64')
    b = a.fillna(-1)
    c = b.astype(np.int64)
    d = c.astype(str)
    e = d.replace('-1', np.nan)
    return e

class FileDataTypeError(Exception):
    pass

# Main function
def read_raw_GTFSdata(dirpath : str, plugin_path : str):
    '''
    Read raw GTFS data from the given directory and gather metadata.
    '''
    dates = read_datetable(plugin_path)
    routeTypes = read_routeTypes(plugin_path)
    
    jsonFile = initialize_metadata(dirpath)
    filenames, filepaths = get_filenames(dirpath)
    
    rawgtfs = {}
    for i in range(len(filepaths)):
        df = pd.read_csv(filepaths[i], encoding='utf-8')
        rawgtfs[filenames[i]] = df
        
        table_info = generate_table_info(df, filenames[i])
        process_special_columns(df, jsonFile)
        
        table_info['columns'] = []
        for column in df.columns:
            column_info = process_column_info(df, column)
            table_info['columns'].append(column_info)
        
        jsonFile["tables"].append(table_info)
    
    return rawgtfs, jsonFile, dates, routeTypes

# --- Modular Functions ---
def read_datetable(plugin_path):
    '''
    Read calendar info for plugin
    '''
    Dates = pd.read_csv(("./Resources/Calendrier.txt"), encoding="utf-8",
                        sep = "\t", parse_dates=['Date_Num'], dtype={'Type_Jour': 'int32'})
    #Dates['Date_GTFS'] = pd.to_datetime(Dates['Date_GTFS'],format="%Y%m%d")
    #encoding='utf-8'
    Dates.drop(['Date_Num','Date_Opendata', 'Ferie', 'Vacances_A', 'Vacances_B', 'Vacances_C',
            'Concat_Select_Type_A', 'Concat_Select_Type_B', 'Concat_Select_Type_C','Type_Jour_IDF','Annee_Scolaire'],axis = 1 , inplace = True)
    logger.info("Date table is imported.")
    return Dates

def read_routeTypes(plugin_path):
    '''
    Read route type info for plugin
    '''
    route_type = pd.read_csv(("./Resources/route_types.csv"),sep = ";", encoding="utf-8")
    #encoding='utf-8'
    logger.info("route_type table is imported.")
    return route_type

def initialize_metadata(dirpath):
    dirname = os.path.basename(dirpath)
    return {
        "dataset": dirname,
        "processing_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "tables": []
    }

def get_filenames(dirpath):
    all_files = os.listdir(dirpath)
    txt_files = [f for f in all_files if f.endswith('.txt')]

    other_files = [f for f in all_files if not f.endswith('.txt')]
    if len(other_files) > 0:
        logger.error("more than 1 non txt file are present in the directory.")
        raise FileDataTypeError("Input data type not correct. Please keep only gtfs related txt file.")
    
    filenames = [os.path.splitext(f)[0] for f in txt_files]
    logger.info(f"File names : {filenames}")
    filepaths = [os.path.join(dirpath, f) for f in txt_files]
    return filenames, filepaths

def generate_table_info(df, table_name):
    return {
        "name": table_name,
        "nb_rows": len(df),
        "nb_cols": len(df.columns),
        "memory_usage": int(round(df.memory_usage(deep=True).sum(), 0)),
    }

def process_special_columns(df, jsonFile):
    if 'start_date' in df.columns:
        try: 
            jsonFile['start_date'] = int(df['start_date'].min())
        except ValueError as e:
            jsonFile['start_date'] = None
            logger.exception(e)
    if 'end_date' in df.columns:
        try:
            jsonFile['end_date'] = int(df['end_date'].max())
        except ValueError as e:
            jsonFile['end_date'] = None
            logger.exception(e)
    if 'date' in df.columns:
        try:
            jsonFile['calendarDate_min'] = int(df['date'].min())
            jsonFile['calendarDate_max'] = int(df['date'].max())
        except ValueError as e:
            jsonFile['calendarDate_min'] = None
            jsonFile['calendarDate_max'] = None
            logger.exception(e)
    if 'route_type' in df.columns:
        jsonFile['route_type'] = ', '.join(df['route_type'].unique().astype(str))

def process_column_info(df, column_name):
    col_data = df[column_name]
    memory_usage = int(round(col_data.memory_usage(deep=True), 0))
    non_null_count = col_data.count()
    non_null_unique_count = col_data.nunique()
    pcent_non_null = non_null_count * 100 / len(df)
    
    # Handle outliers for latitude and longitude columns
    outlier_info = process_outliers(df, column_name)
    
    return {
        "name": column_name,
        "dtype": str(col_data.dtype),
        "column_memory_usage": memory_usage,
        "non_null_count": int(non_null_count),
        "non_null_count_pcent": round(pcent_non_null, 2),
        "non_null_unique_count": non_null_unique_count,
        "nb_outliers": outlier_info['outliers'],
        "pcent_outliers": outlier_info['pcent_outliers']
    }

def process_outliers(df, column_name):
    if column_name in ['stop_lon', 'stop_lat', 'shape_pt_lat', 'shape_pt_lon']:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce')
        mean = df[column_name].mean()
        std = df[column_name].std()
        thh_up = mean + std * 2
        thh_down = mean - std * 2
        outliers = int(((df[column_name] > thh_up) | (df[column_name] < thh_down)).sum())
        pcent_outliers = round(outliers * 100 / len(df), 2)
        return {"outliers": outliers, "pcent_outliers": pcent_outliers}
    return {"outliers": None, "pcent_outliers": None}


import json
import pandas as pd

def datasetLevelInfo(meta):
    dataset = meta['dataset']
    processing_timestamp = meta["processing_timestamp"]
    start_date = meta["start_date"]
    end_date = meta["end_date"]
    route_type = meta["route_type"]
    tbl_count = 0
    tbl_names = []
    memory = 0
    for tbl in meta['tables']:
        tbl_count += 1
        memory += tbl['memory_usage']
        tbl_names.append(tbl.get('name'))
    tbl_names_collapse = ', '.join(tbl_names)

    vars = ["dataset","processing_timestamp","memory_usage","start_date","end_date","nb_tables","table_names","route_type"]
    values = [dataset,processing_timestamp,memory, start_date,end_date,tbl_count,tbl_names_collapse,route_type]
    kpi_table = pd.DataFrame({
        "variable": vars,
        "indicateur":values
    })
    return kpi_table

def tableLevelInfo(meta):
    tbl_info = pd.DataFrame({"table_name":[], "column_name":[],"col_dtype":[],"column_memory_usage":[],
                     "non_null_count": [], "non_null_count_pcent": [],
                    "non_null_unique_count":[], "nb_outliers": [], "pcent_outliers": []})
    for table in meta["tables"]:
        table_name = table['name']
        table_columns = table['columns']
        for col in table_columns:
            col_name = col["name"]
            col_dtype = col["dtype"]
            col_memo = col["column_memory_usage"]
            col_non_null_count = col["non_null_count"]
            col_non_null_count_pcent = col["non_null_count_pcent"]
            col_non_null_unique_count = col["non_null_unique_count"]
            col_nb_outliers =col["nb_outliers"]
            col_pcent_outliers = col["pcent_outliers"]
            # Create a DataFrame for the new row with column names
            new_row_df = pd.DataFrame({
                "table_name": [table_name],
                "column_name": [col_name],
                "col_dtype": [col_dtype],
                "column_memory_usage": [col_memo],
                "non_null_count": [col_non_null_count],
                "non_null_count_pcent": [col_non_null_count_pcent],
                "non_null_unique_count": [col_non_null_unique_count],
                "nb_outliers": [col_nb_outliers],
                "pcent_outliers": [col_pcent_outliers]
            })
            tbl_info = pd.concat([tbl_info, new_row_df], ignore_index=True)
    return tbl_info

def agency_norm(raw_agency):
    agency_v =  pd.DataFrame(columns = ['agency_id', 'agency_name', 'agency_url', 'agency_timezone',
            'agency_lang', 'agency_phone', 'agency_fare_url', 'agency_email'], index = None)
    agency = pd.concat([agency_v, raw_agency], ignore_index=True)  
    agency.dropna(axis = 1, how = 'all', inplace =True)
    return agency

def stops_norm(raw_stops):
    stops_v =  pd.DataFrame(columns = ['stop_id', 'stop_code', 'stop_name', 'stop_desc',
        'stop_lat', 'stop_lon', 'zone_id', 'stop_url','location_type', 'parent_station',
        'stop_timezone','wheelchair_boarding','level_id','platform_code'], index = None)
    stops =  pd.concat([stops_v, raw_stops],  ignore_index=True)
    stops.stop_id = stops.stop_id.astype(str)
    stops.stop_lat = stops.stop_lat.fillna(0)
    stops.stop_lon = stops.stop_lon.fillna(0)
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
    routes =  pd.concat([routes_v, raw_routes], ignore_index=True)
    routes.drop(['route_url', 'route_sort_order',
                 'continuous_pickup','continuous_drop_off'],axis = 1, inplace =True)
    routes.dropna(axis = 1, inplace = True, how = 'all')
    routes.route_id = routes.route_id.astype(str)
    routes.route_type = routes.route_type.astype(np.int32)
    return routes

def trips_norm(raw_trips):
    trips_v =  pd.DataFrame(
        columns = ['route_id', 'service_id', 'trip_id', 'trip_headsign','trip_short_name','direction_id',
                  'block_id','shape_id','wheelchair_accessible' ,'bikes_allowed'], index = None)
    trips =  pd.concat([trips_v, raw_trips],  ignore_index=True)
    trips.drop(['trip_short_name',
                'block_id','wheelchair_accessible' ,'bikes_allowed'],axis = 1, inplace =True)
    tps_cols = ['route_id','service_id','trip_id']
    trips[tps_cols] = trips[tps_cols].apply(lambda x:x.astype(str))
    if pd.isnull(trips.shape_id).all():
        trips.drop('shape_id',axis = 1, inplace = True)
    else:
        trips.shape_id = trips.shape_id.astype(str)
    return trips

def stop_times_norm(raw_stoptimes):
    stop_times_v =  pd.DataFrame(columns = ['trip_id', 'arrival_time', 'departure_time','stop_id', 'stop_sequence',
        'stop_headsign', 'pickup_type', 'drop_off_type', 'continuous_pickup',
        'continuous_drop_off', 'shape_dist_traveled','timepoint'], index = None)
    stop_times =  pd.concat([stop_times_v, raw_stoptimes],  ignore_index=True)
    stop_times.drop(['stop_headsign', 'pickup_type', 'drop_off_type', 'continuous_pickup',
                     'continuous_drop_off'], axis = 1,  inplace =True)
    # check for NA in time cols
    time_cols_na = sum(stop_times[['arrival_time','departure_time']].isna().sum())
    # Traitement NA
    if time_cols_na > 0 :
        stop_times = stop_times.loc[stop_times['timepoint'] == 1]
        time_cols_na2 = sum(stop_times[['arrival_time','departure_time']].isna().sum())
        if time_cols_na2 > 0:
            stop_times['arrival_time'] = stop_times.groupby('trip_id')['arrival_time'].transform(lambda x: x.ffill().bfill())
            stop_times['departure_time'] = stop_times.groupby('trip_id')['departure_time'].transform(lambda x: x.ffill().bfill())
            time_cols_na3 = sum(stop_times[['arrival_time','departure_time']].isna().sum())
            if time_cols_na3 > 0:
                condition1 = stop_times['arrival_time'].notna()
                condition2 = stop_times['departure_time'].notna()
                stop_times = stop_times.loc[condition1 | condition2]
    stp_t_cols = ['trip_id','arrival_time','departure_time','stop_id']
    stop_times[stp_t_cols] = stop_times[stp_t_cols].apply(lambda x:x.astype(str))
    stop_times['stop_sequence'] = stop_times['stop_sequence'].astype(np.int16)
    stop_times['shape_dist_traveled'] = stop_times['shape_dist_traveled'].astype(np.float32)
    # Drop cols where there's only na value
    stop_times.dropna(how = 'all', axis = 1, inplace = True)
    return stop_times

def calendar_norm(raw_cal):
    calendar_v = pd.DataFrame(
        columns =['service_id', 'monday', 'tuesday', 'wednesday',
                  'thursday', 'friday','saturday', 'sunday', 'start_date', 'end_date'], index = None)
    calendar =  pd.concat([calendar_v, raw_cal],  ignore_index=True)
    calendar.service_id = calendar.service_id.astype(str)
    week_cols = ['monday', 'tuesday','wednesday','thursday','friday','saturday', 'sunday']
    calendar[week_cols] = calendar[week_cols].apply(lambda x: x.astype(bool))
    calendar.start_date = calendar.start_date.astype(np.int32)
    calendar.end_date = calendar.end_date.astype(np.int32)
    return calendar

def cal_dates_norm(raw_caldates):
    calendar_dates_v = pd.DataFrame(columns =['service_id', 'date', 'exception_type'], index = None)
    calendar_dates =  pd.concat([calendar_dates_v, raw_caldates],  ignore_index=True)
    calendar_dates.date = calendar_dates.date.astype(np.int32)
    calendar_dates.service_id = calendar_dates.service_id.astype(str)
    calendar_dates.exception_type = calendar_dates.exception_type.astype(np.int8)
    return calendar_dates

def normalize_raw_data(rawgtfs):
    '''
    adapt gtfs data to standard form. Give a dict with normalized gtfs raw data
    '''
    # agency
    agency = agency_norm(rawgtfs['agency'])
    # routes
    routes = routes_norm(rawgtfs['routes'])
    # stops
    stops = stops_norm(rawgtfs['stops'])
    # trips
    trips = trips_norm(rawgtfs['trips'])
    # stop_times
    stop_times = stop_times_norm(rawgtfs['stop_times'])
    # calendar
    calendar = calendar_norm(rawgtfs['calendar'])
    if len(calendar)==0:
        calendar = None
    # calendar_date
    calendar_dates = cal_dates_norm(rawgtfs['calendar_dates'])
    # shapes
    if rawgtfs.get('shapes') is None:
        result = {'agency' :agency, 'stops' :stops, 'stop_times':stop_times,'routes': routes,'trips': trips,
                    'calendar': calendar,'calendar_dates': calendar_dates}        
    else:
        shapes = rawgtfs['shapes']
        result = {'agency' :agency, 'stops' :stops, 'stop_times':stop_times,'routes': routes,'trips': trips,
                    'calendar': calendar,'calendar_dates': calendar_dates, 'shapes':shapes}
    return result


# transformation 1.1
def service_date_generate(calendar,calendar_dates,dates):
    if calendar_dates is None:
        raise KeyError
    calendar_dates['date'] = pd.to_datetime(calendar_dates['date'],format="%Y%m%d")
    dates['Date_GTFS'] = pd.to_datetime(dates['Date_GTFS'],format="%Y%m%d")
    dates_small = dates[['Date_GTFS','Type_Jour']]
    cal_cols = ['service_id','Date_GTFS','Type_Jour','Semaine','Mois','Annee',
                'Type_Jour_Vacances_A','Type_Jour_Vacances_B','Type_Jour_Vacances_C']
    if calendar is None:
        logger.warning("Le fichier calendar n'est pas présent dans le jeu de données")
        result = calendar_dates.merge(dates, left_on = 'date', right_on = 'Date_GTFS',how = 'left')[cal_cols].sort_values(['service_id','Date_GTFS']).reset_index(drop = True)
    elif sum(calendar['monday']) + sum(calendar['tuesday']) + sum(calendar['wednesday']) + sum(calendar['thursday']) + sum(calendar['friday']) + sum(calendar['saturday']) + sum(calendar['sunday'])  ==0:
        logger.warning("Le fichier calendar n'a pas de jour de semaine valide.")
        result = calendar_dates.merge(dates, left_on = 'date', right_on = 'Date_GTFS',how = 'left')[cal_cols].sort_values(['service_id','Date_GTFS']).reset_index(drop = True)
    else :
        logger.info("Les fichiers calendar et calendar_dates sont bien présents.")
        calendar['start_date'] = pd.to_datetime(calendar['start_date'],format="%Y%m%d")
        calendar['end_date'] = pd.to_datetime(calendar['end_date'],format="%Y%m%d")
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
    if gtfs_norm_dict.get('shapes') is not None:
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

def itineraire_geometry_generate(itineraire:pd.DataFrame) -> pd.DataFrame:
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
    if course.get('geom_shape') is None: # if shapes.txt no exist
        course = course.rename({'geom_shape':'geom'},axis =1)
    else: 
        course = course.rename({'geom_vo':'geom'},axis =1)
    gdf_course = gpd.GeoDataFrame(course, geometry = 'geom')
    cols = ['sous_ligne', 'route_id', 'direction_id','id_ag_origine','stop_name_x','id_ag_destination','stop_name_y', 'geom']  
    group_col = ['sous_ligne', 'route_id','id_ag_origine','stop_name_x', 'id_ag_destination','stop_name_y', 'direction_id','nb_arrets']
    gdf_sl = gdf_course.groupby(group_col,as_index=False).first()
    gdf_sl = gdf_sl.set_geometry('geom')
    gdf_sl = gdf_sl[cols]
    gdf_sl.crs = "EPSG:2154" 
    gdf_sl.rename({'route_id':'id_ligne',
                   'stop_name_x':'ag_origin_name',
                   'stop_name_y':'ag_destination_name'}, axis = 1, inplace = True)
    return gdf_sl

def ligne_generate(sl_gdf,raw_routes ,route_type):
    col = ['id_ligne','geom']
    ligne_gdf = sl_gdf[col].groupby(['id_ligne'], as_index = False).first()
    ligne_gdf = ligne_gdf.set_geometry('geom')
    ligne_gdf.crs = "EPSG:2154" 
    raw_routes.rename({'route_id':'id_ligne'}, axis = 1, inplace = True)
    lignes = pd.merge(raw_routes,ligne_gdf, on = 'id_ligne')
    lignes = pd.merge(lignes, route_type[['Code','Description']], how='left',left_on='route_type', right_on='Code')
    lignes.drop(['Code'], axis = 1,inplace=True)
    lignes.dropna(axis = 1, how = 'all', inplace = True)
    return lignes

def consolidation_ligne(lignes,courses, AG):
    #Afficher OD principal
    crs_1dir = courses[courses['direction_id'] == 0]
    ligne_od_count = crs_1dir.groupby(['route_id', 'id_ag_origine', 'id_ag_destination'])['trip_id'].count().reset_index()
    idx = ligne_od_count.groupby(['route_id'])['trip_id'].idxmax()
    od_principal = ligne_od_count.loc[idx]
    od_principal_simplify = od_principal[['route_id','id_ag_origine','id_ag_destination']]
    AG_simplify = AG[['id_ag', 'stop_name']]
    od_principal1 = pd.merge(od_principal_simplify, AG_simplify, left_on = 'id_ag_origine', right_on= 'id_ag').rename(columns={'stop_name':'Origin'})
    od_principal = pd.merge(od_principal1, AG_simplify, left_on = 'id_ag_destination', right_on= 'id_ag').rename(columns={'stop_name':'Destination'})
    od_principal.rename({'route_id':'id_ligne'}, axis = 1, inplace = True)
    lignes_export = pd.merge(lignes, od_principal, on = 'id_ligne')
    lignes_export.drop(['id_ag_x','id_ag_y','route_color'], axis = 1, inplace=True)
    lignes_export.rename({'Description':'mode'}, axis=1, inplace=True)
    col = ['id_ligne','agency_id','route_short_name','route_long_name','route_type','mode',
           'id_ag_origine','id_ag_destination','Origin','Destination','geom']
    gdf_lignes = gpd.GeoDataFrame(lignes_export[col],geometry = 'geom')
    df_lignes = gdf_lignes.drop('geom',axis = 1)
    return df_lignes,gdf_lignes

def consolidation_sl(sl, lignes):
    ligne_name = lignes[['id_ligne','route_short_name','route_long_name']]
    sl = pd.merge(sl,ligne_name, on = 'id_ligne')
    col =  ['sous_ligne', 'id_ligne','route_short_name','route_long_name', 'direction_id',
            'id_ag_origine','ag_origin_name','id_ag_destination','ag_destination_name', 'geom']
    sl_export = sl[col]
    sl_export = sl_export.rename({'geom_shape':'geom'}, axis=1)
    gdf_sl_export = gpd.GeoDataFrame(sl_export,geometry = 'geom')
    df_sl_export = sl_export.drop('geom', axis=1)
    return df_sl_export, gdf_sl_export

def consolidation_course(courses):
    if courses.get('Dist_shape') is None:
        crs_cols = ['trip_id','route_id', 'sous_ligne' ,'service_id','direction_id','heure_depart','heure_arrive', 
                    'id_ap_origine', 'id_ap_destination','id_ag_origine', 'id_ag_destination', 'nb_arrets','Dist_Vol_Oiseau',
                    'geom_vo']
        courses_export = courses[crs_cols].rename({'geom_vo':'geom'},axis = 1)
    else:
        crs_cols = ['trip_id','route_id', 'sous_ligne' ,'service_id','direction_id','heure_depart','heure_arrive', 
            'id_ap_origine', 'id_ap_destination','id_ag_origine', 'id_ag_destination', 'nb_arrets','Dist_Vol_Oiseau',
            'Dist_shape','geom_shape']
        courses_export = courses[crs_cols].rename({'geom_shape':'geom'},axis = 1)

    courses_export.rename({'trip_id':'id_course',
                           'heure_depart' : 'h_dep_num',
                           'heure_arrive' : 'h_arr_num',
                           'route_id':'id_ligne',
                           'service_id':'id_service'
                           },axis = 1, inplace = True)
    courses_export.loc[:,'heure_depart'] = np.vectorize(heure_from_xsltime)(courses_export.loc[:,'h_dep_num'])
    courses_export.loc[:,'heure_arrive'] = np.vectorize(heure_from_xsltime)(courses_export.loc[:,'h_arr_num'])
    return courses_export

def consolidation_iti(itineraire, courses):
    iti_exp = itineraire.rename({'stop_sequence':'ordre',
                                 'arrival_time':'h_dep_num',
                                 'departure_time':'h_arr_num',
                                 'trip_id':'id_course',
                                 'route_id':'id_ligne',
                                 'service_id':'id_service'
                                 },axis = 1)
    iti_exp.loc[:,'heure_depart'] = np.vectorize(heure_from_xsltime)(iti_exp.loc[:,'h_dep_num'])
    iti_exp.loc[:,'heure_arrive'] = np.vectorize(heure_from_xsltime)(iti_exp.loc[:,'h_arr_num'])
    iti_cols = ['id_course','id_ligne','id_service','direction_id','ordre','id_ap','id_ag','heure_depart','h_dep_num','heure_arrive','h_arr_num','TH']
    iti = iti_exp[iti_cols]
    crs_simple = courses[['id_course','sous_ligne']]
    result = crs_simple.merge(iti, on = 'id_course')
    return result

def consolidation_iti_arc(itineraire_arc,courses):
    itiarc_cols = [ 'id_course', 'id_ligne','sous_ligne','id_service','direction_id',
                   'ordre_a','heure_depart','h_dep_num','heure_arrive','h_arr_num' ,'id_ap_a', 'id_ag_a', 'TH_a', 'ordre_b',
                    'id_ap_b', 'id_ag_b', 'TH_b']
    itineraire_arc.rename({
        'trip_id':'id_course',
        'route_id':'id_ligne',
        'service_id':'id_service',
        'heure_depart':'h_dep_num',
        'heure_arrive':'h_arr_num'},axis = 1, inplace = True)
    itineraire_arc.loc[:,'heure_depart'] = np.vectorize(heure_from_xsltime)(itineraire_arc.loc[:,'h_dep_num'])
    itineraire_arc.loc[:,'heure_arrive'] = np.vectorize(heure_from_xsltime)(itineraire_arc.loc[:,'h_arr_num'])
    crs_simple = courses[['id_course','sous_ligne']]
    iti_arc = crs_simple.merge(itineraire_arc, on = 'id_course')
    itiarc_export = iti_arc[itiarc_cols]
    return itiarc_export

def service_jour_type_generate(service_date,course, type_vacances):
    crs_simplifie = course[['id_course', 'id_ligne', 'id_service']]
    crs_et_plage_date = pd.merge(crs_simplifie, service_date, how = 'left', on = 'id_service')
    nb_crs_parligne_parjtype = crs_et_plage_date.groupby(
    ['Date_GTFS','id_ligne',type_vacances], as_index = False)['id_course'].count().rename(
    {'id_course':'count'}, axis=1)
    nb_jr_per_nb_crs = nb_crs_parligne_parjtype.groupby(
    ['id_ligne', type_vacances, 'count'], as_index=False)['Date_GTFS'].count().rename(
    {'Date_GTFS':'count_days'}, axis=1)
    max_nb_jr_zb = nb_jr_per_nb_crs.groupby(
    ['id_ligne', type_vacances], as_index=False)['count_days'].max().rename(
    {'count_days':'max_days'}, axis=1)
    ncours_jtype = nb_jr_per_nb_crs.merge(max_nb_jr_zb, how = 'left', on = ['id_ligne', type_vacances])
    choix_jtype_1 = nb_crs_parligne_parjtype.merge(
    ncours_jtype, how = 'left', on = ['id_ligne',type_vacances,'count'])
    choix_jtype = choix_jtype_1[
    choix_jtype_1['count_days']==choix_jtype_1['max_days']].groupby(
    ['id_ligne',type_vacances],as_index=False ).agg(
    {'Date_GTFS':'first','count':'max'},as_index=False )
    service_jtype_1 = crs_et_plage_date.merge(
    choix_jtype, how = 'left', on=['id_ligne',type_vacances ,'Date_GTFS']).dropna().reset_index(drop = True)
    service_jtype = service_jtype_1.groupby(
    ['id_ligne', 'id_service', type_vacances], as_index = False).agg({'Date_GTFS':'first'}, as_index = False)
    return service_jtype

def nb_passage_ag(service_jour_type_export, itineraire_export,AG,type_vac):
    iti_typejour = itineraire_export.merge(service_jour_type_export, on =['id_ligne','id_service'])
    nb_passage = iti_typejour.groupby(
        ['id_ag', type_vac], as_index = False)['id_course'].count().sort_values(['id_ag'])
    nb_passage_ag = nb_passage.merge(AG)
    nb_passage_ag = nb_passage_ag.reset_index(drop=True).rename({'id_course':'nb_passage'},axis = 1)
    nb_passage_ag_pv = pd.pivot_table(nb_passage_ag,values = 'nb_passage', index = ['id_ag','stop_name', 'stop_lat','stop_lon' ], columns = type_vac, fill_value = 0,  aggfunc=np.sum).reset_index()
    return nb_passage_ag_pv

def passage_arc(iti_arc, service_jour_type, AG,type_vac):
    node = AG[['id_ag','stop_name','stop_lon','stop_lat']].rename(
        {'id_ag':'NO','stop_name':'NAME','stop_lon':'LON','stop_lat':'LAT'},
        axis = 1)
    plink = iti_arc.groupby(
        ['id_ag_a','id_ag_b'],
        as_index = False)['id_course'].count().reset_index(drop = True)
    plink['ID'] = plink.index
    plink = plink[['ID','id_ag_a','id_ag_b']].rename(
        {'ID':'ID_ARC','id_ag_a':'FROMNODE','id_ag_b':'TONODE'},
        axis = 1)
    iti_typejour = iti_arc.merge(service_jour_type, on =['id_ligne','id_service'])
    nb_passage = iti_typejour.groupby(
        ['id_ag_a', 'id_ag_b', type_vac], as_index = False)['id_course'].count().sort_values(
            ['id_ag_a', 'id_ag_b']).rename(
                {'id_course':'nb_passage', 'id_ag_a' : 'FROMNODE', 'id_ag_b': 'TONODE'},axis = 1)
    nb_passage_pv = pd.pivot_table(nb_passage,values = 'nb_passage', index = ['FROMNODE','TONODE'], columns = type_vac, fill_value = 0,  aggfunc=np.sum).reset_index()
    nb_passage_pv = nb_passage_pv.merge(
        node[['NO','NAME','LON','LAT']], left_on = 'FROMNODE', right_on = 'NO').merge(
            node[['NO','NAME','LON','LAT']], left_on = 'TONODE', right_on = 'NO').drop(
                ['NO_x', 'NO_y'], axis=1).reset_index(drop = True)
    nb_passage_pv['ID']= nb_passage_pv.index
    return nb_passage_pv


def nb_course_ligne(service_jour_type_export, courses_export,type_vac, ligne):
    courses_jtype =  courses_export.merge(service_jour_type_export, on =['id_ligne','id_service'])
    nb_courses_par_ligne = courses_jtype.groupby(
        ['id_ligne',type_vac], as_index = False)['id_course'].count().sort_values(
        ['id_ligne']).rename({'id_course':'nb_courses'},axis = 1)
    nb_courses_par_ligne_pv = pd.pivot_table(nb_courses_par_ligne,
                                             values = 'nb_courses', 
                                             index = ['id_ligne'], 
                                             columns = type_vac, 
                                             fill_value = 0,  
                                             aggfunc=np.sum).reset_index()
    ligne_names = ligne[['id_ligne','route_short_name','route_long_name',
                         'id_ag_origine','id_ag_destination','Origin','Destination']]
    result = ligne_names.merge(nb_courses_par_ligne_pv, on = ['id_ligne'])
    return result

def kcc_course_ligne(service_jour_type_export, courses_export,type_vac, ligne, ShapeNoExist):
    courses_jtype =  courses_export.merge(service_jour_type_export, on =['id_ligne','id_service'])
    if ShapeNoExist:
        m_par_ligne = courses_jtype.groupby(
            ['id_ligne',type_vac], as_index = False)['Dist_Vol_Oiseau'].sum().sort_values(
            ['id_ligne'])
        m_par_ligne['Dist_Vol_Oiseau'] = m_par_ligne['Dist_Vol_Oiseau']/1000
        m_par_ligne_pv = pd.pivot_table(m_par_ligne,
                                                values = 'Dist_Vol_Oiseau', 
                                                index = ['id_ligne'], 
                                                columns = type_vac, 
                                                fill_value = 0,  
                                                aggfunc=np.sum).reset_index()
    else:    
        m_par_ligne = courses_jtype.groupby(
            ['id_ligne',type_vac], as_index = False)['Dist_shape'].sum().sort_values(
            ['id_ligne'])
        m_par_ligne['Dist_shape'] = m_par_ligne['Dist_shape']/1000
        m_par_ligne_pv = pd.pivot_table(m_par_ligne,
                                                values = 'Dist_shape', 
                                                index = ['id_ligne'], 
                                                columns = type_vac, 
                                                fill_value = 0,  
                                                aggfunc=np.sum).reset_index()
    ligne_names = ligne[['id_ligne','route_short_name','route_long_name',
                         'id_ag_origine','id_ag_destination','Origin','Destination']]
    result = ligne_names.merge(m_par_ligne_pv, on = ['id_ligne'])
    return result

def kcc_course_sl(service_jour_type_export, courses_export,type_vac, sous_ligne, ShapeNoExist):
    courses_jtype =  courses_export.merge(service_jour_type_export, on =['id_ligne','id_service'])
    if ShapeNoExist:
        m_par_ligne = courses_jtype.groupby(
            ['sous_ligne',type_vac], as_index = False)['Dist_Vol_Oiseau'].sum().sort_values(
            ['sous_ligne'])
        m_par_ligne['Dist_Vol_Oiseau'] = m_par_ligne['Dist_Vol_Oiseau']/1000
        m_par_ligne_pv = pd.pivot_table(m_par_ligne,
                                                values = 'Dist_Vol_Oiseau', 
                                                index = ['sous_ligne'], 
                                                columns = type_vac, 
                                                fill_value = 0,  
                                                aggfunc=np.sum).reset_index()
    else:
        m_par_ligne = courses_jtype.groupby(
            ['sous_ligne',type_vac], as_index = False)['Dist_shape'].sum().sort_values(
            ['sous_ligne'])
        m_par_ligne['Dist_shape'] = m_par_ligne['Dist_shape']/1000
        m_par_ligne_pv = pd.pivot_table(m_par_ligne,
                                                values = 'Dist_shape', 
                                                index = ['sous_ligne'], 
                                                columns = type_vac, 
                                                fill_value = 0,  
                                                aggfunc=np.sum).reset_index()
    ligne_names = sous_ligne[['sous_ligne','id_ligne','route_short_name','route_long_name',
                              'id_ag_origine'	,'id_ag_destination','ag_origin_name','ag_destination_name']]
    result = ligne_names.merge(m_par_ligne_pv, on = ['sous_ligne'])
    return result


def caract_par_sl(service_jour_type_export,courses_export,debut_HPM , fin_HPM, debut_HPS ,fin_HPS, type_vac,sous_ligne):
    courses_jtype =  courses_export.merge(service_jour_type_export,  on =['id_ligne','id_service'])
    # Premier et dernier départ et nb courses par jour type
    caract = courses_jtype.groupby(['sous_ligne', type_vac],as_index = False).agg(
                 {'h_dep_num':'min',
                  'h_arr_num' : 'max',
                  'id_course' : 'count'}, as_index = False).reset_index(drop=True).rename({'id_course':'Nb_courses', 'h_dep_num' : 'Debut', 'h_arr_num': 'Fin'},axis = 1)
    caract.loc[:,'Duree'] = caract.loc[:,'Fin'] - caract.loc[:,'Debut']
    caract.loc[:,'Passages_par_heure'] = (caract.loc[:,'Nb_courses']/(caract.loc[:,'Duree']*24)).round(1)

    # mask pour filtrer
    mask_FM = courses_jtype['h_dep_num'] < debut_HPM
    mask_HPM = (courses_jtype['h_dep_num'] >= debut_HPM) & (courses_jtype['h_dep_num'] < fin_HPM)
    mask_HC = (courses_jtype['h_dep_num'] >= fin_HPM) & (courses_jtype['h_dep_num'] < debut_HPS)
    mask_HPS = (courses_jtype['h_dep_num'] >= debut_HPS) & (courses_jtype['h_dep_num'] < fin_HPS)
    mask_FS = (courses_jtype['h_dep_num'] >= fin_HPS)

    courses_jtype.loc[mask_FM,'periode'] = 'FM'
    courses_jtype.loc[mask_HPM,'periode'] = 'HPM'
    courses_jtype.loc[mask_HC,'periode'] = 'HC'
    courses_jtype.loc[mask_HPS,'periode'] = 'HPS'
    courses_jtype.loc[mask_FS,'periode'] = 'FS'

    headway = courses_jtype.groupby([type_vac, 'sous_ligne','periode'],as_index = False)['id_course'].count().rename({'id_course':'nb_courses'},axis = 1)
    headway_pv = pd.pivot_table(headway,values = 'nb_courses', index = ['sous_ligne',type_vac], columns = 'periode', fill_value = 0,  aggfunc=np.sum).reset_index()
    duration_FM = (debut_HPM - min(courses_jtype['h_dep_num']))*24*60
    duration_HPM = (fin_HPM - debut_HPM)*24*60
    duration_HC = (debut_HPS - fin_HPM)*24*60
    duration_HPS = (fin_HPS - debut_HPS)*24*60
    duration_FS = (max(courses_jtype['h_dep_num']) - fin_HPS)*24*60
    headway_pv.loc[:,'Headway_FM'] = (duration_FM/headway_pv.loc[:,'FM']).round(1)
    headway_pv.loc[:,'Headway_HPM'] = (duration_HPM/headway_pv.loc[:,'HPM']).round(1)
    headway_pv.loc[:,'Headway_HC'] = (duration_HC/headway_pv.loc[:,'HC']).round(1)
    headway_pv.loc[:,'Headway_HPS'] = (duration_HPS/headway_pv.loc[:,'HPS']).round(1)
    headway_pv.loc[:,'Headway_FS'] = (duration_FS/headway_pv.loc[:,'FS']).round(1)
    headway_pv = headway_pv.replace(np.inf, np.nan).drop(['FM','HPM','HC','HPS','FS'],axis = 1)
    caract_fin = caract.merge(headway_pv,  on =['sous_ligne',type_vac])
    ligne_names = sous_ligne[['sous_ligne','id_ligne','route_short_name','route_long_name',
                              'id_ag_origine'	,'id_ag_destination','ag_origin_name','ag_destination_name']]
    result = ligne_names.merge(caract_fin, on = ['sous_ligne'])
    return result

#start


def userInput():
    time1 = datetime.strptime("07:00", "%H:%M").time()
    time2 = datetime.strptime("09:00", "%H:%M").time()
    time3 = datetime.strptime("17:00", "%H:%M").time()
    time4 = datetime.strptime("19:00", "%H:%M").time()
    # Convert time to decimal fraction of the day
    debut_hpm = time1.hour / 24 + time1.minute / 1440  # 1440 = 24 * 60
    fin_hpm = time2.hour / 24 + time2.minute / 1440
    debut_hps = time3.hour / 24 + time3.minute / 1440
    fin_hps = time4.hour / 24 + time4.minute / 1440
    user_inputs = { 'debut_hpm':debut_hpm,
                    'fin_hpm':fin_hpm,
                    'debut_hps':debut_hps,
                    'fin_hps':fin_hps}
    return user_inputs

user_inputs = userInput()
rawPath = os.path.normpath('./Resources/test_data_2/input')
output_path = os.path.normpath('./Resources/test_data_2/output')
# Set up logging to output directory
meta_file_path = os.path.join(output_path, 'metadata.json')

log_file_path = os.path.join(output_path, f"{datetime.now().strftime('%Y%m%d %H%M')}_traitement.log")
plugin_path = os.path.dirname(__file__)

logger = logging.getLogger("GTFS_miner")  # Use a consistent logger name for the entire application
logger.setLevel(logging.DEBUG)        
# File handler
file_handler = logging.FileHandler(log_file_path)
formatter = logging.Formatter('%(asctime)s:%(name)s:%(message)s')
file_handler.setFormatter(formatter)       
# Stream handler
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)    
# Add handlers
logger.addHandler(file_handler)
logger.addHandler(stream_handler)
logger.info("Traitement des données GTFS commence...")

# Lecture données brutes
logger.info(f"{datetime.now():%H:%M:%S}: Début de la lecture des données brutes")
raw,meta,dates,routeTypes = read_raw_GTFSdata(rawPath, plugin_path)
shapes_not_exist = raw.get('shapes') is None
logger.info(f"{datetime.now():%H:%M:%S}: Fin de la lecture des données brutes")

# Write the data to a JSON file
with open(meta_file_path, 'w', encoding='utf-8') as json_file:
    json.dump(meta, json_file, ensure_ascii=False, indent=4)
logger.info(f"metadata json file wrote to {log_file_path}")
logger.info(f"{datetime.now():%H:%M:%S}: Fin de l'écriture du fichier json.")

# Update data quality tables 
datasetQuality = datasetLevelInfo(meta)

tableQuality = tableLevelInfo(meta)
logger.info(f"{datetime.now():%H:%M:%S}: Fin de la mise à jour des tables qualité des données")

# Normalization
logger.info(f"{datetime.now():%H:%M:%S}: Début de la normalisation des données brutes.")
GTFS_norm = normalize_raw_data(raw)

logger.info(f"{datetime.now():%H:%M:%S}: Fin de la normalisation des données brutes.")

logger.info(f"{datetime.now():%H:%M:%S}: Début de la création des tables enrichies et nettoyées...")
# service date
service_date = service_date_generate(GTFS_norm['calendar'],GTFS_norm['calendar_dates'],dates)
logger.info(f"{datetime.now():%H:%M:%S}: Création de la table service_date terminée.")

# AP AG
AP, AG, marker = ag_ap_generate_reshape(GTFS_norm['stops'])
logger.info(f"{datetime.now():%H:%M:%S}: Création des tables AP AG terminée.")
# Itinéraire
iti = itineraire_generate(GTFS_norm['stop_times'], AP, GTFS_norm['trips'])
logger.info(f"{datetime.now():%H:%M:%S}: Création de la table itinéraire terminée.")
# Itinéraire arc
itiarc = itiarc_generate(iti,AG)
logger.info(f"{datetime.now():%H:%M:%S}: Création de la table itinéraire arc terminée.")
# Course
course = course_generate(iti)
logger.info(f"{datetime.now():%H:%M:%S}: Création de la table course terminée.")

# Shapes : check existance shape and add info if so
course_geom = shape_exist_condition(GTFS_norm,course,iti)
logger.info(f"{datetime.now():%H:%M:%S}: Création de la table course geom terminée.")
# Sous lignes
sl = sl_generate(course_geom,AG)
logger.info(f"{datetime.now():%H:%M:%S}: Création de la table sous ligne terminée.")
logger.info(f"{datetime.now():%H:%M:%S}: Fin de la création des tables enrichies et nettoyées.")

# Mise en formes des tables
logger.info(f"{datetime.now():%H:%M:%S}: Début de la mise en forme des tables d'analyse...")
# Lignes
lignes = ligne_generate(sl,GTFS_norm['routes'] ,routeTypes)
# Ligne - export
lignes_export, gdf_lignes = consolidation_ligne(lignes,course_geom,AG)
logger.info(f"{datetime.now():%H:%M:%S}: Mise en forme de la table ligne terminée.")
# sous ligne - export
sl_export, gdf_sl = consolidation_sl(sl, lignes)
logger.info(f"{datetime.now():%H:%M:%S}: Mise en forme de la table sous ligne terminée.")
# Course - export
course_export = consolidation_course(course_geom)
logger.info(f"{datetime.now():%H:%M:%S}: Mise en forme de la table course terminée.")
# Itinéraire - export
iti_export = consolidation_iti(iti,course_export)
logger.info(f"{datetime.now():%H:%M:%S}: Mise en forme de la table itinéraire terminée.")
# Itinéraire arc - export
itiarc_export = consolidation_iti_arc(itiarc,course_export)
logger.info(f"{datetime.now():%H:%M:%S}: Mise en forme de la table itinéraire arc terminée.")
logger.info(f"{datetime.now():%H:%M:%S}: Fin de la mise en forme des tables d'analyse...")

logger.info(f"{datetime.now():%H:%M:%S}: Début de la création des tables analytiques...")
# Service jour type
type_vac = "Type_Jour_Vacances_A"
service_jtype = service_jour_type_generate(service_date,course_export,type_vac)
logger.info(f"{datetime.now():%H:%M:%S}: création de la table service jour type terminée.")
# Passage par ag
psg_ag =  nb_passage_ag(service_jtype, iti_export,AG,type_vac)
logger.info(f"{datetime.now():%H:%M:%S}: création de la table passage ag terminée.")
# Passage par arc
psg_arc = passage_arc(itiarc_export, service_jtype, AG,type_vac)
logger.info(f"{datetime.now():%H:%M:%S}: création de la table passage arc terminée.")
# Nombre de course par ligne
nb_crs_lignes = nb_course_ligne(service_jtype, course_export,type_vac, lignes_export)
logger.info(f"{datetime.now():%H:%M:%S}: création de la table nb courses par ligne terminée.")
# Kcc par ligne
kcc_ligne = kcc_course_ligne(service_jtype, course_export,type_vac, lignes_export, shapes_not_exist)
logger.info(f"{datetime.now():%H:%M:%S}: création de la table kcc par ligne terminée.")
# Kcc par sous ligne
kcc_sl = kcc_course_sl(service_jtype, course_export,type_vac, sl_export, shapes_not_exist)
logger.info(f"{datetime.now():%H:%M:%S}: création de la table kcc par sous ligne terminée.")
# Caractéristiques des sous-ligne
carac_sl = caract_par_sl(service_jtype,course_export, user_inputs['debut_hpm'] , user_inputs['fin_hpm'], 
                            user_inputs['debut_hps'] ,user_inputs['fin_hps'], type_vac,sl_export)
logger.info(f"{datetime.now():%H:%M:%S}: création de la table caractéristics des sous ligne terminée.")
logger.info(f"{datetime.now():%H:%M:%S}: Fin de la création des tables analytiques.")

# Export des fichiers
logger.info(f"{datetime.now():%H:%M:%S}: Début de l'export des tables...")    
AG.to_csv(f"{output_path}/A_1_Arrets_Generiques.csv", sep=';', index = False)
AP.to_csv(f"{output_path}/A_2_Arrets_Physiques.csv", sep=';', index = False)
lignes_export.to_csv(f"{output_path}/B_1_Lignes.csv", sep=';', index = False)
sl_export.to_csv(f"{output_path}/B_2_Sous_Lignes.csv", sep=';', index = False)
course_export.to_csv(f"{output_path}/C_1_Courses.csv", sep=';', index = False)
iti_export.to_csv(f"{output_path}/C_2_Itineraire.csv", sep=';', index = False)
itiarc_export.to_csv(f"{output_path}/C_3_Itineraire_Arc.csv", sep=';', index = False)
service_date.to_csv(f"{output_path}/D_1_Service_Dates.csv", sep=';', index = False)
service_jtype.to_csv(f"{output_path}/D_2_Service_Jourtype.csv", sep=';', index = False)
psg_ag.to_csv(f"{output_path}/E_1_Nombre_Passage_AG.csv", sep=';', index = False)
psg_arc.to_csv(f"{output_path}/E_4_Nombre_Passage_Arc.csv", sep=';', index = False)
nb_crs_lignes.to_csv(f"{output_path}/F_1_Nombre_Courses_Lignes.csv", sep=';', index = False)
carac_sl.to_csv(f"{output_path}/F_2_Caract_SousLignes.csv", sep=';', index = False)
kcc_ligne.to_csv(f"{output_path}/F_3_KCC_Lignes.csv", sep=';', index = False)
kcc_sl.to_csv(f"{output_path}/F_4_KCC_Sous_Ligne.csv", sep=';', index = False)
gdf_sl.to_file(f"{output_path}/G_1_Trace_Sous_Ligne.gpkg", driver="GPKG")
gdf_lignes.to_file(f"{output_path}/G_2_Trace_Ligne.gpkg", driver="GPKG")
logger.info(f"{datetime.now():%H:%M:%S}: Fin de l'export des tables.")
