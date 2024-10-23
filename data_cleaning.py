import pandas as pd
import numpy as np
import os
import datetime
import logging
from .baseline_functions import *

logger = logging.getLogger('GTFS_miner')

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
    calendar[week_cols] = calendar[week_cols].apply(lambda x: x.astype(np.bool8))
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
