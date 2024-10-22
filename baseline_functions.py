import pandas as pd
import math
import chardet
import numpy as np
from scipy.spatial.distance import pdist

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