import pandas as pd
import io


def infoBuffer(df):
    buffer = io.StringIO()
    df.info(buf=buffer)
    info_str = buffer.getvalue()
    return info_str

def DQ_stops(stops):
    # Info table
    info_str = infoBuffer(stops)
    # check if any null values in key columns 
    key_col = ['stop_lon','stop_lat']
    stops_null = stops.loc[stops[key_col].isnull().any(axis = 1)]
    stops_with_null_in_xy = ', '.join(stops_null['stop_id'].astype(str))
    
    msg_attention = f" Les coordonnées des arrêts suivants sont vide:  {stops_with_null_in_xy}"
    return msg_attention, info_str

def DQ_stop_times(stop_times):
    pass

def DQ_shape(courses, trips):
    '''
    A appliquer après création de la table courses
    '''
    corr = trips[['id_course_num','shape_id']]
    courses = pd.merge(courses,corr, on = 'id_course_num')
    combinaison_sl_shape = courses.groupby(
        ['sous_ligne','shape_id'])['id_course_num'].count().reset_index()
    cnt_sl_shape = combinaison_sl_shape.groupby(
        ['sous_ligne'])['id_course_num'].count().reset_index().rename(
            {'id_course_num': 'nb_shape_id'}, axis= 1 
        )
    cnt_sl_multi_shape = cnt_sl_shape.loc[cnt_sl_shape['nb_shape_id'] >1]
    cnt_shape_sl = combinaison_sl_shape.groupby(['shape_id'])['id_course_num'].count().reset_index().rename(
            {'id_course_num': 'nb_sl'}, axis= 1 
        )
    cnt_shape_multi_sl = cnt_shape_sl.loc[cnt_shape_sl['nb_sl'] >1]
    return cnt_sl_multi_shape, cnt_shape_multi_sl
