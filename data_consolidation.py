import pandas as pd
import numpy as np
from .baseline_functions import *
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

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
    od_principal = od_principal.rename({'route_id':'id_ligne'}, axis = 1)
    lignes_export = pd.merge(lignes, od_principal, on = 'id_ligne')
    lignes_export = lignes_export.drop(['id_ag_x','id_ag_y','route_color'], axis = 1)
    lignes_export = lignes_export.rename({'Description':'mode'}, axis=1)
    col = ['id_ligne','agency_id','route_short_name','route_long_name','route_type','mode',
           'id_ag_origine','id_ag_destination','Origin','Destination','geom']
    gdf_lignes = gpd.GeoDataFrame(lignes_export[col],geometry = 'geom')
    df_lignes = gdf_lignes.drop('geom',axis = 1)
    return df_lignes,gdf_lignes

def consolidation_sl(sl, lignes):
    ligne_name = lignes[['id_ligne','route_short_name','route_long_name']]
    sl = pd.merge(sl,ligne_name, on = 'id_ligne')
    if sl.get('geom_shape') is not None:
        col =  ['sous_ligne', 'id_ligne','route_short_name','route_long_name', 'direction_id',
                'id_ag_origine','ag_origin_name','id_ag_destination','ag_destination_name', 'geom_shape']
        sl_export = sl[col]
        sl_export.rename({'geom_shape':'geom'}, axis=1,inplace=True)
    else:
        col =  ['sous_ligne', 'id_ligne','route_short_name','route_long_name', 'direction_id',
                'id_ag_origine','ag_origin_name','id_ag_destination','ag_destination_name', 'geom_vo']
        sl_export = sl[col]
        sl_export.rename({'geom_vo':'geom'}, axis=1,inplace=True)
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

    courses_export = courses_export.rename({'trip_id':'id_course',
                                            'heure_depart' : 'h_dep_num',
                                            'heure_arrive' : 'h_arr_num',
                                            'route_id':'id_ligne',
                                            'service_id':'id_service'
                                            },axis = 1)
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
    itineraire_arc= itineraire_arc.rename({
                                            'trip_id':'id_course',
                                            'route_id':'id_ligne',
                                            'service_id':'id_service',
                                            'heure_depart':'h_dep_num',
                                            'heure_arrive':'h_arr_num'},axis = 1)
    itineraire_arc.loc[:,'heure_depart'] = np.vectorize(heure_from_xsltime)(itineraire_arc.loc[:,'h_dep_num'])
    itineraire_arc.loc[:,'heure_arrive'] = np.vectorize(heure_from_xsltime)(itineraire_arc.loc[:,'h_arr_num'])
    crs_simple = courses[['id_course','sous_ligne']]
    iti_arc = crs_simple.merge(itineraire_arc, on = 'id_course')
    itiarc_export = iti_arc[itiarc_cols]
    return itiarc_export