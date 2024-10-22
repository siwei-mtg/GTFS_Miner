import pandas as pd
import numpy as np
from .baseline_functions import *

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
            ['id_ligne',type_vac], as_index = False)['DIST_Vol_Oiseau'].sum().sort_values(
            ['id_ligne'])
        m_par_ligne['DIST_Vol_Oiseau'] = m_par_ligne['DIST_Vol_Oiseau']/1000
        m_par_ligne_pv = pd.pivot_table(m_par_ligne,
                                                values = 'DIST_Vol_Oiseau', 
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
            ['sous_ligne',type_vac], as_index = False)['DIST_Vol_Oiseau'].sum().sort_values(
            ['sous_ligne'])
        m_par_ligne['DIST_Vol_Oiseau'] = m_par_ligne['DIST_Vol_Oiseau']/1000
        m_par_ligne_pv = pd.pivot_table(m_par_ligne,
                                                values = 'DIST_Vol_Oiseau', 
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


