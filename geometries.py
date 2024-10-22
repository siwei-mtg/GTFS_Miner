import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon

def geom_sous_ligne(courses, trips, shapes, sl):
    '''
    A appliquer après création de la table courses
    '''
    crs_sample = courses.groupby(['sous_ligne'])['id_course_num'].first().reset_index() # course représentative
    corr = trips[['id_course_num','shape_id']] # correspondane trip shape
    corr_sl_shp = crs_sample.merge(corr, on = 'id_course_num').groupby(['sous_ligne'])['shape_id'].first().reset_index() # ajouter shape_id sur course
    corr = corr_sl_shp[['sous_ligne','shape_id']] # créer correspondance sous-ligne shape
    sl_simple = sl[['sous_ligne','id_ligne_num','route_short_name','route_long_name']] # simplifier table sous-ligne
    corr2 = sl_simple.merge(corr, on = 'sous_ligne') 
    shp_sl = shapes.merge(corr2, on = 'shape_id') # correspondance shape => sous-ligne
    # créer 1 à 1 shape => sl relationship  ,
    shp_sl['geometry'] = [Point(xy) for xy in zip(shp_sl['shape_pt_lon'],shp_sl['shape_pt_lat'])]
    return shp_sl

def create_linestring_by_group(group):
    group = group.sort_values(by='shape_pt_sequence')
    return LineString(zip(group['shape_pt_lon'],group['shape_pt_lat']))

def create_linestring_shape(df, ligne_export, sl):
    gdf = df.groupby(['sous_ligne','id_ligne_num','route_short_name','route_long_name']).apply(lambda x: create_linestring_by_group(x)).reset_index()
    gdf.columns = ['sous_ligne', 'id_ligne_num','route_short_name','route_long_name','geometry']
    gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
    # Set CRS (WGS84)
    gdf.crs = "EPSG:4326"
    gdf_projected = gdf.to_crs(epsg = 2154)
    gdf_projected['Dist_shape'] = gdf_projected.length
    # Save the result as a shapefile (optional)
    gdf_ligne = gdf_projected.dissolve(by = ['id_ligne_num','route_short_name','route_long_name']).reset_index()
    gdf_ligne['Dist_shape'] = gdf_ligne.length

    gdf_sl = pd.merge(gdf_projected, sl[['sous_ligne','id_ag_num_debut','id_ag_num_terminus','ag_origin_name','ag_destination_name']])
    gdf_ligne = pd.merge(gdf_ligne, ligne_export[['id_ligne_num','id_ag_num_debut','id_ag_num_terminus','Origin','Destination']])
    return gdf_sl,gdf_ligne