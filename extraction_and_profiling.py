import pandas as pd
import numpy as np
import os
import datetime
from .baseline_functions import *
import logging

logger = logging.getLogger('GTFS_miner')

class FileDataTypeError(Exception):
    pass

# Main function
def read_raw_GTFSdata(dirpath, plugin_path):
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
    Dates = pd.read_csv((plugin_path+"/Resources/Calendrier.txt"), encoding="utf-8",
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
    route_type = pd.read_csv((plugin_path+"/Resources/route_types.csv"),sep = ";", encoding="utf-8")
    #encoding='utf-8'
    logger.info("route_type table is imported.")
    return route_type

def initialize_metadata(dirpath):
    dirname = os.path.basename(dirpath)
    return {
        "dataset": dirname,
        "processing_timestamp": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
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
        jsonFile['start_date'] = int(df['start_date'].min())
    if 'end_date' in df.columns:
        jsonFile['end_date'] = int(df['end_date'].max())
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


