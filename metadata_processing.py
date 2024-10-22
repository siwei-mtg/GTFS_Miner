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