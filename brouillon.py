import pandas as pd

dates = pd.read_csv("./Resources/Calendrier.txt", encoding="utf-8",
                    sep = "\t", parse_dates=['Date_Num'], dtype={'Type_Jour': 'int32'})
calendar = pd.read_csv('./Resources/test_data_2/input/calendar.txt')
calendar_dates = pd.read_csv('./Resources/test_data_2/input/calendar_dates.txt')

calendar['start_date'] = pd.to_datetime(calendar['start_date'],format="%Y%m%d")
calendar['end_date'] = pd.to_datetime(calendar['end_date'],format="%Y%m%d")
calendar_dates['date'] = pd.to_datetime(calendar_dates['date'],format="%Y%m%d")
dates['Date_GTFS'] = pd.to_datetime(dates['Date_GTFS'],format="%Y%m%d")
dates_small = dates[['Date_GTFS','Type_Jour']]

date_range_per_service = []
for _, row in calendar.iterrows():
    dates_service = dates_small.loc[(dates['Date_GTFS'] >= row['start_date']) & (dates['Date_GTFS'] <= row['end_date']) ].copy()
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
cal_cols = ['service_id','Date_GTFS','Type_Jour','Semaine','Mois','Annee',
            'Type_Jour_Vacances_A','Type_Jour_Vacances_B','Type_Jour_Vacances_C']
result = service_dates[cal_cols]
print(result.head())
