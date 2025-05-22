import pandas as pd

def load_dataset(path):
    df = pd.read_csv(path)
    return df

def clean_crash_data(df):
    # Drop irrelevant columns
    columns_to_drop = ['STREET_DIRECTION', 'STREET_NO', 'DOORING_I', 'PHOTOS_TAKEN_I',
                       'STATEMENTS_TAKEN_I', 'WORK_ZONE_TYPE']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Fill missing values
    df['LATITUDE'] = df['LATITUDE'].astype('float64')
    df['LONGITUDE'] = df['LONGITUDE'].astype('float64')
    df['POSTED_SPEED_LIMIT'] = df['POSTED_SPEED_LIMIT'].astype('float64')
    df['CRASH_HOUR'] = df['CRASH_HOUR'].fillna(df['CRASH_HOUR'].median()).astype('int64')
    df['INJURIES_TOTAL'] = df['INJURIES_TOTAL'].fillna(0).astype('int64')

    cat_cols = ['DEVICE_CONDITION', 'WEATHER_CONDITION', 'LIGHTING_CONDITION', 'TRAFFICWAY_TYPE', 'CRASH_TYPE']
    for col in cat_cols:
        df[col] = df[col].fillna('UNKNOWN').astype('category')

    df = df.dropna(subset=['LATITUDE', 'LONGITUDE', 'LOCATION'])
    df = df.drop_duplicates()
    
    return df
