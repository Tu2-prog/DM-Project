import pandas as pd

from preprocessors import info_messages, train_type


def df_converter(df):
    # Convert specified columns to desired types
    df = df.astype({
        'ID_Base': 'string',
        'ID_Timestamp': 'string',
        'stop_number': 'int',
        'line': 'string',
        'starting_station_IBNR': 'string',
        'zip': 'string',
        'last_station': 'string',
        'IBNR': 'string',
        'long': 'float',
        'lat': 'float',
        'arrival_delay_m': 'float',
        'departure_delay_m': 'float',
        'info': 'string',
        'canceled': 'bool'
    })

    # Convert columns with timestamp format
    timestamp_columns = ['arrival_plan', 'departure_plan', 'arrival_change', 'departure_change']
    for column in timestamp_columns:
        df[column] = pd.to_datetime(df[column])

    return df

if __name__ == "__main__":
    # Preprocessing steps
    train_df = pd.read_csv("exploded_stations.csv")

    train_df = df_converter(train_df)

    cleaner = info_messages.InfoMessageCleaner("InfoMessageCleaner")
    cleaner.transform_df(train_df)

    train_type_cf = train_type.TrainTypeClassifier("TrainTypeClassifier")
    train_type_cf.transform_df(train_df)

    # Code for training
