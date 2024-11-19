import pandas as pd


def df_converter(df):
    # Convert specified columns to desired types
    df = df.astype(
        {
            "ID_Base": "string",
            "ID_Timestamp": "string",
            "stop_number": "int",
            "line": "string",
            "starting_station_IBNR": "string",
            "zip": "string",
            "last_station": "string",
            "IBNR": "string",
            "long": "float",
            "lat": "float",
            "arrival_delay_m": "float",
            "departure_delay_m": "float",
            "info": "string",
            "canceled": "bool",
        }
    )

    # Convert columns with timestamp format
    timestamp_columns = [
        "arrival_plan",
        "departure_plan",
        "arrival_change",
        "departure_change",
    ]
    for column in timestamp_columns:
        df[column] = pd.to_datetime(df[column])

    return df

def clean_up_df(df, columns):
    return df.drop(columns=columns)

def filter_canceled(df):
    df = df[df["canceled"] == False]
    df = df.drop(columns=["canceled"])
    return df