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

def fill_missing_times(group):
    # Iterate through the rows in the group and find rows with NaT values
    for i in range(len(group)):
        # Check if both arrival_plan and departure_plan are NaT
        if pd.isna(group.iloc[i]['arrival_plan']) and pd.isna(group.iloc[i]['departure_plan']):
            # Try to find the next row in the group with a valid arrival_plan
            next_row = group.iloc[i + 1:] if i + 1 < len(group) else pd.DataFrame()
            
            # If the next row exists and has a valid arrival_plan, fill the NaT
            if not next_row.empty and pd.notna(next_row.iloc[0]['arrival_plan']):
                next_arrival = next_row.iloc[0]['arrival_plan']
                # Subtract 5 minutes from the next valid arrival_plan
                group.at[group.index[i], 'arrival_plan'] = next_arrival - pd.Timedelta(minutes=5)
                # Set departure_plan equal to arrival_plan (or adjust logic if needed)
                group.at[group.index[i], 'departure_plan'] = group.iloc[i]['arrival_plan']
    return group

def ordinal_scaling(df, column, new_target_column, ordering):
    df[column] = pd.Categorical(
        df['transformed_info_message'],
        categories=ordering,
        ordered=True
    )
    
    df[new_target_column] = df[column].cat.codes
    return df

def normalize_dates(df):
    df['arrival_plan'] = pd.to_datetime(df['arrival_plan'], errors='coerce')
    df['departure_plan'] = pd.to_datetime(df['departure_plan'], errors='coerce')

    df = df.sort_values(by=["ID_Base", "ID_Timestamp", "stop_number"])
    df = df.groupby(["ID_Base", "ID_Timestamp"], group_keys=False).apply(fill_missing_times)

    # Find the min and max for arrival and departure
    arrival_min = df['arrival_plan'].min()
    arrival_max = df['arrival_plan'].max()
    departure_min = df['departure_plan'].min()
    departure_max = df['departure_plan'].max()

    # Normalize arrival and departure times to [0, 1]
    df['arrival_normalized'] = (df['arrival_plan'] - arrival_min) / (arrival_max - arrival_min)
    df['departure_normalized'] = (df['departure_plan'] - arrival_min) / (arrival_max - arrival_min)

    df = df.dropna(subset=['arrival_plan', 'departure_plan'])
    # Replace naN values in targets with 0 because they are all coming from the first stop of a train ride (others are filtered out above):
    df["arrival_delay_m"]= df["arrival_delay_m"].fillna(0)

    # Replace strange NaN in IBNR, long and lat
    df["IBNR"] = df["IBNR"].fillna(0.0)
    return df