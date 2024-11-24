import requests
import time
import os
import pandas as pd
import numpy as np
from preprocessors.preprocessor import Preprocessor


class GeoEncoder(Preprocessor):
    def __init__(self, name="GeoEncoder") -> None:
        super().__init__(name)

    def get_coordinates_here(self, station_name, api_key):
        url = f"https://geocode.search.hereapi.com/v1/geocode?q={station_name}&apiKey={api_key}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            if data['items']:
                location = data['items'][0]['position']
                return location['lat'], location['lng']
        return None, None
    
    def update_lat_long(self, row, api_key):
        # Only update if lat or long is NaN
        if pd.isna(row['lat']) or pd.isna(row['long']):
            lat, long = self.get_coordinates_here(row['clear_station_name'], api_key)
            # Replace lat and long with the new coordinates
            row['lat'] = lat
            row['long'] = long
        return row

    def transform(self, df):
        self.logger.info("Load European IBNR data")
        df.loc[df['stop_number'] == 1, 'IBNR'] = df['starting_station_IBNR']
        df = df.astype({"IBNR": "float"})
        european_train_stations = pd.read_csv("train_stations_europe.csv")
        european_train_stations.rename(columns={"uic": "IBNR", "longitude": "long", "latitude": "lat"}, inplace=True)
        european_train_stations = european_train_stations[european_train_stations["country"]=="DE"]

        self.logger.info("Filter out missing long lats")
        df_missing_lat_long = df[df['lat'].isna() | df['long'].isna()]

        self.logger.info("Process missing long lats sequentially in chunks")
        
        n_chunks = 10
        chunks = np.array_split(df_missing_lat_long, n_chunks)

        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1}/{n_chunks}...")

            merged_chunk = chunk.merge(
                european_train_stations[['IBNR', 'lat', 'long']],
                on='IBNR', how='left', suffixes=('', '_european_train_stations')
            )

            # Replace NaN lat/long in the original dataframe (df_train_rides) for the current chunk
            df.loc[chunk.index, 'lat'] = df.loc[chunk.index, 'lat'].fillna(merged_chunk['lat_european_train_stations'])
            df.loc[chunk.index, 'long'] = df.loc[chunk.index, 'long'].fillna(merged_chunk['long_european_train_stations'])

        self.logger.info("Retrived missing coordinates")
        df.to_csv("DBtrainrides_replaced_missing_coordinates.csv", index=False)

        preprocessed_df = pd.read_csv("station_coordinates_final_manually_updated.csv")

        incomplete_df = df[df[["lat","long"]].isna().all(axis=1)]
        df_sorted = incomplete_df.sort_values(by=['ID_Base', 'stop_number'])

        # Shift the 'last_station' within each 'ID_Base' to get the next stop's 'last_station' in the current row
        df_sorted['clear_station_name'] = df_sorted.groupby('ID_Base')['last_station'].shift(-1)
        merged = pd.merge(
            df_sorted, preprocessed_df,
            on='clear_station_name',
            how='left',
            suffixes=('_original', '_new')
        )

        # Replace NaN values in 'long' and 'lat' using the values from the second DataFrame
        merged['long'] = merged['long_original'].combine_first(merged['long_new'])
        merged['lat'] = merged['lat_original'].combine_first(merged['lat_new'])

        # Drop intermediate columns
        result = merged.drop(columns=['long_original', 'long_new', 'lat_original', 'lat_new'])

        api_key = os.environ.get("HERE_API_KEY")

        result = result.apply(self.update_lat_long, axis=1, api_key=api_key)
        result.to_csv("test.csv",index=False)

        missing_long_lat_after_second_cleaning = result[(
                                                    (result['lat'].isna()) &
                                                    (result['long'].isna()))]

        missing_long_lat_after_second_cleaning = missing_long_lat_after_second_cleaning.apply(self.update_lat_long, axis=1, api_key=api_key)

        combined_df = pd.concat([result, missing_long_lat_after_second_cleaning])

        final_result_df = combined_df.drop_duplicates(subset=['ID_Base', 'ID_Timestamp', 'stop_number'])

        combined_df_all_rides = pd.concat([df, final_result_df])

        final_result_df_all_rides = combined_df_all_rides.drop_duplicates(subset=['ID_Base', 'ID_Timestamp', 'stop_number'])

        final_result_df_all_rides.to_csv("DBtrainrides_restored_lat_long.csv",index=False)
        return final_result_df_all_rides