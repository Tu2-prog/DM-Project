import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from datetime import datetime
import matplotlib.pyplot as plt
import logging
from preprocessors.preprocessor import Preprocessor


class TrainTypeClassifier(Preprocessor):
    def __init__(self, name="TrainTypeClassifier") -> None:
        super().__init__(name)

    def categorize_line(self, line):
        if pd.isnull(line) or line.strip() == '':
            return 'No Prefix'
        # Extract the alphabetic prefix from the line
        prefix = ''.join(filter(str.isalpha, line))
        if prefix == '':
            return 'No Prefix'
        elif prefix in ['RE', 'RB']:
            return 'RE/RB Prefix'
        else:
            return 'Other Prefix'

    def determine_line_prefix(self, df):
        df['line_prefix'] = df['line'].str.extract(r'^([A-Za-z]+)', expand=False)

    def parse_id(self, id_str):
        if pd.isnull(id_str):
            return None, None, None
        parts = id_str.split('-')
        if len(parts) == 3:
            route_id = parts[0]
            departure_time_str = parts[1]
            station_number = parts[2]
        elif len(parts) == 4 and parts[0] == '':
            # This is when route_id starts with a minus sign
            route_id = '-' + parts[1]
            departure_time_str = parts[2]
            station_number = parts[3]
        else:
            # ID does not conform to expected pattern
            return None, None, None
        return route_id, departure_time_str, station_number

    def parse_departure_time(self, departure_time_str):
        if not isinstance(departure_time_str, str) or len(departure_time_str) != 10:
            return None
        try:
            year = int('20' + departure_time_str[0:2])  # Assuming years are 2020+
            month = int(departure_time_str[2:4])
            day = int(departure_time_str[4:6])
            hour = int(departure_time_str[6:8])
            minute = int(departure_time_str[8:10])
            dt = datetime(year, month, day, hour, minute)
        except ValueError:
            dt = None
        return dt

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        R = 6371  # Earth radius in kilometers
        phi1 = radians(lat1)
        phi2 = radians(lat2)
        delta_phi = radians(lat2 - lat1)
        delta_lambda = radians(lon2 - lon1)
        # Compute haversine formula
        a = sin(delta_phi / 2.0) ** 2 + \
            cos(phi1) * cos(phi2) * sin(delta_lambda / 2.0) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        distance = R * c
        return distance  # Distance in kilometers

    def compute_average_distance(self, group):
        group = group.sort_values('station_number')
        # Compute distances between consecutive stops
        latitudes = group['lat'].values
        longitudes = group['long'].values
        distances = []
        for i in range(len(group) - 1):
            lat1, lon1 = latitudes[i], longitudes[i]
            lat2, lon2 = latitudes[i + 1], longitudes[i + 1]
            distance = self.haversine_distance(lat1, lon1, lat2, lon2)
            distances.append(distance)
        # Calculate average distance
        if distances:
            avg_distance = np.mean(distances)
        else:
            avg_distance = 0  # Only one stop in the journey
        group['avg_distance_between_stops'] = avg_distance
        return group

    def classify_train_type(self, avg_distance, threshold=3):
        if avg_distance <= threshold:
            return 'Tram'
        else:
            return 'Regional Train'

    def final_classification(self, row):
        if row['line_category'] in ['Regional Train', 'Tram']:
            return row['line_category']
        else:
            return row['train_type']

    def transform_df(self, dataframe):
        self.logger.info("Preprocess data")
        dataframe['line_category'] = dataframe['line'].apply(self.categorize_line)
        self.determine_line_prefix(dataframe)
        dataframe[['route_id', 'departure_time_str', 'station_number']] = dataframe['ID'].apply(
            lambda x: pd.Series(self.parse_id(x))
        )
        dataframe['departure_time'] = dataframe['departure_time_str'].apply(self.parse_departure_time)
        dataframe['station_number'] = pd.to_numeric(dataframe['station_number'], errors='coerce')
        dataframe['long'] = pd.to_numeric(dataframe['long'], errors='coerce')
        dataframe['lat'] = pd.to_numeric(dataframe['lat'], errors='coerce')
        dataframe = dataframe.dropna(subset=['long', 'lat'])
        dataframe = dataframe.sort_values(by=['route_id', 'departure_time', 'station_number'])

        self.logger.info("Compute Final Train Type")
        grouped_dataframe = dataframe.groupby(['route_id', 'departure_time']).apply(self.compute_average_distance)
        grouped_dataframe[['route_id', 'departure_time', 'avg_distance_between_stops']].drop_duplicates()
        grouped_dataframe['train_type'] = grouped_dataframe['avg_distance_between_stops'].apply(
            self.classify_train_type)
        grouped_dataframe['final_train_type'] = grouped_dataframe.apply(self.final_classification, axis=1)

        # The plot already exists in this repository in directory plots so only execute this when necessary
        # self.logger.info("Plot avg distance per stope")
        # self.visualize_distance_distribution(grouped_dataframe)

        self.logger.info("Split grouped data and save it")
        # Create a DataFrame for Regional Trains
        df_regional_train = grouped_dataframe[grouped_dataframe['final_train_type'] == 'Regional Train'].copy()

        # Create a DataFrame for Trams
        df_tram = grouped_dataframe[grouped_dataframe['final_train_type'] == 'Tram'].copy()

        # Save dataframes for later inspection
        df_regional_train.to_csv('./regional_trains.csv', index=False)
        df_tram.to_csv('./trams.csv', index=False)

    def visualize_distance_distribution(self, df):
        avg_distances = df[['route_id', 'departure_time', 'avg_distance_between_stops']].drop_duplicates()[
            'avg_distance_between_stops']

        plt.figure(figsize=(10, 6))
        plt.hist(avg_distances, bins=100)
        plt.xlabel('Average Distance Between Stops (km)')
        plt.ylabel('Number of Journeys')
        plt.title('Distribution of Average Distances Between Stops')
        plt.savefig("./plots/distribution_avg_distance.png")
        plt.clf()
