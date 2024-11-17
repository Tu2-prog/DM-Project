from preprocessors.preprocessor import Preprocessor
import numpy as np
import pandas as pd
from datetime import datetime


class LagInfoExtractor(Preprocessor):
    def __init__(self, name="LagInfoExtractor") -> None:
        super().__init__(name)

    def convert_df(self, df):
        # Convert 'station_number' to numeric
        df["stop_number"] = pd.to_numeric(df["stop_number"], errors="coerce")

        # Convert 'arrival_plan' and 'departure_plan' to datetime
        df["arrival_plan"] = pd.to_datetime(df["arrival_plan"])
        df["departure_plan"] = pd.to_datetime(df["departure_plan"])

        # Ensure 'arrival_delay_m' and 'departure_delay_m' are numeric
        df["arrival_delay_m"] = pd.to_numeric(df["arrival_delay_m"], errors="coerce")
        df["departure_delay_m"] = pd.to_numeric(
            df["departure_delay_m"], errors="coerce"
        )

    def parse_departure_time(self, departure_time_str):
        if not isinstance(departure_time_str, str) or len(departure_time_str) != 10:
            return None
        try:
            year = int("20" + departure_time_str[0:2])  # Assuming years are 2020+
            month = int(departure_time_str[2:4])
            day = int(departure_time_str[4:6])
            hour = int(departure_time_str[6:8])
            minute = int(departure_time_str[8:10])
            dt = datetime(year, month, day, hour, minute)
        except ValueError:
            dt = None
        return dt

    def calculate_weighted_avg_delay_vectorized(self, group):
        delays = group["arrival_delay_m"].fillna(0).values
        weights = np.arange(1, len(delays) + 1)
        weighted_delays = delays * weights
        numerator = np.cumsum(weighted_delays)
        denominator = np.cumsum(weights)
        weighted_avg = numerator / denominator
        # Shift weighted_avg by one to exclude current delay
        weighted_avg_prev = np.insert(weighted_avg[:-1], 0, 0)
        group["weighted_avg_prev_delay"] = weighted_avg_prev
        return group

    # Optimized function to calculate distance features without for loops
    def calculate_distance_features_vectorized(self, group):
        group = group.sort_values("stop_number")
        latitudes = group["lat"].values
        longitudes = group["long"].values
        R = 6371  # Earth radius in kilometers

        # Convert degrees to radians
        lat_rad = np.radians(latitudes)
        lon_rad = np.radians(longitudes)

        # Compute differences between consecutive coordinates
        delta_phi = np.diff(lat_rad)
        delta_lambda = np.diff(lon_rad)

        # Compute haversine formula
        a = (
            np.sin(delta_phi / 2.0) ** 2
            + np.cos(lat_rad[:-1])
            * np.cos(lat_rad[1:])
            * np.sin(delta_lambda / 2.0) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        distances = R * c  # distances between consecutive points

        # distances_to_prev_stop
        distances_to_prev = np.insert(distances, 0, 0)  # Insert 0 for the first stop

        # distances_from_origin
        distances_from_origin = np.cumsum(distances_to_prev)

        # total_distance
        total_distance = (
            distances_from_origin[-1] if len(distances_from_origin) > 0 else 0
        )

        # distance_to_next_stop
        distances_to_next = np.append(distances, 0)  # Append 0 for the last stop

        group = group.copy()
        group["distance_to_prev_stop"] = distances_to_prev
        group["distance_from_origin"] = distances_from_origin
        group["total_distance"] = total_distance  # Same for all rows in the group
        group["distance_to_next_stop"] = distances_to_next

        return group

    def transform(self, df):
        self.logger.info("Preprocess data")
        df["departure_time"] = df["ID_Timestamp"].apply(self.parse_departure_time)
        self.convert_df(df)
        df = df.sort_values(by=["ID_Base", "ID_Timestamp", "stop_number"])
        df.reset_index(drop=True, inplace=True)
        # Group the data by journey
        df["prev_arrival_delay_m"] = df.groupby(["ID_Base", "departure_time"])[
            "arrival_delay_m"
        ].shift(1)
        df["prev_departure_delay_m"] = df.groupby(["ID_Base", "departure_time"])[
            "departure_delay_m"
        ].shift(1)

        # Replace NaN values (which occur at the first station) with 0
        df["prev_arrival_delay_m"] = df["prev_arrival_delay_m"].fillna(0)
        df["prev_departure_delay_m"] = df["prev_departure_delay_m"].fillna(0)
        df = df.groupby(["ID_Base", "departure_time"], group_keys=False).apply(
            self.calculate_weighted_avg_delay_vectorized
        )
        df["cumulative_delay"] = df.groupby(["ID_Base", "departure_time"])[
            "arrival_delay_m"
        ].cumsum()
        # Calculate the gain in delay over stations
        df["delay_gain"] = (
            df.groupby(["ID_Base", "departure_time"])["cumulative_delay"]
            .diff()
            .fillna(0)
        )
        df["max_station_number"] = df.groupby(["ID_Base", "departure_time"])[
            "stop_number"
        ].transform("max")

        # Calculate the ratio of current station number to max station number
        df["station_progress"] = df["stop_number"] / df["max_station_number"]
        df["origin_departure_plan"] = df.groupby(["ID_Base", "departure_time"])[
            "departure_plan"
        ].transform("first")

        # Calculate planned elapsed time since departure from origin station
        df["planned_elapsed_time"] = (
            df["arrival_plan"] - df["origin_departure_plan"]
        ).dt.total_seconds() / 60  # in minutes

        # Calculate total planned time for the journey
        df["total_planned_time"] = (
            df.groupby(["ID_Base", "departure_time"])["arrival_plan"].transform("last")
            - df["origin_departure_plan"]
        ).dt.total_seconds() / 60  # in minutes

        # Calculate ratio of elapsed time to total time
        df["time_progress"] = df["planned_elapsed_time"] / df["total_planned_time"]

        # Calculate planned travel time to the next stop
        df["next_arrival_plan"] = df.groupby(["ID_Base", "departure_time"])[
            "arrival_plan"
        ].shift(-1)
        df["planned_travel_time_to_next_stop"] = (
            df["next_arrival_plan"] - df["departure_plan"]
        ).dt.total_seconds() / 60  # in minutes

        # Calculate the ratio of station progress to time progress (progress_ratio = station_progress / time_progress)
        df["progress_ratio"] = df["station_progress"] / df["time_progress"].replace(
            {0: np.nan}
        )

        # Handle infinite or NaN values
        df["progress_ratio"] = (
            df["progress_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
        )

        df["long"] = pd.to_numeric(df["long"], errors="coerce")
        df["lat"] = pd.to_numeric(df["lat"], errors="coerce")

        # Remove entries with missing coordinates
        df = df.dropna(subset=["long", "lat"])

        df = df.groupby(["ID_Base", "departure_time"], group_keys=False).apply(
            self.calculate_distance_features_vectorized
        )

        # Calculate ratio of distance from origin to total distance
        df["distance_progress"] = df["distance_from_origin"] / df[
            "total_distance"
        ].replace({0: np.nan})

        # Handle infinite or NaN values
        df["distance_progress"] = (
            df["distance_progress"].replace([np.inf, -np.inf], np.nan).fillna(0)
        )
        city_avg_delay = df.groupby("city")["arrival_delay_m"].transform("mean")

        # Add the average city delay as a feature
        df["avg_city_delay"] = city_avg_delay
        self.logger.info("Finalize preprocessing")
        df.to_csv("DBtrainrides_optimized.csv", index=False)
        return df
