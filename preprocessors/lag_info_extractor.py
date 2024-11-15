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

    def transform(self, dataframe):
        self.logger.info("Starting data preprocessing pipeline")
        self.logger.info("Parse the id")
        self.convert_df(dataframe)
        dataframe = self._preprocess_data(dataframe)
        dataframe = self._add_previous_station_data(dataframe)
        dataframe = self._calculate_delays_and_progress(dataframe)
        dataframe = self._calculate_station_progress(dataframe)
        dataframe = self._calculate_time_progress(dataframe)
        dataframe = self._calculate_distance_progress(dataframe)
        dataframe = self._add_city_delay_feature(dataframe)

        self.logger.info("Data transformation complete")
        dataframe.to_csv("DBtrainrides_optimized.csv")
        return dataframe

    def _preprocess_data(self, dataframe):
        """Preprocesses initial data columns and sorting."""
        self.logger.info("Preprocess data")
        dataframe["departure_time"] = dataframe["ID_Timestamp"].apply(
            self.parse_departure_time
        )
        dataframe = dataframe.sort_values(
            by=["ID_Base", "departure_time", "stop_number"]
        )
        return dataframe

    def _add_previous_station_data(self, dataframe):
        """Adds delay information from previous stations."""
        self.logger.info("Incorporate data from Previous Stations")
        dataframe["prev_arrival_delay_m"] = (
            dataframe.groupby(["ID_Base", "departure_time"])["arrival_delay_m"]
            .shift(1)
            .fillna(0)
        )
        dataframe["prev_departure_delay_m"] = (
            dataframe.groupby(["ID_Base", "departure_time"])["departure_delay_m"]
            .shift(1)
            .fillna(0)
        )
        return dataframe

    def _calculate_delays_and_progress(self, dataframe):
        """Calculates weighted delay, cumulative delay, and delay gain."""
        self.logger.info("Calculate Average Weighted Delay from previous stops")
        dataframe = dataframe.groupby(
            ["ID_Base", "departure_time"], group_keys=False
        ).apply(self.calculate_weighted_avg_delay_vectorized)

        dataframe["cumulative_delay"] = dataframe.groupby(
            ["ID_Base", "departure_time"]
        )["arrival_delay_m"].cumsum()
        dataframe["delay_gain"] = (
            dataframe.groupby(["ID_Base", "departure_time"])["cumulative_delay"]
            .diff()
            .fillna(0)
        )
        return dataframe

    def _calculate_station_progress(self, dataframe):
        """Calculates progress through stations and the ratio of station progress."""
        dataframe["max_station_number"] = dataframe.groupby(
            ["ID_Base", "departure_time"]
        )["stop_number"].transform("max")
        dataframe["station_progress"] = (
            dataframe["stop_number"] / dataframe["max_station_number"]
        )
        return dataframe

    def _calculate_time_progress(self, dataframe):
        """Calculates time-based progress ratios."""
        dataframe["origin_departure_plan"] = dataframe.groupby(
            ["ID_Base", "departure_time"]
        )["departure_plan"].transform("first")

        dataframe["planned_elapsed_time"] = (
            dataframe["arrival_plan"] - dataframe["origin_departure_plan"]
        ).dt.total_seconds() / 60  # in minutes

        dataframe["total_planned_time"] = (
            dataframe.groupby(["ID_Base", "departure_time"])["arrival_plan"].transform(
                "last"
            )
            - dataframe["origin_departure_plan"]
        ).dt.total_seconds() / 60  # in minutes

        dataframe["time_progress"] = (
            dataframe["planned_elapsed_time"] / dataframe["total_planned_time"]
        )

        dataframe["next_arrival_plan"] = dataframe.groupby(
            ["ID_Base", "departure_time"]
        )["arrival_plan"].shift(-1)
        dataframe["planned_travel_time_to_next_stop"] = (
            dataframe["next_arrival_plan"] - dataframe["departure_plan"]
        ).dt.total_seconds() / 60  # in minutes

        dataframe["progress_ratio"] = dataframe["station_progress"] / dataframe[
            "time_progress"
        ].replace({0: np.nan})
        dataframe["progress_ratio"] = (
            dataframe["progress_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0)
        )
        return dataframe

    def _calculate_distance_progress(self, dataframe):
        """Calculates distance-based progress ratios and filters out NaN location data."""
        dataframe = dataframe.dropna(subset=["long", "lat"])
        dataframe = dataframe.groupby(
            ["ID_Base", "departure_time"], group_keys=False
        ).apply(self.calculate_distance_features_vectorized)

        dataframe["distance_progress"] = dataframe["distance_from_origin"] / dataframe[
            "total_distance"
        ].replace({0: np.nan})
        dataframe["distance_progress"] = (
            dataframe["distance_progress"].replace([np.inf, -np.inf], np.nan).fillna(0)
        )
        return dataframe

    def _add_city_delay_feature(self, dataframe):
        """Adds the average city delay as a new feature."""
        city_avg_delay = dataframe.groupby("city")["arrival_delay_m"].transform("mean")
        dataframe["avg_city_delay"] = city_avg_delay
        return dataframe
