from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessors.preprocessor import Preprocessor


class TrainTypeClassifier(Preprocessor):
    def __init__(self, name="TrainTypeClassifier") -> None:
        super().__init__(name)

    def categorize_line(self, line):
        if pd.isnull(line) or line.strip() == "":
            return "No Prefix"
        # Extract the alphabetic prefix from the line
        prefix = "".join(filter(str.isalpha, line))
        if prefix == "":
            return "No Prefix"
        elif prefix in ["RE", "RB"]:
            return "RE/RB Prefix"
        else:
            return "Other Prefix"

    def determine_line_prefix(self, df):
        df["line_prefix"] = df["line"].str.extract(r"^([A-Za-z]+)", expand=False)

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

    def haversine_vectorised(self, lat1, lon1, lat2, lon2):
        # Earth radius in kilometres
        R = 6371
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Vectorised haversine calculation
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        a = (
            np.sin(delta_lat / 2.0) ** 2
            + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2.0) ** 2
        )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return R * c

    def compute_average_distance(self, dataframe):
        # Calculate average distances for each group without apply
        dataframe["lat_next"] = dataframe.groupby(["ID_Base", "departure_time"])[
            "lat"
        ].shift(-1)
        dataframe["long_next"] = dataframe.groupby(["ID_Base", "departure_time"])[
            "long"
        ].shift(-1)

        # Filter rows with missing next coordinates
        mask = dataframe["lat_next"].notna() & dataframe["long_next"].notna()

        # Calculate distances
        dataframe["distance"] = 0  # Default distance as zero
        dataframe.loc[mask, "distance"] = self.haversine_vectorised(
            dataframe.loc[mask, "lat"].values,
            dataframe.loc[mask, "long"].values,
            dataframe.loc[mask, "lat_next"].values,
            dataframe.loc[mask, "long_next"].values,
        )

        # Calculate average distance per group
        avg_distances = dataframe.groupby(["ID_Base", "departure_time"])[
            "distance"
        ].transform("mean")
        dataframe["avg_distance_between_stops"] = avg_distances

        # Drop temporary columns
        dataframe.drop(columns=["lat_next", "long_next", "distance"], inplace=True)

        return dataframe

    def classify_train_type(self, avg_distance, threshold=3):
        if avg_distance <= threshold:
            return "Tram"
        else:
            return "Regional Train"

    def final_classification(self, row):
        if row["line_category"] in ["Regional Train", "Tram"]:
            return row["line_category"]
        else:
            return row["train_type"]

    def transform_df(self, dataframe):
        self.logger.info("Preprocess data")
        dataframe["line_category"] = dataframe["line"].apply(self.categorize_line)
        self.determine_line_prefix(dataframe)
        dataframe["departure_time"] = dataframe["ID_Timestamp"].apply(
            self.parse_departure_time
        )

        # Drop rows with missing coordinates and sort by necessary columns
        dataframe = dataframe.dropna(subset=["long", "lat"])
        dataframe = dataframe.sort_values(
            by=["ID_Base", "departure_time", "stop_number"]
        )

        # Compute average distances for each group without groupby-apply
        self.logger.info("Compute Final Train Type")
        dataframe = self.compute_average_distance(dataframe)

        # Remove duplicates before classification
        # dataframe = dataframe.drop_duplicates(subset=['ID_Base', 'departure_time', 'avg_distance_between_stops'])
        dataframe["train_type"] = dataframe["avg_distance_between_stops"].apply(
            self.classify_train_type
        )
        dataframe["final_train_type"] = dataframe.apply(
            self.final_classification, axis=1
        )
        dataframe.drop(["train_type", "line_prefix", "last_station"])
        # Save grouped data for inspection
        dataframe.to_csv("DBtrainrides_final_train_type.csv")
        self.logger.info("Split grouped data and save it")
        df_regional_train = dataframe[
            dataframe["final_train_type"] == "Regional Train"
        ].copy()
        df_tram = dataframe[dataframe["final_train_type"] == "Tram"].copy()
        df_regional_train.to_csv("./regional_trains.csv", index=False)
        df_tram.to_csv("./trams.csv", index=False)

        # The plot already exists in this repository in directory plots so only execute this when necessary
        self.logger.info("Plot avg distance per stop")
        self.visualize_distance_distribution(dataframe)

    def visualize_distance_distribution(self, df):
        avg_distances = df[
            ["ID_Base", "departure_time", "avg_distance_between_stops"]
        ].drop_duplicates()["avg_distance_between_stops"]

        plt.figure(figsize=(10, 6))
        plt.hist(avg_distances, bins=100)
        plt.xlabel("Average Distance Between Stops (km)")
        plt.ylabel("Number of Journeys")
        plt.title("Distribution of Average Distances Between Stops")
        plt.savefig("./plots/distribution_avg_distance.png")
        plt.clf()
