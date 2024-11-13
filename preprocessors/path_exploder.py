from preprocessors.preprocessor import Preprocessor
from utils.utils import df_converter

import pandas as pd


class PathExploder(Preprocessor):

    def __init__(self, name="PathExploder") -> None:
        super().__init__(name)

    def create_station_mapping(self, df):
        self.logger.info(
            "Create station_mapping to understand the spread of the stations"
        )
        station_mapping_df = df["starting_station_IBNR"].value_counts().reset_index()
        station_mapping_df.columns = ["starting_station_IBNR", "count"]
        return station_mapping_df

    def filter_rows_with_max_id(self, df):
        self.logger.info("Keep the rows with the max ID")
        max_stop_numbers_df = (
            df.loc[df.groupby(["ID_Base", "ID_Timestamp"])["ID_Stop_Number"].idxmax()]
            .drop_duplicates(subset=["ID_Base", "ID_Timestamp"], keep="first")
            .sort_values(by=["starting_station_IBNR", "ID_Base", "ID_Timestamp"])
        )
        return max_stop_numbers_df

    def split_up_path(self, df):
        exploded_stations_df = (
            df.explode("last_station")
            .drop(columns=["path"])
            .assign(
                stop_number=lambda df: df.groupby(
                    ["ID_Base", "ID_Timestamp"]
                ).cumcount()
                + 1
            )
            .sort_values(
                by=["starting_station_IBNR", "ID_Base", "ID_Timestamp", "stop_number"]
            )
            .reset_index(drop=True)
        )
        return exploded_stations_df

    def add_ibnr(self, df):
        self.logger.info(
            "Load ibnr_stations_index.csv and clean 'last_station' and 'Station Name' columns"
        )
        ibnr_index_df = pd.read_csv("ibnr_stations_index.csv")
        df["last_station"] = df["last_station"].str.strip().str.lower()
        ibnr_index_df["Station Name"] = (
            ibnr_index_df["Station Name"].str.strip().str.lower()
        )

        self.logger.info(
            "Merge with ibnr_index_df on 'last_station' and drop unnecessary columns"
        )
        result_df = df.merge(
            ibnr_index_df, how="left", left_on="last_station", right_on="Station Name"
        ).drop(columns=["ID_Stop_Number", "Station Name"])
        return result_df

    def extract_cancelation_info(self, df):
        df["canceled"] = df["path"].isna().astype(bool)
        df.drop(
            columns=[
                "station",
                "zip",
                "state",
                "city",
                "category",
                "line",
                "path",
                "eva_nr",
                "arrival_delay_check",
                "departure_delay_check",
            ],
            inplace=True,
        )
        return df

    def merge_ibnr_train_df(self, ibnr_df, train_df):
        ibnr_df["stop_number"].astype(int)
        train_df["ID_Stop_Number"] = train_df["ID_Stop_Number"].astype(int)
        exploded_stations_df_with_ibnr_time_df = ibnr_df.merge(
            train_df,
            left_on=["ID_Base", "ID_Timestamp", "stop_number"],
            right_on=["ID_Base", "ID_Timestamp", "ID_Stop_Number"],
            how="left",
        ).sort_values(by=["starting_station_IBNR", "ID_Base", "ID_Timestamp"])
        return exploded_stations_df_with_ibnr_time_df

    def finalize(self, df):
        self.logger.info("Finalize Preprocessing")
        df = df.drop("ID_Stop_Number", axis=1)

        # Place 'stop_number' after 'ID_Timestamp'
        columns = df.columns.tolist()
        columns.remove("stop_number")
        columns.insert(columns.index("ID_Timestamp") + 1, "stop_number")
        df = df[columns]

        # Sort by relevant columns
        df = df.sort_values(by=["starting_station_IBNR", "ID_Base", "ID_Timestamp"])
        # Replace NA with False in canceled
        df["canceled"] = df["canceled"].fillna(False)
        return df

    def split_up_id(self, df):
        df[["ID_Base", "ID_Timestamp", "ID_Stop_Number"]] = df["ID"].str.rsplit(
            "-", n=2, expand=True
        )
        df.drop(columns=["ID"], inplace=True)
        new_column_order = ["ID_Base", "ID_Timestamp", "ID_Stop_Number"] + [
            col
            for col in df.columns
            if col not in ["ID_Base", "ID_Timestamp", "ID_Stop_Number"]
        ]
        df = df[new_column_order]
        return df

    def transform(self, df):
        self.logger.info("Start Preprocessing data")
        self.logger.info("Split up ID")
        df = self.split_up_id(df)

        train_rides_df_copy = df.copy()

        self.logger.info(
            "Rename 'eva_nr' to 'starting_station_IBNR' and drop unnecessary columns"
        )
        df.rename(columns={"eva_nr": "starting_station_IBNR"}, inplace=True)
        df.drop(
            columns=[
                "station",
                "state",
                "city",
                "long",
                "lat",
                "category",
                "arrival_plan",
                "departure_plan",
                "arrival_change",
                "departure_change",
                "arrival_delay_m",
                "departure_delay_m",
                "info",
                "arrival_delay_check",
                "departure_delay_check",
            ],
            inplace=True,
        )

        self.logger.info(
            "Create station_mapping to understand the spread of the stations"
        )
        station_mapping_df = self.create_station_mapping(df)

        self.logger.info("Keep the rows with the max ID")
        max_stop_numbers_df = self.filter_rows_with_max_id(df)

        self.logger.info("Split up Path variable")
        max_stop_numbers_df = max_stop_numbers_df.assign(
            last_station=max_stop_numbers_df["path"].str.split("|")
        )
        exploded_stations_df = self.split_up_path(max_stop_numbers_df)

        exploded_stations_df_with_ibnr_df = self.add_ibnr(exploded_stations_df)
        exploded_stations_df_with_ibnr_df["last_station"] = (
            exploded_stations_df_with_ibnr_df["last_station"].replace("", pd.NA)
        )

        self.logger.info("Create new column 'cancelled'")
        train_rides_df_copy = self.extract_cancelation_info(train_rides_df_copy)

        self.logger.info(
            "Merge with exploded_stations_df_with_ibnr_df on relevant keys"
        )
        exploded_stations_df_with_ibnr_time_df = self.merge_ibnr_train_df(
            exploded_stations_df_with_ibnr_df, train_rides_df_copy
        )

        exploded_stations_df_with_ibnr_time_df = self.finalize(
            exploded_stations_df_with_ibnr_time_df
        )

        exploded_stations_df_with_ibnr_time_df = df_converter(
            exploded_stations_df_with_ibnr_time_df
        )

        df = exploded_stations_df_with_ibnr_time_df
        df.to_csv("DBtrainrides_exploded_stations.csv")
        return df
