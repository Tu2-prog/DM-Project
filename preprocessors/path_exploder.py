from preprocessors.preprocessor import Preprocessor
from utils.utils import df_converter

import pandas as pd


class PathExploder(Preprocessor):

    def __init__(self, name="PathExploder") -> None:
        super().__init__(name)

    def transform(self, df):
        self.logger.info("Split up ID")
        df[["ID_Base", "ID_Timestamp", "ID_Stop_Number"]] = df["ID"].str.rsplit(
            "-", n=2, expand=True
        )
        df["ID_Stop_Number"] = pd.to_numeric(df["ID_Stop_Number"])

        df.drop(columns=["ID"], inplace=True)
        new_column_order = ["ID_Base", "ID_Timestamp", "ID_Stop_Number"] + [
            col
            for col in df.columns
            if col not in ["ID_Base", "ID_Timestamp", "ID_Stop_Number"]
        ]
        df = df[new_column_order]
        df_copy = df.copy()

        df.rename(columns={"eva_nr": "starting_station_IBNR"}, inplace=True)
        df.drop(
            columns=[
                "station",
                "state",
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
        max_stop_numbers_df = (
            df.loc[df.groupby(["ID_Base", "ID_Timestamp"])["ID_Stop_Number"].idxmax()]
            .drop_duplicates(subset=["ID_Base", "ID_Timestamp"], keep="first")
            .sort_values(by=["starting_station_IBNR", "ID_Base", "ID_Timestamp"])
        )
        max_stop_numbers_df = max_stop_numbers_df.assign(
            last_station=max_stop_numbers_df["path"].str.split("|")
        )
        exploded_stations_df = (
            max_stop_numbers_df.explode("last_station")
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
        ibnr_index_df = pd.read_csv("ibnr_stations_index.csv")
        exploded_stations_df["last_station"] = (
            exploded_stations_df["last_station"].str.strip().str.lower()
        )
        ibnr_index_df["Station Name"] = (
            ibnr_index_df["Station Name"].str.strip().str.lower()
        )
        exploded_stations_df_with_ibnr_df = exploded_stations_df.merge(
            ibnr_index_df, how="left", left_on="last_station", right_on="Station Name"
        ).drop(columns=["ID_Stop_Number", "Station Name"])
        exploded_stations_df_with_ibnr_df["last_station"] = (
            exploded_stations_df_with_ibnr_df["last_station"].replace("", pd.NA)
        )
        exploded_stations_df_with_no_ibnr_df = exploded_stations_df_with_ibnr_df[
            exploded_stations_df_with_ibnr_df["last_station"].isna()
        ]
        df_copy["canceled"] = df_copy["path"].isna().astype(bool)
        # Drop unnecessary columns from train_rides_df_copy
        df_copy.drop(
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
        # Merge with exploded_stations_df_with_ibnr_df on relevant keys
        exploded_stations_df_with_ibnr_time_df = (
            exploded_stations_df_with_ibnr_df.merge(
                df_copy,
                left_on=["ID_Base", "ID_Timestamp", "stop_number"],
                right_on=["ID_Base", "ID_Timestamp", "ID_Stop_Number"],
                how="left",
            ).sort_values(by=["starting_station_IBNR", "ID_Base", "ID_Timestamp"])
        )
        # Drop duplicate column and reorder columns
        exploded_stations_df_with_ibnr_time_df = (
            exploded_stations_df_with_ibnr_time_df.drop("ID_Stop_Number", axis=1)
        )

        # Place 'stop_number' after 'ID_Timestamp'
        columns = exploded_stations_df_with_ibnr_time_df.columns.tolist()
        columns.remove("stop_number")
        columns.insert(columns.index("ID_Timestamp") + 1, "stop_number")
        exploded_stations_df_with_ibnr_time_df = exploded_stations_df_with_ibnr_time_df[
            columns
        ]

        # Sort by relevant columns
        exploded_stations_df_with_ibnr_time_df = (
            exploded_stations_df_with_ibnr_time_df.sort_values(
                by=["starting_station_IBNR", "ID_Base", "ID_Timestamp"]
            )
        )

        # Replace NA with False in canceled
        exploded_stations_df_with_ibnr_time_df["canceled"] = (
            exploded_stations_df_with_ibnr_time_df["canceled"].fillna(False)
        )
        exploded_stations_df_with_ibnr_time_df = df_converter(
            exploded_stations_df_with_ibnr_time_df
        )

        self.logger.info("Save the data")
        exploded_stations_df_with_ibnr_time_df.to_csv(
            "DBtrainrides_exploded_stations.csv", index=False
        )
        self.logger.info("Finalize the preprocessing")
        return exploded_stations_df_with_ibnr_time_df
