import pandas as pd

from preprocessors import info_messages , lag_info_extractor, path_exploder, geo_encoder
from utils.utils import df_converter, clean_up_df, filter_canceled, ordinal_scaling, normalize_dates

if __name__ == "__main__":
    # Preprocessing steps
    train_df = pd.read_csv("DBtrainrides.csv")

    path_exploder = path_exploder.PathExploder()
    train_df = path_exploder.transform(train_df)
    
    train_df = df_converter(train_df)

    cleaner = info_messages.InfoMessageCleaner()
    train_df = cleaner.transform_df(train_df)

    geo_encoder = geo_encoder.GeoEncoder()
    train_df = geo_encoder.transform(train_df)

    train_df = df_converter(train_df)

    lag_info_extractor = lag_info_extractor.LagInfoExtractor()
    train_df = lag_info_extractor.transform(train_df)
    
    columns = [
        "info", 
        "last_station", 
        "city",
        "line",
        "starting_station_IBNR",
        "zip",
        "arrival_change",
        "departure_change",
        "info_present",
        "clear_station_name",
        "departure_time",
        "origin_departure_plan",
        "planned_elapsed_time",
        "total_planned_time",
        "next_arrival_plan",
        "planned_travel_time_to_next_stop",
        "progress_ratio",
        "distance_to_prev_stop",
        "distance_to_next_stop",
        "distance_from_origin",
        "distance_progress",
        "total_distance",
        "avg_city_delay",
        "departure_delay_m",
        "cumulative_delay",
        "delay_gain",
        "time_progress",
    ]

    train_df = clean_up_df(train_df, columns)
    train_df = filter_canceled(train_df)
    mesage_order = ["No message", "Information", "Bauarbeiten", "Störung", "Großstörung"]
    train_df = ordinal_scaling(train_df, "transformed_info_message", "info_label_encoded", mesage_order)
    train_df = normalize_dates(train_df)
    train_df.to_csv("DBtrainrides_final_result.csv", index=False)
