import pandas as pd

from preprocessors import info_messages, train_type, lag_info_extractor, path_exploder
from utils.utils import df_converter, clean_up_df

if __name__ == "__main__":
    # Preprocessing steps
    train_df = pd.read_csv("DBtrainrides.csv")

    path_exploder = path_exploder.PathExploder()
    train_df = path_exploder.transform(train_df)
    train_df = df_converter(train_df)

    cleaner = info_messages.InfoMessageCleaner()
    train_df = cleaner.transform_df(train_df)

    lag_info_extractor = lag_info_extractor.LagInfoExtractor()
    train_df = lag_info_extractor.transform(train_df)

    train_type_cf = train_type.TrainTypeClassifier()
    train_df = train_type_cf.transform_df(train_df)
    
    train_df = clean_up_df(train_df, ["info", "line_prefix", "train_type", "last_station", "city"])
    train_df.to_csv("DBtrainrides_final_result.csv", index=False)
    # Code for training
