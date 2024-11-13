import pandas as pd

from preprocessors import info_messages, train_type, lag_info_extractor, path_exploder
from utils.utils import df_converter

if __name__ == "__main__":
    # Preprocessing steps
    train_df = pd.read_csv("DBtrainrides.csv")

    lag_info_extractor = lag_info_extractor.LagInfoExtractor()
    train_df = lag_info_extractor.transform(train_df)

    path_exploder = path_exploder.PathExploder()
    train_df = path_exploder.transform(train_df)

    train_df = df_converter(train_df)

    cleaner = info_messages.InfoMessageCleaner()
    cleaner.transform_df(train_df)

    train_type_cf = train_type.TrainTypeClassifier()
    train_type_cf.transform_df(train_df)

    # Code for training
