import pandas as pd
from preprocessors.info_messages import InfoMessageCleaner
from preprocessors.train_type import TrainTypeClassifier

if __name__ == "__main__":
    # Preprocessing steps unified until now
    train_df = pd.read_csv("DBtrainrides.csv")
    cleaner = InfoMessageCleaner("InfoMessageCleaner")
    cleaner.transform_df(train_df)
    train_type_cf = TrainTypeClassifier("TrainTypeClassifier")
    train_type_cf.transform_df(train_df)

    # Code for training
