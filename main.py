import pandas as pd
from preprocessors import info_messages
from preprocessors import train_type

if __name__ == "__main__":
    # Preprocessing steps unified until now
    train_df = pd.read_csv("DBtrainrides.csv")
    cleaner = info_messages.InfoMessageCleaner("InfoMessageCleaner")
    cleaner.transform_df(train_df)
    # train_type_cf = train_type.TrainTypeClassifier("TrainTypeClassifier")
    # train_type_cf.transform_df(train_df)

    # Code for training
