import pandas as pd
import re
import logging
from matplotlib.pyplot import plt
from preprocessors.preprocessor import Preprocessor

class InfoMessageCleaner(Preprocessor):
    def __init__(self) -> None:
        super().__init__()

    def __init__(self, name) -> None:
        super().__init__()
        logging.basicConfig()
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

    def check_info(self, df):
        """Checks if an info message is present and produces a boolean attribute for the dataframe."""

        df["info_present"] = False
        for index, row in df.iterrows():
            if pd.notna(row['info']):
                df.at[index, "info_present"] = True

    def transform_info_message(self, df):
        """Clean the info messages if present or add a dummy string and add the values as a new attribute into the dataframe."""

        df["transformed_info_message"] = "No message"
        for index, row in df.iterrows():
            if pd.notna(row["info"]):
                df.at[index, "transformed_info_message"] = re.sub(r'\.\s*\(.*?\)', '', row["info"])

    def compute_statistics(self, df):
        """Computes matrix for statistics in terms of relation between a present info and a potential delay."""
        return df.groupby(['info_present', 'arrival_delay_check']).size().unstack(fill_value=0)
    
    def transform_df(self, dataframe):
        self.logger.info("Preprocess data")
        self.check_info(dataframe)
        self.transform_info_message(dataframe)

    def visualize_statistics(self, df):
        mean_delay = df.groupby("transformed_info_message")["arrival_delay_m"].mean()

        plt.figure(figsize=(10, 6))
        mean_delay.plot(kind='bar', color='skyblue')
        plt.xlabel("Transformed Info Message")
        plt.ylabel("Average Arrival Delay (minutes)")
        plt.title("Average Arrival Delay by Transformed Info Message")
        plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels if needed
        plt.tight_layout()  # Adjust layout to make room for labels
        plt.savefig("./plots/avg-delay-per-message.png")
        plt.clf()
      