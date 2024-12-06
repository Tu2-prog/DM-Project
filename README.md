# DM-Project HWS24

This is the code repository for the Data Mining project of the HWS24 at the University Mannheim.

# Prerequisites

It is recommended to use some kind of virtual environment, i.e. conda and install all the necessary packages through the **requirements.txt** file provided in this repository.

```bash
pip install -r requirements.txt
```

# Quickstart

This project's code is two-fold. The data preprocessing pipeline can be executed using the **main.py** script.

```bash
python main.py
```

Due to time constraints, there was no time to objectify the code for training the selected models.
There are two folders containing jupyter notebooks:
* ./training
* ./analysis

In the folder *./training* contains the jupyter notebooks containing code for training the models.
In the folder *./analysis* the jupyter contains the jupyter notebooks containing the code for analysing and profiling the dataset and subsequent versions of the dataset.

# Data
Since Github does not allow to upload large files. The data is provided on demand.

Optionally:

The data can be downloaded from the following links:
* https://www.kaggle.com/datasets/nokkyu/deutsche-bahn-db-delays
* https://www.kaggle.com/datasets/headsortails/train-stations-in-europe

These csv files can be used to execute the preprocessing pipeline.

The Python version used here was **3.11.3**, however for the baseline_regression_tree please use **3.8**.
