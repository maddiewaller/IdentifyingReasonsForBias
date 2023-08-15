import copy
import pandas as pd
import numpy as np
from os import path
from project.paths import PROJECT_ROOT

def categorise_age(age):
    if age < 25 or age > 60:
        return 'YoungOrOld'
    else:
        return 'MidAge'

def load_data():
    """
    Loads the original full dataset
    @return: the training and test datasets combined
    """
    data_path = path.join(PROJECT_ROOT, 'raw_data/bank_marketing/bank-full.csv')
    df = pd.read_csv(data_path, sep=';')
    return df


def preprocess():
    """
    Preprocesses the original dataset and saves the preprocessed dataframe to a csv file
    """
    df_orig = load_data()

    df = copy.deepcopy(df_orig)

    df['age'] = df['age'].astype(int)
    df['balance'] = df['balance'].astype(int)
    df['day'] = df['day'].astype(int)
    df['duration'] = df['duration'].astype(int)

    # Categorise continuous attributes
    # categorise age into two Midage and YoungAndOld
    df['age'] = df['age'].apply(categorise_age)

    # Categorise the numerical columns in the same way as
    # Le Quy et. al (2021) A survey on datasets for fairness-aware machine learning
    balance_bins = [-float('inf'), 0, float('inf')]  # Binning balance as 0, >0
    balance_labels = ['0', '>0']
    df['balance'] = pd.cut(df['balance'], bins=balance_bins, labels=balance_labels)

    day_bins = [-float('inf'), 15, float('inf')]  # Binning day as ≤15, >15
    day_labels = ['≤15', '>15']
    df['day'] = pd.cut(df['day'], bins=day_bins, labels=day_labels)

    duration_bins = [-float('inf'), 120, 600, float('inf')]  # Binning duration as ≤120, 121-600, >600
    duration_labels = ['≤120', '121-600', '>600']
    df['duration'] = pd.cut(df['duration'], bins=duration_bins, labels=duration_labels)

    campaign_bins = [-float('inf'), 1, 5, float('inf')]  # Binning campaign as ≤1, 2-5, >5
    campaign_labels = ['≤1', '2-5', '>5']
    df['campaign'] = pd.cut(df['campaign'], bins=campaign_bins, labels=campaign_labels)

    pdays_bins = [-float('inf'), 30, 180, float('inf')]  # Binning pdays as ≤30, 31-180, >180
    pdays_labels = ['≤30', '31-180', '>180']
    df['pdays'] = pd.cut(df['pdays'], bins=pdays_bins, labels=pdays_labels)

    previous_bins = [-float('inf'), 0, 5, float('inf')]  # Binning previous as 0, 1-5, >5
    previous_labels = ['0', '1-5', '>5']
    df['previous'] = pd.cut(df['previous'], bins=previous_bins, labels=previous_labels)

    save_path = path.join(PROJECT_ROOT, 'processed_data/bank_marketing/bank_processed.csv')
    df.to_csv(save_path)



