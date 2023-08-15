import copy
import pandas as pd
import numpy as np
from os import path
from project.paths import PROJECT_ROOT


def load_columns():
    """
    Loads the column names
    @return: column names
    """
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
               'native-country', 'Probability']
    return columns


def load_data():
    """
    Loads the original full dataset
    @return: the training and test datasets combined
    """
    # read original dataset, ? values treated as NULL, with column names defined
    colnames = load_columns()

    train_path = path.join(PROJECT_ROOT, 'raw_data/adult_census/adult.data.csv')
    test_path = path.join(PROJECT_ROOT, 'raw_data/adult_census/adult.test.csv')
    df_train = pd.read_csv(train_path, na_values=' ?', names=colnames, header=None)
    df_test = pd.read_csv(test_path, na_values=' ?', names=colnames, header=None).tail(-1)
    return pd.concat([df_train, df_test], axis=0)


def preprocess():
    df_orig = load_data()

    # Drop NULL values
    df_orig = df_orig.dropna(how='any', axis=0)

    df = copy.deepcopy(df_orig)

    # Remove . from Probability
    df['Probability'] = df['Probability'].str.replace(' <=50K.', ' <=50K', regex=False)
    df['Probability'] = df['Probability'].str.replace(' >50K.', ' >50K', regex=False)

    # Drop attributes fnlwgt and education-num
    df.drop('fnlwgt', axis=1, inplace=True)
    df.drop('education-num', axis=1, inplace=True)

    df['age'] = df['age'].astype(int)
    df['hours-per-week'] = df['hours-per-week'].astype(int)
    df['capital-gain'] = df['capital-gain'].astype(float)
    df['capital-loss'] = df['capital-loss'].astype(float)

    # Categorize the numerical columns in the same way as
    # Le Quy et. al (2021) A survey on datasets for fairness-aware machine learning
    age_bins = [0, 25, 60, float('inf')]  # Binning ages as <25, 25-60, >60
    age_labels = ['<25', '25-60', '>60']
    df['age'] = pd.cut(df['age'], bins=age_bins, labels=age_labels)

    capital_gain_bins = [-float('inf'), 5000, float('inf')]  # Binning capital gains as ≤5000, >5000
    capital_gain_labels = ['≤5000', '>5000']
    df['capital-gain'] = pd.cut(df['capital-gain'], bins=capital_gain_bins, labels=capital_gain_labels)

    capital_loss_bins = [-float('inf'), 40, float('inf')]  # Binning capital losses as ≤40, >40
    capital_loss_labels = ['≤40', '>40']
    df['capital-loss'] = pd.cut(df['capital-loss'], bins=capital_loss_bins, labels=capital_loss_labels)

    hours_per_week_bins = [-float('inf'), 40, 60, float('inf')]  # Binning hours per week as <40, 40-60, >60
    hours_per_week_labels = ['<40', '40-60', '>60']
    df['hours-per-week'] = pd.cut(df['hours-per-week'], bins=hours_per_week_bins, labels=hours_per_week_labels)

    save_path = path.join(PROJECT_ROOT, 'processed_data/adult_census/adult_processed.csv')
    df.to_csv(save_path)
