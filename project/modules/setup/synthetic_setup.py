from sklearn.neighbors import NearestNeighbors

from project.paths import PROJECT_ROOT
from os import path, remove, makedirs
import pandas as pd
import numpy as np
import glob


def encode(df):
    cat_columns = ['age', 'workclass', 'education', 'marital-status', 'occupation',
                   'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                   'native-country']

    # Use pd.get_dummies() to create dummy variables
    dummies_df = pd.get_dummies(df, drop_first=True, columns=cat_columns)
    return dummies_df


# this is the same as written in the paper (Hamming distance for categorical features)
def define_similar(test_data, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='hamming').fit(test_data.values)
    return nbrs


def set_bias():
    # get adult_processed.csv data
    data_path = path.join(PROJECT_ROOT, 'processed_data/adult_census/test_data.csv')
    df = pd.read_csv(data_path, index_col=0)

    # add bias-attr column with random values based on the probability of 0.5
    random_bias = np.random.choice([0, 1], size=len(df), p=[0.5, 0.5])
    df.insert(1, 'bias-attr', random_bias)

    # set the 'Probability' column values based on the random bias values
    df.loc[df['bias-attr'] == 0, 'Probability'] = ' <=50K'
    df.loc[df['bias-attr'] == 1, 'Probability'] = ' >50K'

    return df


def find_similar_inds(k):
    # load test_data in original form
    test_data = set_bias()

    # clear the save directory of any previous results
    save_path = path.join(PROJECT_ROOT, f'processed_data/synthetic/synthetic_sim_inds_k{k}')
    files_in_directory = glob.glob(path.join(save_path, "*"))
    for file in files_in_directory:
        remove(file)

    test_data_encode = encode(test_data)
    test_data_encode = test_data_encode.loc[:, test_data_encode.columns != 'Probability']

    # for each queried individual in the test data
    for index, row in test_data.iterrows():
        # if they are negatively classified (i.e. predicted to earn less than 50k)
        if row['Probability'] == " <=50K":
            q_individual = test_data_encode.loc[[index]]

            # cluster individuals in test_data without the queried individual
            test_data_copy = test_data_encode.drop(index)
            nbrs = define_similar(test_data_copy, k)

            # find the k most similar individuals to the queried individual
            distances, knn_list = nbrs.kneighbors(q_individual.values)
            neighbours = test_data.iloc[knn_list[0]]

            # find the queried individual in the original dataframe
            individual = test_data.loc[[index]]
            # combine the queried individual and their 5 most similar neighbours
            combined = pd.concat([individual, neighbours])

            # save the combined individuals to csv files in the processed_data folder
            save_path = path.join(PROJECT_ROOT, f'processed_data/synthetic/synthetic_sim_inds_k{k}')
            makedirs(save_path, exist_ok=True)

            # define a filename based on the index
            filename = path.join(save_path, f"synthetic_inds_{index}.csv")

            # save the combined DataFrame to a CSV file
            combined.to_csv(filename, index=False)

