import pandas as pd
from os import path, makedirs, remove
from sklearn.neighbors import NearestNeighbors, VALID_METRICS
import glob
from project.paths import PROJECT_ROOT
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle5 as pickle


def encode(df):
    cat_columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
                   'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome']
    # Use pd.get_dummies() to create dummy variables
    dummies_df = pd.get_dummies(df, drop_first=True, columns=cat_columns)
    return dummies_df


def train():
    data_path = path.join(PROJECT_ROOT, 'processed_data/bank_marketing/bank_processed.csv')
    df = pd.read_csv(data_path, index_col=0)

    # encode data for training and testing
    df_encode = encode(df)
    # 80% training, 20% test, no randomness
    df_train_encode, df_test_encode = train_test_split(df_encode, test_size=0.2, shuffle=False, stratify=None)
    X_train, y_train = df_train_encode.loc[:, df_train_encode.columns != 'y'], df_train_encode['y']

    # train a logistic regression classifier
    clf = LogisticRegression(max_iter=100000)
    clf.fit(X_train, y_train)
    print("TRAINED!")

    # save classifier (so setup only needs to be run once)
    classifier_path = path.join(PROJECT_ROOT, 'classifiers/bank_clf')
    pickle.dump(clf, open(classifier_path, 'wb'))

    # save encoded test data for analysing classifier performance
    save_path = path.join(PROJECT_ROOT, 'processed_data/bank_marketing/test_data_encode.csv')
    df_test_encode.to_csv(save_path)

    # save test data from original dataset to be used for experiments (identifying reasons for bias)
    df_train, df_test = train_test_split(df, test_size=0.2, shuffle=False, stratify=None)
    save_path = path.join(PROJECT_ROOT, 'processed_data/bank_marketing/test_data.csv')
    df_test.to_csv(save_path)


# this is the same as written in the paper (Hamming distance for categorical features)
def define_similar(X_test, k):
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree', metric='hamming').fit(X_test.values)
    return nbrs


def find_similar_inds(k):
    # load test_data in original form
    data_path = path.join(PROJECT_ROOT, 'processed_data/bank_marketing/test_data.csv')
    test_data = pd.read_csv(data_path, index_col=0)

    # load X_test_encode in encoded form for prediction
    data_path = path.join(PROJECT_ROOT, 'processed_data/bank_marketing/test_data_encode.csv')
    test_data_encode = pd.read_csv(data_path, index_col=0)
    X_test_encode = test_data_encode.loc[:, test_data_encode.columns != 'y']

    # load the classifier
    classifier_path = path.join(PROJECT_ROOT, 'classifiers/bank_clf')
    clf = pickle.load(open(classifier_path, 'rb'))
    y_pred = clf.predict(X_test_encode)
    # replace label with y_pred in test data
    test_data['y'] = y_pred

    # clear the save directory of any previous results
    save_path = path.join(PROJECT_ROOT, 'processed_data/bank_marketing/bank_sim_inds')
    files_in_directory = glob.glob(path.join(save_path, "*"))
    for file in files_in_directory:
        remove(file)

    # for each queried individual in X_test_encode
    for index, row in test_data.iterrows():
        # if they are negatively classified (i.e. predicted to earn less than 50k)
        if row['y'] == "yes":
            q_individual = X_test_encode.loc[[index]]

            # cluster individuals in X_test without the queried individual
            X_copy = X_test_encode.drop(index)
            nbrs = define_similar(X_copy, k)

            # find the 5 most similar individuals to the queried individual
            distances, knn_list = nbrs.kneighbors(q_individual.values)
            neighbour_indices = X_test_encode.index[knn_list[0]]
            neighbours = test_data.loc[neighbour_indices]

            # find the queried individual in the original dataframe
            individual = test_data.loc[[index]]
            # combine the queried individual and their 5 most similar neighbours
            combined = pd.concat([individual, neighbours])

            # save the combined individuals to csv files in the processed_data folder
            save_path = path.join(PROJECT_ROOT, 'processed_data/bank_marketing/bank_sim_inds')
            makedirs(save_path, exist_ok=True)

            # define a filename based on the index
            filename = path.join(save_path, f"bank_inds_{index}.csv")

            # save the combined DataFrame to a CSV file
            combined.to_csv(filename, index=False)
