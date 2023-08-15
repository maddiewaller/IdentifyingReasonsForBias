import pandas as pd
from os import path
import pickle5 as pickle
from project.paths import PROJECT_ROOT
from sklearn.metrics import confusion_matrix, classification_report


def prevalence():
    data_path = path.join(PROJECT_ROOT, 'processed_data/bank_marketing/bank_processed.csv')
    df = pd.read_csv(data_path)

    categories = ["MidAge", "YoungOrOld", "married", "single", "divorced"]
    results = {}

    for category in categories:
        if category == "YoungOrOld" or category == "MidAge":
            neg_count = len(df[(df['age'] == category) & (df['y'] == 'yes')])
            pos_count = len(df[(df['age'] == category) & (df['y'] == 'no')])
        else:
            neg_count = len(df[(df['marital'] == category) & (df['y'] == 'yes')])
            pos_count = len(df[(df['marital'] == category) & (df['y'] == 'no')])
        total_samples = neg_count + pos_count
        neg_percentage = (neg_count / total_samples) * 100
        results[category] = {'Negative': neg_count, 'Positive': pos_count, 'Negative Percentage': neg_percentage}

    # Specify the file path to save the output
    output_path = path.join(PROJECT_ROOT, 'results/bank_marketing/bank_prevalence.txt')

    with open(output_path, 'w') as f:
        f.write("Prevalence of individuals in each protected group in the Bank Marketing dataset\n")
        for category, counts in results.items():
            f.write(
                f"{category}: Negative: {counts['Negative']}, Positive: {counts['Positive']}, Negative Percentage: {counts['Negative Percentage']:.2f}%\n")



def performance():
    # load test data
    data_path = path.join(PROJECT_ROOT, 'processed_data/bank_marketing/test_data_encode.csv')
    df_test = pd.read_csv(data_path, index_col=0)

    X_test, y_test = df_test.loc[:, df_test.columns != 'y'], df_test['y']

    # load classifier
    classifier_path = path.join(PROJECT_ROOT, 'classifiers/bank_clf')
    clf = pickle.load(open(classifier_path, 'rb'))

    # get predictions for test data
    y_pred = clf.predict(X_test)

    output_file_path = path.join(PROJECT_ROOT, 'results/bank_marketing/bank_clf_performance.txt')
    # open the file for writing
    with open(output_file_path, 'w') as f:
        # Original confusion matrix and classification report
        cnf_matrix = confusion_matrix(y_test, y_pred)
        f.write("Original Confusion Matrix:\n")
        f.write(str(cnf_matrix) + "\n")

        f.write("Original Classification Report:\n")
        f.write(classification_report(y_test, y_pred) + "\n")

