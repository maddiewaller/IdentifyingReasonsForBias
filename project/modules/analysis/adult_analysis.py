import pandas as pd
from os import path
import pickle5 as pickle
from sklearn.preprocessing import OneHotEncoder
from project.paths import PROJECT_ROOT
from sklearn.metrics import confusion_matrix, classification_report


def encode(df):
    cat_columns = ['age', 'workclass', 'education', 'marital-status', 'occupation',
               'relationship', 'race', 'sex', 'hours-per-week',
               'native-country']
    # Use pd.get_dummies() to create dummy variables
    dummies_df = pd.get_dummies(df, drop_first=True, columns=cat_columns)
    return dummies_df

def prevalence():
    data_path = path.join(PROJECT_ROOT, 'processed_data/adult_census/adult_processed.csv')
    df = pd.read_csv(data_path)

    male_neg = df[(df['sex'] == ' Male') & (df['Probability'] == ' <=50K')]
    male_pos = df[(df['sex'] == ' Male') & (df['Probability'] == ' >50K')]
    female_neg = df[(df['sex'] == ' Female') & (df['Probability'] == ' <=50K')]
    female_pos = df[(df['sex'] == ' Female') & (df['Probability'] == ' >50K')]

    white_neg = df[(df['race'] == ' White') & (df['Probability'] == ' <=50K')]
    white_pos = df[(df['race'] == ' White') & (df['Probability'] == ' >50K')]
    black_neg = df[(df['race'] == ' Black') & (df['Probability'] == ' <=50K')]
    black_pos = df[(df['race'] == ' Black') & (df['Probability'] == ' >50K')]
    asian_neg = df[(df['race'] == ' Asian-Pac-Islander') & (df['Probability'] == ' <=50K')]
    asian_pos = df[(df['race'] == ' Asian-Pac-Islander') & (df['Probability'] == ' >50K')]
    amer_neg = df[(df['race'] == ' Amer-Indian-Eskimo') & (df['Probability'] == ' <=50K')]
    amer_pos = df[(df['race'] == ' Amer-Indian-Eskimo') & (df['Probability'] == ' >50K')]
    other_neg = df[(df['race'] == ' Other') & (df['Probability'] == ' <=50K')]
    other_pos = df[(df['race'] == ' Other') & (df['Probability'] == ' >50K')]

    categories = ["male", "female", "white", "black", "asian", "amer", "other"]

    # Create lists of negative and positive samples for each category
    neg_samples = [male_neg, female_neg, white_neg, black_neg, asian_neg, amer_neg, other_neg]
    pos_samples = [male_pos, female_pos, white_pos, black_pos, asian_pos, amer_pos, other_pos]

    # Specify the file path to save the output
    output_path = path.join(PROJECT_ROOT, 'results/adult_census/adult_prevalence.txt')

    with open(output_path, 'w') as f:
        f.write("Prevalence of individuals in each protected group in the Adult Census dataset\n")
        # Loop through the categories and their corresponding negative and positive sample lists
        for category, neg_count, pos_count in zip(categories, neg_samples, pos_samples):
            total_samples = len(neg_count) + len(pos_count)
            neg_percentage = (len(neg_count) / total_samples) * 100
            # Write the category label and counts to the file
            f.write(f"{category}: Negative: {len(neg_count)}, Positive: {len(pos_count)}, Negative Percentage: {neg_percentage:.2f}%\n")


def performance():
    # load test data
    data_path = path.join(PROJECT_ROOT, 'processed_data/adult_census/test_data_encode.csv')
    df_test = pd.read_csv(data_path, index_col=0)

    X_test, y_test = df_test.loc[:, df_test.columns != 'Probability'], df_test['Probability']

    # load classifier
    classifier_path = path.join(PROJECT_ROOT, 'classifiers/adult_clf')
    clf = pickle.load(open(classifier_path, 'rb'))

    # get predictions for test data
    y_pred = clf.predict(X_test)

    output_file_path = path.join(PROJECT_ROOT, 'results/adult_census/adult_clf_performance.txt')
    # open the file for writing
    with open(output_file_path, 'w') as f:
        # Original confusion matrix and classification report
        cnf_matrix = confusion_matrix(y_test, y_pred)
        f.write("Original Confusion Matrix:\n")
        f.write(str(cnf_matrix) + "\n")

        f.write("Original Classification Report:\n")
        f.write(classification_report(y_test, y_pred) + "\n")

