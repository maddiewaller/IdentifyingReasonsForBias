'''
This file contains the code to replicate the experiments in the paper.
Can choose between the following experiments:
    - Replicating the simple example
    - Running experiments on the Adult Census dataset
    - Running experiments on the Bank Marketing dataset
    - Running experiments on the Adult Census dataset with synthetic bias
'''


from os import path, remove
from paths import PROJECT_ROOT
from modules.graph_construction.construct import construct_graph
import csv
import glob
import pandas as pd

import modules.preprocessing.adult_pre as adult_pre
import modules.analysis.adult_analysis as adult_analysis
import modules.setup.adult_setup as adult_setup

import modules.preprocessing.bank_pre as bank_pre
import modules.analysis.bank_analysis as bank_analysis
import modules.setup.bank_setup as bank_setup

import modules.setup.synthetic_setup as synthetic_setup
from modules.graph_construction.construct import construct_graph


def append_to_file(string, filename):
    # if file doesn't exist, create it
    if not path.exists(filename):
        with open(filename, "w") as f:
            f.write(string + "\n")
    else:
        with open(filename, "a") as f:
            f.write(string + "\n")


def simple_example():
    ''' Replicates example found in the paper'''
    filename = path.join(PROJECT_ROOT, 'raw_data/toy_data/example_data1.csv')
    final_weights, weakest_args = construct_graph(filename)
    save_path = path.join(PROJECT_ROOT, 'results/example1.csv')
    # save the final weights and weakest arguments to csv file
    with open(save_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Argument', ' Weight'])  # Writing header for final_weights
        for arg, weight in final_weights.items():
            writer.writerow([arg, weight])
    with open(save_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])  # Empty row
        writer.writerow(['Weakest arguments'])  # Writing header for weakest_args
        writer.writerow(weakest_args)


def adult_experiments():
    '''
    Replicates experiments on the Adult census dataset
    Delete the following steps as desired, all preproessing has been stored
    so only necessary to run graph construction code to replicate experiments
    '''
    # preprocess the data
    #adult_pre.preprocess()

    # analyse the prevalence of the sensitive attribute
    #adult_analysis.prevalence()

    # train the classifier
    #adult_setup.train()

    # analyse the performance of the classifier
    #adult_analysis.performance()

    # find similar individuals (specifying the number of similar individuals to consider)
    k = 5
    #adult_setup.find_similar_inds(k)

    # construct the graph for each queried individual
    count = 0
    count_consistent = 0
    weakest_counts = {}
    for filename in glob.glob(path.join(PROJECT_ROOT, 'processed_data/adult_census/adult_sim_inds/*.csv')):
        final_weights, weakest = construct_graph(filename)

        inds = pd.read_csv(filename)
        unique_values_count = inds['Probability'].nunique()
        if unique_values_count == 1:
            count_consistent = count_consistent + 1

        # count how many times different arguments were the weakest
        count += 1
        for arg in weakest:
            if arg in weakest_counts:
                weakest_counts[arg] += 1
            else:
                weakest_counts[arg] = 1

    filename = 'results/adult_census/adult_results_k' + str(k) + '.csv'
    filepath = path.join(PROJECT_ROOT, filename)
    # delete file if already exists
    if path.exists(filepath):
        remove(filepath)
    append_to_file("Experiment: Adult Census", filepath)
    append_to_file("Number of similar individuals: " + str(k), filepath)
    append_to_file("", filepath)

    # saves the number of times each sensitive attribute value was the weakest or was consistent
    append_to_file("Arguments representing sensitive attributes:", filepath)
    for key, value in weakest_counts.items():
        if "sex" in key or "race" in key or "consistent" in key:
            percentage = (value / count) * 100
            formatted_percentage = "{:.2f}".format(percentage)
            string = key + " count: " + str(value) + " percentage: " + formatted_percentage
            append_to_file(string, filepath)

    # save all weakest arguments and their counts
    append_to_file("", filepath)
    append_to_file("All weakest arguments and their counts :", filepath)
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Arguments', ' Weakest count'])
        for key, value in weakest_counts.items():
            writer.writerow([key, value])


def bank_experiments():
    '''
    Replicates experiments on the Bank Marketing census dataset
    Delete the following steps as desired, all preprocessing has been stored
    so only necessary to run graph construction code to replicate experiments
    '''
    # preprocess the data
    #bank_pre.preprocess()

    # analyse the prevalence of the sensitive attribute
    #bank_analysis.prevalence()

    # train the classifier
    #bank_setup.train()

    # analyse the performance of the classifier
    #bank_analysis.performance()

    # find similar individuals (specifying the number of similar individuals to consider)
    k = 5
    #bank_setup.find_similar_inds(k)

    # construct the graph for each queried individual
    count = 0
    weakest_counts = {}
    for filename in glob.glob(path.join(PROJECT_ROOT, 'processed_data/bank_marketing/bank_sim_inds/*.csv')):
        final_weights, weakest = construct_graph(filename)
        # count how many times different arguments were the weakest
        count += 1
        for arg in weakest:
            if arg in weakest_counts:
                weakest_counts[arg] += 1
            else:
                weakest_counts[arg] = 1

    # add ordered to csv file
    filename = 'results/bank_marketing/bank_results_k' + str(k) + '.csv'
    filepath = path.join(PROJECT_ROOT, filename)
    # delete file if already exists
    if path.exists(filepath):
        remove(filepath)
    append_to_file("Experiment: Bank Marketing", filepath)
    append_to_file("Number of similar individuals: " + str(k), filepath)
    append_to_file("", filepath)

    # saves the number of times each sensitive attribute value was the weakest or was consistent
    append_to_file("Arguments representing sensitive attributes:", filepath)
    for key, value in weakest_counts.items():
        if "age=" in key or "marital" in key or "consistent" in key:
            percentage = (value / count) * 100
            formatted_percentage = "{:.2f}".format(percentage)
            string = key + " count: " + str(value) + " percentage: " + formatted_percentage
            append_to_file(string, filepath)

    # save all weakest arguments and their counts
    append_to_file("", filepath)
    append_to_file("All weakest arguments and their counts :", filepath)
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Arguments', ' Weakest count'])
        for key, value in weakest_counts.items():
            writer.writerow([key, value])


def synthetic_experiments():
    # preprocess the adult census dataset
    adult_pre.preprocess()

    # find the similar individuals (this also adds the synthetic bias to the processed adult dataset)
    k = 15
    synthetic_setup.find_similar_inds(k)

    # construct the graph for each queried individual
    count = 0
    weakest_counts = {}
    for filename in glob.glob(path.join(PROJECT_ROOT, f'processed_data/synthetic/synthetic_sim_inds_k{k}', '*.csv')):
        final_weights, weakest = construct_graph(filename)
        # count how many times different arguments were the weakest
        count += 1
        for arg in weakest:
            if arg in weakest_counts:
                weakest_counts[arg] += 1
            else:
                weakest_counts[arg] = 1

    # add to csv file
    filename = f'results/synthetic/synthetic_results_k{k}.csv'
    filepath = path.join(PROJECT_ROOT, filename)
    # delete file if already exists
    if path.exists(filepath):
        remove(filepath)
    append_to_file("Experiment: Adult Census with added biased attribute", filepath)
    append_to_file("Number of similar individuals: " + str(k), filepath)
    append_to_file("", filepath)

    # saves the number of times each sensitive attribute value was the weakest or was consistent
    append_to_file("Arguments representing intentionally biased attribute:", filepath)
    for key, value in weakest_counts.items():
        if "bias-attr" in key or "consistent" in key:
            percentage = (value / count) * 100
            formatted_percentage = "{:.2f}".format(percentage)
            string = key + " count: " + str(value) + " percentage: " + formatted_percentage
            append_to_file(string, filepath)

    # save all weakest arguments and their counts
    append_to_file("", filepath)
    append_to_file("All weakest arguments and their counts:", filepath)
    with open(filepath, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Arguments', ' Weakest count'])
        for key, value in weakest_counts.items():
            writer.writerow([key, value])



def experiments(choice):
    if choice == "example":
        simple_example()
    elif choice == "adult":
        adult_experiments()
    elif choice == "bank":
        bank_experiments()
    elif choice == "synthetic":
        synthetic_experiments()
    else:
        print("Invalid choice")


def main():
    choice = input("Enter experiment to run: ")
    experiments(choice)

main()


