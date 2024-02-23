import csv
import os

import matplotlib.pyplot as plt
import enum
import numpy as np
import pandas as pd
from numba.tests.test_extending import sc
from numpy.random import RandomState
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import minkowski_distance, distance
import scipy as scip
from scipy import stats


# creating enumerations using class
class DistanceType(enum.Enum):
    minkowski = 1
    euclidean = 2


def compute_distance(a, b, distance_type: DistanceType):
    if distance_type == DistanceType.minkowski:
        return minkowski_distance(a, b, p=2)
    elif distance_type == DistanceType.euclidean:
        return distance.euclidean(a, b)


def compute_intersection(d, r):
    # references:
    # - https://is.gd/SdhyOj
    # - https://is.gd/zUT0Ij
    # - https://is.gd/INMFhC
    # - calcoli fatti da me
    # Per vedere l'intersezione, basta andare a vedere la distanza tra i due centri dei detector e confrontarla
    # con la somma dei due raggi:
    #   if d <= r1-r2:
    #       print("C2  is in C1")
    #   elif d <= r2-r1:
    #       print("C1  is in C2")
    #   elif d < r1+r2:
    #       print("Circumference of C1  and C2  intersect")
    #   elif d == r1+r2:
    #       print("Circumference of C1 and C2 will touch")
    #   else:
    #       print("C1 and C2  do not overlap")
    if d == 0:  # (d == 0 and r1==r2) or d < abs(r1-r1)
        return 1
    if d >= 2 * r:
        return 0
    if d < 2 * r:
        return abs(1 - (d / (2 * r)))  # non so se funziona anche con due raggi diversi: abs(1 - (d / (r1 + r2)))


def mkdir(path):
    """ Create a folder with os.mkdir only if it doesn't exist (in this way it's not needed to check the error). """
    if not os.path.isdir(path):
        os.makedirs(path)


def calculate_accuracy(y_test, y_pred):
    return accuracy_score(y_pred, y_test)


def calculate_confusion_matrix(y_test, y_pred):
    return confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(conf_matrix, output_folder):
    fig, ax = plt.subplots(figsize=(7.2, 6.5))
    ax.matshow(conf_matrix, cmap=plt.get_cmap("Blues"), alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i, s=conf_matrix[i, j], va='center', ha='center', size='x-large')
    labels = ['0\nNotSelf', '1\nSelf']
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig(os.path.join(output_folder, 'confusion_matrix.png'))
    plt.close()


def normalize_dataset(dataset, export=False, output_path="normalized_output.csv"):
    scaler = MinMaxScaler()

    columns = list(dataset.columns)
    dataset[columns] = scaler.fit_transform(dataset)

    if export:
        dataset.to_csv(output_path, index=False)

    return dataset


def create_train_test_database(input_folder, dataset_filename, shuffle=False, shuffle_seed=None, percentage=0.1,
                               show_plot=False, normalize=True):
    input_path = os.path.join(input_folder, dataset_filename)
    output_filename = os.path.splitext(dataset_filename)[0] + '_normalized.csv'
    output_path = os.path.join(input_folder, output_filename)

    if normalize and os.path.exists(output_path):
        # if a normalized dataset with the same name created by this application already exist, then that is the one
        # to load:
        input_path = output_path

    dataset = pd.read_csv(input_path)
    columns, dataset = move_column_on_first_position(dataset)

    if normalize and not os.path.exists(output_path):
        dataset = normalize_dataset(dataset, export=True, output_path=output_path)

    dt_self = dataset[dataset['target'] == 1]
    dt_self = dt_self.drop(['target'], axis=1)
    dt_not_self = dataset[dataset['target'] == 0]
    dt_not_self = dt_not_self.drop(['target'], axis=1)
    dt_all = dataset.drop(['target'], axis=1)

    euclidean_distance_all, euclidean_distance_self, minkowski_distance_all, minkowski_distance_not_self, \
        minkowski_distance_self = execute_measurements(dt_all, dt_not_self, dt_self)

    if show_plot:
        show_plots(euclidean_distance_all, euclidean_distance_self, minkowski_distance_all, minkowski_distance_not_self,
                   minkowski_distance_self)

    if shuffle:
        # Shuffle the dataset
        rng = RandomState(shuffle_seed)
        X_shuffled = dataset
        X_shuffled = X_shuffled.sample(frac=1.0, random_state=rng)
        X_shuffled = X_shuffled.loc[X_shuffled.index.isin(X_shuffled.index)]
        X_shuffled = X_shuffled.sort_values(by='target', ascending=False)
        # Prepare Train and Test
        X_train_self = X_shuffled.loc[dataset['target'] == 1]
        n = np.size(X_train_self, axis=0) - int(np.size(X_train_self, axis=0) * percentage)
        # X_train_self = X_train_self.iloc[0:np.size(X_train_self, axis=0) -
        # (int)(np.size(X_train_self, axis=0) * percentage), :]
        X_train_self = X_train_self.iloc[0:n, :]
        X_train = X_train_self.iloc[:, 1:]
        X_train = X_train.to_numpy()
        y_train = X_train_self.iloc[:, 0]
        y_train = y_train.to_numpy()
        # X_test_self = X_shuffled.iloc[np.size(X_train_self, axis=0) -
        # (int)(np.size(X_train_self, axis=0) * percentage):, :]
        X_test_self = X_shuffled.iloc[n:, :]
        X_test = X_test_self.iloc[:, 1:]
        X_test = X_test.to_numpy()
        y_test = X_test_self.iloc[:, 0]
        y_test = y_test.to_numpy()
    else:
        X_train_self = dataset.loc[dataset['target'] == 1]
        X_train_self = X_train_self.iloc[0:np.size(X_train_self, axis=0) - int(np.size(X_train_self, axis=0) * 0.1), :]
        X_train = X_train_self.iloc[:, 1:]
        X_train = X_train.to_numpy()
        y_train = X_train_self.iloc[:, 0]
        y_train = y_train.to_numpy()
        # X_test_self = dataset.loc[dataset['target'] == 0]
        # X_test_self = dataset
        X_test_self = dataset.iloc[np.size(X_train_self, axis=0) - int(np.size(X_train_self, axis=0) * 0.1):, :]
        X_test = X_test_self.iloc[:, 1:]
        X_test = X_test.to_numpy()
        y_test = X_test_self.iloc[:, 0]
        y_test = y_test.to_numpy()

    return columns, X_train, X_test, y_train, y_test


def move_column_on_first_position(dataset, column_name='target'):
    """ Given a column name in input, move that column in the first position of a panda DataFrame. If it's already
    there, this doesn't do anything. Return the DataFrame and a list of its column. """
    if dataset.columns.get_loc(column_name) != 0:
        first_column = dataset[column_name]
        dataset = dataset.drop([column_name], axis=1)
        dataset.insert(0, column_name, first_column)
    columns = list(dataset.columns)
    return columns, dataset


def execute_measurements(dt_all, dt_not_self, dt_self, verbose=False):
    euclidean_distance_self = cdist(dt_self, dt_self, metric='euclidean')
    minkowski_distance_self = cdist(dt_self, dt_self, metric='minkowski', p=1.0)
    euclidean_distance_not_self = cdist(dt_not_self, dt_not_self, metric='euclidean')
    minkowski_distance_not_self = cdist(dt_not_self, dt_not_self, metric='minkowski', p=1.0)
    euclidean_distance_all = cdist(dt_all, dt_all, metric='euclidean')
    minkowski_distance_all = cdist(dt_all, dt_all, metric='minkowski', p=1.0)
    max_euclidean_distance_self = euclidean_distance_self.max()
    min_euclidean_distance_self = euclidean_distance_self[euclidean_distance_self > 0.0].min()
    avg_euclidean_distance_self = euclidean_distance_self.sum() / np.size(euclidean_distance_self)
    max_minkowski_distance_self = minkowski_distance_self.max()
    min_minkowski_distance_self = minkowski_distance_self[minkowski_distance_self > 0.0].min()
    avg_minkowski_distance_self = minkowski_distance_self.sum() / np.size(euclidean_distance_self)
    max_euclidean_distance_not_self = euclidean_distance_not_self.max()
    min_euclidean_distance_not_self = euclidean_distance_not_self[euclidean_distance_not_self > 0.0].min()
    avg_euclidean_distance_not_self = euclidean_distance_self.sum() / np.size(euclidean_distance_not_self)
    max_minkowski_distance_not_self = minkowski_distance_not_self.max()
    min_minkowski_distance_not_self = minkowski_distance_not_self[minkowski_distance_not_self > 0.0].min()
    avg_minkowski_distance_not_self = minkowski_distance_not_self.sum() / np.size(euclidean_distance_not_self)
    max_euclidean_distance_all = euclidean_distance_all.max()
    min_euclidean_distance_all = euclidean_distance_all[euclidean_distance_all > 0.0].min()
    avg_euclidean_distance_all = euclidean_distance_all.sum() / np.size(euclidean_distance_all)
    max_minkowski_distance_all = minkowski_distance_all.max()
    min_minkowski_distance_all = minkowski_distance_all[minkowski_distance_all > 0.0].min()
    avg_minkowski_distance_all = minkowski_distance_all.sum() / np.size(euclidean_distance_all)

    if verbose:
        print('max_euclidean_distance_self = ', max_euclidean_distance_self)
        print('min_euclidean_distance_self = ', min_euclidean_distance_self)
        print('avg_euclidean_distance_self = ', avg_euclidean_distance_self)
        print('max_euclidean_distance_not_self = ', max_euclidean_distance_not_self)
        print('min_euclidean_distance_not_self = ', min_euclidean_distance_not_self)
        print('avg_euclidean_distance_not_self = ', avg_euclidean_distance_not_self)
        print('max_euclidean_distance_all = ', max_euclidean_distance_all)
        print('min_euclidean_distance_all = ', min_euclidean_distance_all)
        print('avg_euclidean_distance_all = ', avg_euclidean_distance_all)
        print('max_minkowski_distance_self = ', max_minkowski_distance_self)
        print('min_minkowski_distance_self = ', min_minkowski_distance_self)
        print('avg_minkowski_distance_self = ', avg_minkowski_distance_self)
        print('max_minkowski_distance_not_self = ', max_minkowski_distance_not_self)
        print('min_minkowski_distance_not_self = ', min_minkowski_distance_not_self)
        print('avg_minkowski_distance_not_self = ', avg_minkowski_distance_not_self)
        print('max_minkowski_distance_all = ', max_minkowski_distance_all)
        print('min_minkowski_distance_all = ', min_minkowski_distance_all)
        print('avg_minkowski_distance_all = ', avg_minkowski_distance_all)
    return euclidean_distance_all, euclidean_distance_self, minkowski_distance_all, minkowski_distance_not_self, \
        minkowski_distance_self


def show_plots(euclidean_distance_all, euclidean_distance_self, minkowski_distance_all, minkowski_distance_not_self,
               minkowski_distance_self):
    plt.hist(euclidean_distance_self[euclidean_distance_self > 0.0], density=True, bins=30,
             label='Self')  # density=False would make counts
    # plt.hist(euclidean_distance_not_self[euclidean_distance_not_self > 0.0],
    #          density=True, bins=30, label='not Self')  # density=False would make counts
    plt.ylabel('Density')
    plt.xlabel('Euclidean Distance')
    plt.legend()
    plt.show()
    plt.hist(minkowski_distance_self[minkowski_distance_self > 0.0], density=True, bins=30,
             label='Self')  # density=False would make counts
    plt.hist(minkowski_distance_not_self[minkowski_distance_not_self > 0.0], density=True, bins=30,
             label='not Self', alpha=.4)  # density=False would make counts
    plt.ylabel('Density')
    plt.xlabel('Minkowski Distance')
    plt.legend()
    plt.show()
    plt.hist(euclidean_distance_all[euclidean_distance_all > 0.0], density=True, bins=30,
             label='All')  # density=False would make counts
    plt.ylabel('Density')
    plt.xlabel('Euclidean Distance All')
    plt.legend()
    plt.show()
    plt.hist(minkowski_distance_all[minkowski_distance_all > 0.0], density=True, bins=30,
             label='All')  # density=False would make counts
    plt.ylabel('Density')
    plt.xlabel('Minkowski Distance')
    plt.legend()
    plt.show()


def create_prediction_csv(columns, patient, not_patient, output_folder=""):
    """ This will write on disk a new csv file containing the prediction on the test set. The csv will be the same as
    the test set with two more columns: distance (will tell you the distance from the detector which capture the sample
    in case the prediction is 'self' or the distance from the closer detector otherwise) and target
    (the actual prediction self/non-self indicated with 0/1). """
    with open(os.path.join(output_folder, 'final_classification.csv'), 'w', newline='') as file:
        writer = csv.writer(file)
        new_columns = ['distance'] + columns
        rows = [new_columns]
        rows = rows + patient + not_patient
        writer.writerows(rows)


def test_null_hypothesis(first_distribution, second_distribution, alpha=0.05):
    """ Test that the first distribution is the same as the second distribution (Null Hypothesis)
        :param first_distribution: N-d arrays of samples
        :param second_distribution: N-d arrays of samples
        :param alpha: The probability of making the wrong decision when the null hypothesis is true

        :return support: True if the Null Hypothesis is supported, False otherwise
        :return statistic: statistic corresponding with the first distribution
        :return p_value: The associated p-value for the chosen alternative.
    """
    statistic, p_value = scip.stats.mannwhitneyu(first_distribution, second_distribution, alternative='two-sided')
    return p_value > alpha, statistic, p_value
