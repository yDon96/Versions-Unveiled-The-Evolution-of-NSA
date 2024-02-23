import os
import sys
import warnings
from datetime import datetime

import numpy as np
from numba import NumbaWarning, NumbaDeprecationWarning, NumbaPendingDeprecationWarning
from sklearn.metrics import precision_score, recall_score

from config.Config import Config
from app.algorithms.NSA import NSA
from app.algorithms.NSAFactory import NSAFactory
from app.utils.utils import calculate_accuracy, create_prediction_csv, create_train_test_database, calculate_confusion_matrix, \
    mkdir, plot_confusion_matrix

warnings.simplefilter('ignore', category=NumbaWarning)
warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def simulate(columns, X_train, y_train, X_test, y_test, negative_selection: NSA, seed_nsa, seed_ds, output_folder=''):
    negative_selection.train(X_train, y_train)
    selfs, not_selfs = negative_selection.test(X_test)
    print(f'\n*** Settings: seed_ds={seed_ds}; ***')
    negative_selection.print_configuration()
    print(f'Predicted Selfs Nr.: {np.size(selfs, axis=0)}')
    print(f'Predicted Not Selfs Nr.: {np.size(not_selfs, axis=0)}')

    negative_selection.save_model(columns, output_folder)
    create_prediction_csv(columns, selfs, not_selfs, output_folder)

    # y_test = [0.0 for x in X_test]
    y_pred = np.array([x[1] for x in selfs + not_selfs])
    print(f'Actual: {np.size(y_test)} \n{y_test}')
    print(f'Predicted: {np.size(y_pred)} \n{y_pred}')

    # calculate accuracy
    accuracy = calculate_accuracy(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label=0)
    specificity = precision_score(y_test, y_pred, pos_label=0)
    print(f'Accuracy: {accuracy}')

    conf_matrix = calculate_confusion_matrix(y_test, y_pred)
    print(f'Confusion matrix:\n {conf_matrix}')
    plot_confusion_matrix(conf_matrix, output_folder)

    original_stdout = sys.stdout
    with open(os.path.join(output_folder, 'results.txt'), 'w') as f:
        sys.stdout = f  # Change the standard output to the output file.
        print(f'Seed NSA: {seed_nsa}')
        print(f'Seed shuffle dataset: {seed_ds}')
        print(f'Actual: {np.size(y_test)} \n{y_test}')
        print(f'Predicted: {np.size(y_pred)} \n{y_pred}')
        print(f'Accuracy: {accuracy}')
        print(f'Confusion matrix:\n {conf_matrix} \n')
        negative_selection.print_statistics()
        sys.stdout = original_stdout

    return accuracy, sensitivity, specificity


def run():
    config = Config()
    cwd = os.getcwd()
    config.load(os.path.join(cwd, 'config/config.yml'))
    start_time = datetime.now().strftime(config.date_format)
    output_parent_folder = os.path.join(config.output_folder, start_time)
    mkdir(output_parent_folder)

    nsa_factory = NSAFactory(config)

    if config.shutdown_on_end:
        print('[!] PC will be shut down after saving the results!\n')

    radius = config.nsa_radius if config.algorithm == 'NSA' else config.self_radius
    for i in range(len(radius)):
        output_folder = ""
        problem_dims = -1
        final_accuracy = []
        final_sensitivity = []
        final_specificity = []

        for seed_nsa in config.nsa_seeds:
            for ds_shuffle_seed in config.dataset_shuffle_seeds:
                # Create output folder structure in this way:
                #  /StartDateHour/DetectorRadius/NsaSeed/DS ShuffleSeed/all the output files
                output_folder = os.path.join(output_parent_folder, f'Rad_{radius[i]}', f'NsaSeed_{seed_nsa}',
                                             f'ShuffleSeed_{ds_shuffle_seed}')
                mkdir(output_folder)

                columns, X_train, X_test, y_train, y_test = \
                    create_train_test_database(os.path.join(cwd, config.input_folder),
                                               config.dataset_filename,
                                               config.shuffle_dataset, ds_shuffle_seed,
                                               config.train_split_percentage,
                                               config.show_plot,
                                               config.normalize_dataset)
                negative_selection = nsa_factory.get_nsa(seed_nsa=seed_nsa,
                                                         radius_index=i,
                                                         problem_dim=X_train.shape[1])
                accuracy, sensitivity, specificity = \
                    simulate(columns, X_train, X_test, y_train, y_test,
                             negative_selection=negative_selection,
                             seed_nsa=seed_nsa,
                             seed_ds=ds_shuffle_seed,
                             output_folder=output_folder)
                problem_dims = len(columns)
                final_accuracy.append(accuracy)
                final_sensitivity.append(sensitivity)
                final_specificity.append(specificity)

            average_accuracy = np.mean(final_accuracy)
            stddev_accuracy = np.std(final_accuracy)
            max_accuracy = np.max(final_accuracy)
            min_accuracy = np.min(final_accuracy)

            average_sensitivity = np.mean(final_sensitivity)
            stddev_sensitivity = np.std(final_sensitivity)
            max_sensitivity = np.max(final_sensitivity)
            min_sensitivity = np.min(final_sensitivity)

            average_specificity = np.mean(final_specificity)
            stddev_specificity = np.std(final_specificity)
            max_specificity = np.max(final_specificity)
            min_specificity = np.min(final_specificity)

            with open(os.path.join(os.path.dirname(output_folder), 'final_results.txt'), 'w') as f:
                sys.stdout = f  # Change the standard output to the output file.
                print("\n", config.to_string())
                print("--- Specific test settings and results ---")
                print("Seed NSA: ", seed_nsa)
                print("Dataset: ", config.dataset_filename)
                print("Number of features: ", problem_dims)
                print("Number of n_lymphocytes: ", config.nsa_detectors_nr)
                print("Radius: ", radius[i])
                print("Percentage of Selfs: ", config.train_split_percentage)
                print("Final Accuracy: ", average_accuracy)
                print("StdDev Accuracy: ", stddev_accuracy)
                print("Max Accuracy: ", max_accuracy)
                print("Min Accuracy: ", min_accuracy)
                print("--- Sensitivity [tp / (tp + fn)] (riga superiore della matrice di confusione) ---")
                print("Final Sensitivity: ", average_sensitivity)
                print("StdDev Sensitivity: ", stddev_sensitivity)
                print("Max Sensitivity: ", max_sensitivity)
                print("Min Sensitivity: ", min_sensitivity)
                print("--- Specificity [tn / (tn + fp)] (riga inf. della mat. di conf.) ---")
                print("Final Specificity: ", average_specificity)
                print("StdDev Specificity: ", stddev_specificity)
                print("Max Specificity: ", max_specificity)
                print("Min Specificity: ", min_specificity)
            sys.stdout = sys.__stdout__

    config.reload()
    if config.shutdown_on_end:
        print('\n\n\n[!] Shutting down!')
        os.system("shutdown /s /t 1")

