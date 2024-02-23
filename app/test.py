import ast
import os
import re
import numpy as np

import pandas as pd

from app.utils.utils import test_null_hypothesis


def get_results(folder):
    results = []
    for root, dirs, files in os.walk(folder):
        if root != folder:
            filename = os.path.join(root, 'results.txt')
            if os.path.exists(filename):
                with open(filename) as file:
                    lines = file.readlines()
                    x = ' '.join(lines[9:14])
                    x = re.sub("\s+", ",", x.strip())
                    y = ast.literal_eval(x)
                    results.extend(y)
    results = np.array(results)
    return results


def get_avg_gen_disc(folder):
    generated = []
    discarded = []
    for root, dirs, files in os.walk(folder):
        if root != folder:
            filename = os.path.join(root, 'results.txt')
            if os.path.exists(filename):
                # print(filename)
                with open(filename) as file:
                    lines = file.readlines()

                    actual_generated = int(lines[21][27:])
                    actual_discarded = int(lines[22][27:])

                    generated.append(actual_generated)
                    discarded.append(actual_discarded)
    generated = np.array(generated)
    discarded = np.array(discarded)
    avg_gen = np.mean(generated)
    avg_dis = np.mean(discarded)
    return avg_gen, avg_dis


def get_det_res(path='detectors.csv'):
    detectors = pd.read_csv(path)
    reduced_det = []
    print(detectors)
    for detector in detectors.iterrows():
        s = 0
        for point in detector:
            s += point
        reduced_det.append(s)
    reduced_det = np.array(reduced_det)
    return reduced_det


if __name__ == '__main__':
    # Check results between two runs

    # path: results/<folder_name>/<folder_radius>/<folder_nsa_seed>
    first_folder = "results/prof_solution/Rad_5.03/NsaSeed_2222/"
    second_folder = "results/nsa-4000dts/Rad_5.03/NsaSeed_9311/"

    first_results = get_results(first_folder)
    second_results = get_results(second_folder)

    is_same, stat, p = test_null_hypothesis(first_results, second_results)
    print(f'Are the same? {is_same}')
    print('Statistics=%f, p=%f' % (stat, p))


if __name__ == '__main__':
    # Check average detectors generated and discarded

    first_folder = "results/nsa-4000dts/Rad_5.03/"
    print(get_avg_gen_disc(first_folder))
