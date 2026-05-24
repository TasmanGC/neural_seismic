import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import winsound
from keras.utils import normalize


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1, length=100, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end='\r', flush=True)
    if iteration == total:
        print()


def notify():
    _16 = 125
    _08 = 250
    _04 = 500
    winsound.Beep(293, _16)
    winsound.Beep(293, _16)
    winsound.Beep(587, _08)
    winsound.Beep(440, _04)
    time.sleep(0.125)
    winsound.Beep(391, _08)
    time.sleep(0.100)
    winsound.Beep(415, _08)
    time.sleep(0.125)
    winsound.Beep(349, _16)
    winsound.Beep(349, _16)
    winsound.Beep(293, _16)
    winsound.Beep(349, _16)
    winsound.Beep(391, _16)


def keras_input(learn_set, feat_num):
    num_samples = len(learn_set)
    signal_set = []
    target_a_set = []
    target_b_set = []
    for i in range(num_samples):
        training = learn_set[i].feat_space.fillna(0)
        training = training.values
        signal_sel = training[:, :feat_num]
        signal_set.append(normalize(signal_sel, axis=-1, order=2))
        target_a_set.append(training[:, 35])
        target_b_set.append(training[:, 36])
    signal_set = np.stack(signal_set)
    target_a_set = np.stack(target_a_set)
    target_b_set = np.stack(target_b_set)
    return signal_set, target_a_set, target_b_set


def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]


def calc_iter(seismic_df):
    """Identify unique station locations from Northing and Easting."""
    num_iter = seismic_df.groupby(['Northing', 'Easting']).ngroups
    combo_list = seismic_df.groupby(['Northing', 'Easting']).mean().reset_index()
    combo_list = combo_list[['Northing', 'Easting']]
    return num_iter, combo_list


def custom_sort(t):
    return t[1]
