# >standard library imports< #
import glob
import math
import datetime
import dill
dill._dill._reverse_typemap['ClassType'] = type
import random
import warnings
import tkinter as tk
from ipywidgets import widgets
from tkinter import filedialog
import colorcet as cc
import matplotlib
import winsound
import time

# >third party imports< #
import pywt                     # wavelet processing 
import imageio                  # wavelet image
import numpy as np              # data handling
import pandas as pd             # data handling
import seaborn as sns           # violin plotting
import matplotlib.pyplot as plt # series plotting

# preprocessing core
#import scipy
#from scipy import stats
#from scipy.signal import butter, lfilter, freqz
#from scipy import interp, arange, exp
#from scipy.interpolate import Rbf,griddata
## ML core
#from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,make_scorer
#
#import tensorflow as tf
##Callback = tf.keras.callbacks.Callback
#import sklearn
#from keras.layers import AveragePooling1D
#from keras.utils import normalize
#from keras.models import Model, Sequential
#from keras.wrappers.scikit_learn import KerasRegressor
#from keras.preprocessing.sequence import TimeseriesGenerator
#from keras.layers import Input, Dense, LSTM, Activation, Conv1D, Bidirectional, MaxPooling1D, Flatten,Dropout,Reshape,GlobalAveragePooling1D,GlobalMaxPooling1D