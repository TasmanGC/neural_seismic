import numpy as np
from tensorflow.keras import activations
from keras.models import Sequential
from keras.layers import (Dense, Conv1D, LSTM, MaxPooling1D, Dropout,
                           Reshape, Flatten, GlobalAveragePooling1D, Input)


def series_data_gen(batch_T_obj, feat_num):
    samples_per_epoch = len(batch_T_obj)
    counter = 0
    while 1:
        iter_trace = batch_T_obj[counter]
        if iter_trace.feat_space is None:
            iter_trace.gen_feat_space()
        training = iter_trace.feat_space.fillna(0).astype(np.float32).values
        signal_sel = training[:, :feat_num]
        target = training[:, 35]
        signal_sel = np.stack(signal_sel)
        shape = signal_sel.shape
        signal_sel = signal_sel.reshape((1, shape[0], shape[1]))
        target = np.stack(target)
        target = target.reshape((1, 500, 1))
        counter += 1
        yield (signal_sel, target)
        if counter >= samples_per_epoch:
            counter = 0


# BPNN MODEL MAKERS

def BPNN_1(hidden_neuron, n_hidden, optim):
    modelexp = Sequential()
    modelexp.add(Input(shape=(500, 1)))
    modelexp.add(Dense(50, activation=None))
    for i in range(n_hidden):
        modelexp.add(Dense(hidden_neuron, activation=activations.tanh))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def BPNN_2(hidden_neuron, n_hidden, optim):
    modelexp = Sequential()
    modelexp.add(Input(shape=(500, 2)))
    modelexp.add(Dense(50, activation=None))
    for i in range(n_hidden):
        modelexp.add(Dense(hidden_neuron, activation=activations.tanh))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def BPNN_33(hidden_neuron, n_hidden, optim):
    modelexp = Sequential()
    modelexp.add(Input(shape=(500, 33)))
    modelexp.add(Dense(50, activation=None))
    for i in range(n_hidden):
        modelexp.add(Dense(hidden_neuron, activation=activations.tanh))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def BPNN_1_DO(hidden_neuron, n_hidden, optim, a_func, per):
    modelexp = Sequential()
    modelexp.add(Input(shape=(500, 1)))
    modelexp.add(Dropout(per))
    modelexp.add(Dense(50, activation=None))
    for i in range(n_hidden):
        modelexp.add(Dense(hidden_neuron, activation=activations.tanh))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


# CONV1D MODEL MAKERS

def CONV_1(hidden_neuron, n_hidden, optim):
    modelexp = Sequential()
    modelexp.add(Input(shape=(500, 1)))
    modelexp.add(Conv1D(10, kernel_size=20, padding="same", activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500, 1)))
    modelexp.add(Conv1D(50, kernel_size=20, padding="same", activation=activations.tanh))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500, 1)))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def CONV_2(hidden_neuron, n_hidden, optim):
    modelexp = Sequential()
    modelexp.add(Input(shape=(500, 2)))
    modelexp.add(Conv1D(10, kernel_size=20, padding="same", activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500, 1)))
    modelexp.add(Conv1D(10, kernel_size=20, padding="same", activation=activations.tanh))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500, 1)))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def CONV_33(hidden_neuron, n_hidden, optim):
    modelexp = Sequential()
    modelexp.add(Input(shape=(500, 33)))
    modelexp.add(Conv1D(10, kernel_size=20, padding="same", activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500, 1)))
    modelexp.add(Conv1D(50, kernel_size=20, padding="same", activation=activations.tanh))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500, 1)))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def CONV_1_DO(hidden_neuron, n_hidden, optim, a_func, per):
    modelexp = Sequential()
    modelexp.add(Input(shape=(500, 1)))
    modelexp.add(Dropout(per))
    modelexp.add(Conv1D(10, kernel_size=20, padding="same", activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500, 1)))
    modelexp.add(Conv1D(50, kernel_size=20, padding="same", activation=activations.tanh))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500, 1)))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


# LSTM MODEL MAKERS

def LSTM_1(hidden_neuron, n_hidden, optim):
    modelexp = Sequential()
    modelexp.add(Input(shape=(500, 1)))
    modelexp.add(LSTM(20, return_sequences=True, activation=None, unit_forget_bias=True))
    for i in range(n_hidden):
        modelexp.add(LSTM(hidden_neuron, return_sequences=True, unit_forget_bias=True))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def LSTM_2(hidden_neuron, n_hidden, optim):
    modelexp = Sequential()
    modelexp.add(Input(shape=(500, 2)))
    modelexp.add(LSTM(20, return_sequences=True, activation=None, unit_forget_bias=True))
    for i in range(n_hidden):
        modelexp.add(LSTM(hidden_neuron, return_sequences=True, unit_forget_bias=True))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def LSTM_33(hidden_neuron, n_hidden, optim):
    modelexp = Sequential()
    modelexp.add(Input(shape=(500, 33)))
    modelexp.add(LSTM(20, return_sequences=True, activation=None, unit_forget_bias=True))
    for i in range(n_hidden):
        modelexp.add(LSTM(hidden_neuron, return_sequences=True, unit_forget_bias=True))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def LSTM_conv_33(hidden_neuron, n_hidden, optim):
    modelexp = Sequential()
    modelexp.add(LSTM(10, input_shape=(500, 33), return_sequences=True, activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500, 1)))
    modelexp.add(LSTM(10, input_shape=(500, 33), return_sequences=True, activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500, 1)))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def LSTMPOOL_2(hidden_neuron, optim):
    modelexp = Sequential()
    modelexp.add(LSTM(20, input_shape=(500, 2), return_sequences=True, activation=None))
    modelexp.add(LSTM(20, unit_forget_bias=True, return_sequences=True))
    modelexp.add(GlobalAveragePooling1D('channels_first'))
    modelexp.add(Reshape((500, 1)))
    modelexp.add(LSTM(20, unit_forget_bias=True, return_sequences=True))
    modelexp.add(GlobalAveragePooling1D('channels_first'))
    modelexp.add(Reshape((500, 1)))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def LSTMPOOL_33(hidden_neuron, optim):
    modelexp = Sequential()
    modelexp.add(LSTM(20, input_shape=(500, 33), return_sequences=True, activation=None))
    modelexp.add(LSTM(20, unit_forget_bias=True, return_sequences=True))
    modelexp.add(GlobalAveragePooling1D('channels_first'))
    modelexp.add(Reshape((500, 1)))
    modelexp.add(LSTM(20, unit_forget_bias=True, return_sequences=True))
    modelexp.add(GlobalAveragePooling1D('channels_first'))
    modelexp.add(Reshape((500, 1)))
    modelexp.compile(loss='mean_absolute_error', optimizer=optim, metrics=['accuracy'])
    return modelexp


def run_all_models(train, vali, test):
    BP_NN_1 = BPNN_1(10, 10, 'Adamax')
    BP_NN_1.summary()
    BP_01_hist = BP_NN_1.fit(series_data_gen(train, 1), steps_per_epoch=100, epochs=int(len(train) / 100))
    BP_NN_1_PRED = BP_NN_1.predict(series_data_gen(test, 1), steps=len(test))

    BP_NN_33 = BPNN_33(10, 10, 'Adamax')
    BP_NN_33.summary()
    BP_33_hist = BP_NN_33.fit(series_data_gen(train, 33), steps_per_epoch=100, epochs=int(len(train) / 100))
    BP_NN_33_PRED = BP_NN_33.predict(series_data_gen(test, 33), steps=len(test))

    CV_NN_1 = CONV_1(10, 10, 'Adamax')
    CV_NN_1.summary()
    CV_NN_1_hist = CV_NN_1.fit(series_data_gen(train, 1), steps_per_epoch=100, epochs=int(len(train) / 100))
    CV_NN_1_PRED = CV_NN_1.predict(series_data_gen(test, 1), steps=len(test))

    CV_NN_33 = CONV_33(10, 10, 'Adamax')
    CV_NN_33.summary()
    CV_NN_33_hist = CV_NN_33.fit(series_data_gen(train, 33), steps_per_epoch=100, epochs=int(len(train / 100)))
    CV_NN_33_PRED = CV_NN_33.predict(series_data_gen(test, 33), steps=len(test))

    LS_TM_1 = LSTM_1(10, 10, 'Adamax')
    LS_TM_1.summary()
    LS_TM_1_hist = LS_TM_1.fit(series_data_gen(train, 1), steps_per_epoch=100, epochs=int(len(train / 100)))
    LS_TM_1_PRED = LS_TM_1.predict(series_data_gen(test, 1), steps=len(test))

    LS_TM_33 = LSTM_33(10, 10, 'Adamax')
    LS_TM_33.summary()
    LS_TM_33_hist = LS_TM_33.fit(series_data_gen(train, 33), steps_per_epoch=100, epochs=int(len(train) / 100))
    LS_TM_33_PRED = LS_TM_33.predict(series_data_gen(test, 33), steps=len(test))

    models = [BP_NN_1, BP_NN_33, CV_NN_1, CV_NN_33, LS_TM_1, LS_TM_33]
    predictions = [BP_NN_1_PRED, BP_NN_33_PRED, CV_NN_1_PRED, CV_NN_33_PRED, LS_TM_1_PRED, LS_TM_33_PRED]
    return models, predictions


def run_all_models_consis(train, vali, test,
                          optimizer='Adamax', hidden_neurons=10, hidden_layers=1,
                          steps_per_epoch=100, epochs=5):
    BP_NN_1 = BPNN_1(hidden_neurons, hidden_layers, optimizer)
    BP_NN_1.summary()
    BP_01_hist = BP_NN_1.fit(series_data_gen(train, 1), steps_per_epoch=steps_per_epoch, epochs=epochs)
    BP_NN_1_PRED = BP_NN_1.predict(series_data_gen(test, 1), steps=len(test))

    BP_NN_2 = BPNN_2(hidden_neurons, hidden_layers, optimizer)
    BP_NN_2.summary()
    BP_02_hist = BP_NN_2.fit(series_data_gen(train, 2), steps_per_epoch=steps_per_epoch, epochs=epochs)
    BP_NN_2_PRED = BP_NN_2.predict(series_data_gen(test, 2), steps=len(test))

    BP_NN_33 = BPNN_33(hidden_neurons, hidden_layers, optimizer)
    BP_NN_33.summary()
    BP_33_hist = BP_NN_33.fit(series_data_gen(train, 33), steps_per_epoch=steps_per_epoch, epochs=epochs)
    BP_NN_33_PRED = BP_NN_33.predict(series_data_gen(test, 33), steps=len(test))

    CV_NN_1 = CONV_1(hidden_neurons, hidden_layers, optimizer)
    CV_NN_1.summary()
    CV_NN_1_hist = CV_NN_1.fit(series_data_gen(train, 1), steps_per_epoch=steps_per_epoch, epochs=epochs)
    CV_NN_1_PRED = CV_NN_1.predict(series_data_gen(test, 1), steps=len(test))

    CV_NN_2 = CONV_2(hidden_neurons, hidden_layers, optimizer)
    CV_NN_2.summary()
    CV_NN_2_hist = CV_NN_2.fit(series_data_gen(train, 2), steps_per_epoch=steps_per_epoch, epochs=epochs)
    CV_NN_2_PRED = CV_NN_2.predict(series_data_gen(test, 2), steps=len(test))

    CV_NN_33 = CONV_33(hidden_neurons, hidden_layers, optimizer)
    CV_NN_33.summary()
    CV_NN_33_hist = CV_NN_33.fit(series_data_gen(train, 33), steps_per_epoch=steps_per_epoch, epochs=epochs)
    CV_NN_33_PRED = CV_NN_33.predict(series_data_gen(test, 33), steps=len(test))

    LS_TM_1 = LSTM_1(hidden_neurons, hidden_layers, optimizer)
    LS_TM_1.summary()
    LS_TM_1_hist = LS_TM_1.fit(series_data_gen(train, 1), steps_per_epoch=steps_per_epoch, epochs=epochs)
    LS_TM_1_PRED = LS_TM_1.predict(series_data_gen(test, 1), steps=len(test))

    LS_TM_2 = LSTM_2(hidden_neurons, hidden_layers, optimizer)
    LS_TM_2.summary()
    LS_TM_2_hist = LS_TM_2.fit(series_data_gen(train, 2), steps_per_epoch=steps_per_epoch, epochs=epochs)
    LS_TM_2_PRED = LS_TM_2.predict(series_data_gen(test, 2), steps=len(test))

    LS_TM_33 = LSTM_33(hidden_neurons, hidden_layers, optimizer)
    LS_TM_33.summary()
    LS_TM_33_hist = LS_TM_33.fit(series_data_gen(train, 33), steps_per_epoch=steps_per_epoch, epochs=epochs)
    LS_TM_33_PRED = LS_TM_33.predict(series_data_gen(test, 33), steps=len(test))

    models = [BP_NN_1, BP_NN_2, BP_NN_33, CV_NN_1, CV_NN_2, CV_NN_33, LS_TM_1, LS_TM_2, LS_TM_33]
    predictions = [BP_NN_1_PRED, BP_NN_2_PRED, BP_NN_33_PRED,
                   CV_NN_1_PRED, CV_NN_2_PRED, CV_NN_33_PRED,
                   LS_TM_1_PRED, LS_TM_2_PRED, LS_TM_33_PRED]
    histories = [BP_01_hist, BP_02_hist, BP_33_hist,
                 CV_NN_1_hist, CV_NN_2_hist, CV_NN_33_hist,
                 LS_TM_1_hist, LS_TM_2_hist, LS_TM_33_hist]
    return models, predictions, histories
