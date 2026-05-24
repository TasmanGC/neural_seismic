import numpy as np


def process_fb_calc(predicted_set, predictions):
    if len(predicted_set) != len(predictions):
        raise Exception('Missmatched number of predictions and traces.')
    pred_val = []
    real = []
    pred_ser = []
    tid = []
    for i in range(len(predicted_set)):
        predicted_set_i = predicted_set[i]
        prediction_i = predictions[i].flatten()
        predicted_set_i.prediction_series[0] = prediction_i
        edited = []
        for j in range(len(prediction_i)):
            iter_sel = max(prediction_i[:j + 1])
            edited.append(iter_sel)
        predicted_set_i.prediction_series[1] = edited
        dif = np.diff(edited, 1)
        area = np.trapezoid(dif)
        normalized = dif / area
        predicted_set_i.prediction_series[2] = normalized
        predicted_set_i.FB_Picks[2] = np.argmax(normalized)
        real.append(predicted_set_i.FB_Picks[1])
        pred_val.append(predicted_set_i.FB_Picks[2])
        pred_ser.append(prediction_i)
        tid.append(predicted_set_i.iD)
    return pred_val, real, pred_ser, tid


def convert_series(prediction_array):
    running_max = []
    for j in range(len(prediction_array)):
        iter_sel = max(prediction_array[:j + 1])
        running_max.append(iter_sel)
    dif = np.diff(running_max, 1)
    area = np.trapezoid(dif)
    pseudo_pdf = dif / area
    return pseudo_pdf


def test_coppens(listoftraces, window):
    results = []
    lad = []
    for i in range(len(listoftraces)):
        samples = np.asarray(listoftraces[i].Signal_1[1]) ** 2
        sub_samples = []
        for i in range(int(len(samples))):
            sub_samples.append(samples[i:i + window])
        tau_window = [np.trapezoid(x) for x in sub_samples]
        normiliser = []
        for i in range(len(tau_window)):
            normilizer_i = samples[:i + 5]
            normiliser.append(np.trapezoid(normilizer_i))
        x = np.array(tau_window) / np.array(normiliser)
        x = x[50:]
        results.append(int(np.argmax(x)))
        lad.append(x)
    return results, lad


def mean_residual_calcs(test_set):
    model_keys = ['BPNN01', 'BPNN02', 'BPNN33', 'CVNN01', 'CVNN02', 'CVNN33', 'LSTM01', 'LSTM02', 'LSTM33']
    return [
        np.mean([int(trace.FB_Picks[1] * 1000) - trace.prediction__value[k] for trace in test_set])
        for k in model_keys
    ]


def confidence_calcs(pred_array, model_array):
    confidence = []
    for i in range(len(model_array)):
        prediction_series = model_array[i]
        prediction_val = pred_array[i]
        if not np.isnan(prediction_series).any():
            if prediction_val != 0 and prediction_val != 500:
                prediction_window = prediction_series[prediction_val - 2:prediction_val + 2]
            if prediction_val == 0:
                prediction_window = prediction_series[:5]
            if prediction_val == 500:
                prediction_window = prediction_series[495:]
            mods = np.trapezoid(abs(prediction_window)) / np.trapezoid(abs(prediction_series))
        if np.isnan(prediction_series).any():
            mods = 0
        confidence.append(mods)
    return confidence
