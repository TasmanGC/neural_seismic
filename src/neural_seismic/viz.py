import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns
import colorcet as cc
import pywt
from scipy.interpolate import griddata


def plot_predictions(trace_selection, comp_trace, name):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(len(trace_selection), figsize=(9, 8), sharex=True, dpi=100)
    labels = ['A', 'B', 'C']
    for j in range(len(trace_selection)):
        trace = [x for x in comp_trace if x.iD == trace_selection[j]][0]
        titles = []
        predictions = []
        preds = list(trace.prediction__value.keys())
        no_plz = ['BPNN02', 'CVNN02', 'LSTM02']
        for q in range(len(preds)):
            if preds[q] not in no_plz:
                titles.append(preds[q])
                predictions.append(trace.prediction__value[preds[q]])
        color = ['m', 'C0', 'C0', 'C1', 'C1', 'C2', 'C2']
        ls = ['-', '-', ':', '-', ':', '-', ':']
        titles = [x.replace("BPNN", "NN") for x in titles]
        titles = [x.replace("CVNN", "CNN") for x in titles]
        print(titles)
        y = np.linspace(1, 500, 500)
        a = trace.Signal_1[0]
        b = trace.Signal_1[1]
        rio = trace.FB_Picks[1] * 1000
        axes[j].plot(y, a, label='Raw', color='k', alpha=0.6)
        axes[j].plot(y, b, label='Low-Pass Filter', color='k')
        axes[j].axvline(rio, label='Interpreted First Break', color='r')
        axes[j].text(480, 0.000018, labels[j], fontsize=20)
        axes[j].set_xlim((-10, 510))
        axes[j].set_ylim((-0.00003, 0.00003))
        axes[j].set_yticks([0.00002, 0, -0.00002])
        axes[j].set_yticklabels(['+', '0', '-'])
        for i in range(len(ls)):
            axes[j].axvline(predictions[i], ls=ls[i], color=color[i], label=titles[i])
    axes[len(trace_selection) - 1].set_xlabel('Time (ms)')
    axes[len(trace_selection) - 1].legend(bbox_to_anchor=(0.00, -0.3), loc='upper left', ncol=5)
    axes[1].set_ylabel('Amplitude')
    if len(trace_selection) == 2:
        axes[0].set_ylabel('Amplitude')
    plt.gcf().subplots_adjust(bottom=0.30)
    plt.savefig('Figures/' + name + '.png')


def plot_scalogram(sel_trace):
    fig, axes = plt.subplots(2, figsize=(9, 8), sharex=True, dpi=100, gridspec_kw={'height_ratios': [1, 2]})
    trace_i = sel_trace
    raw = trace_i.Signal_1[0]
    fil = trace_i.Signal_1[1]
    first_break = trace_i.FB_Picks[1] * 1000
    tim = np.linspace(1, 500, 500)
    sel_trace_iD = trace_i.iD

    wp = pywt.WaveletPacket(raw, 'db2', 'sym', maxlevel=5)
    nodes = wp.get_level(5, order="freq")
    labels = [n.path for n in nodes]
    values = np.array([n.data for n in nodes], 'd')
    values = abs(values)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('CET', cc.CET_D1A, 256)

    axes[0].plot(tim, raw, label='Raw', color='k', alpha=0.6)
    axes[0].plot(tim, fil, label='Low-Pass Filter', color='k')
    axes[0].axvline(first_break, label='Interpreted First Break', color='r')
    axes[0].axhline(0, color='k', ls=':')
    axes[0].text(460, (max(raw) - max(raw) * 0.3), 'A', fontsize=35)
    axes[0].set_yticks([0.00002, 0, -0.00002])
    axes[0].set_yticklabels(['+', '0', '-'])
    axes[0].set_ylabel("Amplitude")

    im = plt.imshow(values, interpolation='gaussian', cmap=cmap, aspect="auto",
                    origin="lower", extent=[0, 500, 0, len(values)])
    axes[1].imshow(values, interpolation='gaussian', cmap=cmap, aspect="auto",
                   origin="lower", extent=[0, 500, 0, len(values)])
    axes[1].axvline(first_break, label='Interpreted First Break', color='r')
    plt.colorbar(im, orientation='horizontal')
    axes[1].set_ylabel('Scale')
    axes[1].text(460, 25, 'B', fontsize=35, color='white')

    plt.tight_layout()
    plt.savefig('Figures/Example_Scalogram_iD_' + str(sel_trace_iD) + '.png')
    plt.show()


def plot_scalogram_alt(trace):
    fig, axes = plt.subplots(1, figsize=(10, 4), sharex=True, dpi=500)
    trace_i = trace
    raw = trace_i.Signal_1[0]
    fil = trace_i.Signal_1[1]
    sel_trace_iD = trace_i.iD
    first_break = trace_i.FB_Picks[1] * 1000
    tim = np.linspace(1, 500, 500)

    wp = pywt.WaveletPacket(raw, 'db2', 'sym', maxlevel=5)
    nodes = wp.get_level(5, order="freq")
    labels = [n.path for n in nodes]
    values = np.array([n.data for n in nodes], 'd')
    values = abs(values)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('CET', cc.CET_D1A, 256)

    im = plt.imshow(values, interpolation='gaussian', cmap=cmap, aspect="auto",
                    origin="lower", extent=[0, 500, 0, len(values)])
    axes.imshow(values, interpolation='gaussian', cmap=cmap, aspect="auto",
                origin="lower", extent=[0, 500, 0, len(values)])
    axes.axvline(first_break, label='Interpreted First Break', color='r')
    axes.set_ylabel('Pseudo Frequency')

    ax2 = axes.twinx()
    ax2.plot(tim, raw, label='Raw', color='w', alpha=0.3)
    ax2.plot(tim, fil, label='Low-Pass Filter', color='w', alpha=0.8)
    ax2.axvline(first_break, label='Interpreted First Break', color='r')
    ax2.set_yticks([0.00002, 0, -0.00002])
    ax2.set_yticklabels(['+', '0', '-'])
    ax2.set_ylabel("Amplitude")
    axes = plt.gca()
    axes.set_xlim([0, 500])
    plt.tight_layout()
    plt.savefig('Figures/Example_Scalogram_iD_' + str(sel_trace_iD) + '_alt.png')
    plt.show()


def trace_comparison(comp_trace, sel_trace_iD):
    if len(sel_trace_iD) > 3:
        print('Select fewer traces')
    if len(sel_trace_iD) <= 3:
        labels = ['A', 'B', 'C']
        plt.rcParams["font.family"] = "Times New Roman"
        fig, axes = plt.subplots(len(sel_trace_iD), sharex=True, sharey=True, figsize=(9, 8))
        fig.add_subplot(111, frameon=False)
        plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
        plt.grid(False)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude ")
        for idx, ax in enumerate(axes):
            trace_i = [x for x in comp_trace if x.iD == sel_trace_iD[idx]][0]
            raw = trace_i.Signal_1[0]
            fil = trace_i.Signal_1[1]
            first_break = trace_i.FB_Picks[1] * 1000
            tim = np.linspace(1, 500, 500)
            ax.plot(tim, raw, label='Raw', color='k', alpha=0.6)
            ax.plot(tim, fil, label='Low-Pass Filter', color='k')
            ax.axvline(first_break, label='Interpreted First Break', color='r')
            ax.axhline(0, color='k', ls=':')
            ax.set_xlim((100, 300))
            ax.set_yticks([0.00002, 0, -0.00002])
            ax.set_yticklabels(['+', '0', '-'])
            ax.text(290, 0.000025, labels[idx], fontsize=20)
        plt.savefig('Figures/Trace_Comparison.png', dpi=700)


def plot_uncertainty(selection, method):
    cet_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('CET', cc.CET_L3, 256)
    x0 = np.array([trace.rec_loc[0] for trace in selection])
    y0 = np.array([trace.rec_loc[1] for trace in selection])
    z1 = np.array([trace.Unc_Metrics[0] for trace in selection])
    z2 = np.array([trace.Unc_Metrics[1] for trace in selection])
    z3 = np.array([trace.Unc_Metrics[2] for trace in selection])
    xi = np.linspace(x0.min(), x0.max(), 1000)
    yi = np.linspace(y0.min(), y0.max(), 1000)
    if method == 'UM1':
        zi = griddata((x0, y0), z1, (xi[None, :], yi[:, None]), method='linear')
    if method == 'UM2':
        zi = griddata((x0, y0), z2, (xi[None, :], yi[:, None]), method='linear')
    if method == 'UM3':
        zi = griddata((x0, y0), z3, (xi[None, :], yi[:, None]), method='linear')
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    axes.set_xlabel('Easting')
    axes.set_ylabel('Northing')
    plt.contourf(xi, yi, zi, 15, cmap=cet_cmap)
    fig.canvas.draw()


def visualise_classifier(trace, comp_trace, all_classifiers):
    y = np.linspace(1, 500, 500)
    a = trace.Signal_1[0]
    b = trace.Signal_1[1]
    Post = int(trace.FB_Picks[1] * 1000)

    plt.rcParams["font.family"] = "Times new roman"
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(20, 10))

    raw = plt.plot(y, a, label='Raw', color='k', alpha=0.6)
    filt = plt.plot(y, b, label='Low-Pass Filter', color='k')

    ax.set_yticks([0.00003, 0, -0.00003])
    ax.set_yticklabels(['+', '0', '-'])
    ax.set_ylabel("Amplitude")
    ax.set_xlabel('Time (ms)')

    ax2 = ax.twinx()
    classif = ax2.plot(np.append([np.zeros(Post)], [np.ones(500 - Post)]), '--', color='red', label='Target Series')

    if all_classifiers == True:
        interp = ax2.axvline(Post, label='Interperter Prediction', color='r', linewidth=2)
        pred_series = trace.prediction_series['CVNN01']
        run_max = []
        for i in range(len(pred_series)):
            if i == 0:
                run_max.append(pred_series[0])
            if i > 0:
                run_max.append(max(pred_series[:i]))

        dif = np.diff(run_max, 1)
        area = np.trapezoid(dif)
        normalized = dif / area

        cdf = ax2.plot(run_max, color='b', label='CDF*')
        raw_p = ax2.plot(pred_series, '--', color='b', label='Returned Series')
        pdf = ax2.plot(normalized, '--', label='PDF', color='orange', linewidth=2.0)
        pred = ax2.axvline(np.argmax(normalized), label='Model Prediction', color='orange', linewidth=2)

    ax2.set_ylabel('Classification Series', rotation=270)

    lns = [raw, filt, classif, interp, raw_p, cdf, pdf, pred]
    lns = [line if not isinstance(line, list) else line[0] for line in lns]
    labs = [l.get_label() for l in lns]

    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['0', '1'])
    lgd = ax.legend(lns, labs, loc='lower center', bbox_to_anchor=(0.5, -0.4), ncol=4, fancybox=True)
    plt.xlim([125, 300])
    fig.set_size_inches(11, 5)
    fig.savefig('Figures/Example Classifcation Series.png', bbox_extra_artists=[lgd], bbox_inches='tight', dpi=500)


def plot_violin(dataframe):
    plt.rcParams.update({'font.size': 18})
    plt.rcParams["font.family"] = "Times new roman"
    fig, ax = plt.subplots(figsize=(15, 8))
    dataframe = dataframe[(dataframe['Feat Space'] != '1')]
    ax = sns.violinplot(x='Model', y='Variance', data=dataframe, split=True, hue="Feat Space",
                        palette="Blues_d", scale="count", bw=.2, inner="quartile")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    plt.title('Distribution of Prediction Residuals')
    plt.tight_layout()
    ax.set(ylabel='Prediction Residuals (ms)')
    ax.set(xlabel='')
    ax.set_xticklabels(['Coppens', 'NN', 'CNN', 'LSTM'])
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.savefig('Figures/Prediction_Residuals.png', dpi=700)
    return ax


def plot_viola(test):
    data_frame = []
    for i in range(len(test)):
        trace_i = test[i]
        keys = list(trace_i.prediction__value.keys())
        for key in keys:
            model = key[:4]
            flag = False
            if 'Co' in key:
                flag = True
            if '01' in key:
                feats = '01'
            if '02' in key:
                feats = '02'
            if '33' in key:
                feats = '33'
            value = trace_i.prediction__value[key]
            trace = trace_i.iD
            first = int(trace_i.FB_Picks[1] * 1000)
            if flag == True:
                data_frame.append({'Model': model, 'Feat Space': '01', 'Predicted First Break': value,
                                   'Trace ID': trace, 'RFB': first})
                data_frame.append({'Model': model, 'Feat Space': '33', 'Predicted First Break': value,
                                   'Trace ID': trace, 'RFB': first})
            if flag != True:
                data_frame.append({'Model': model, 'Feat Space': feats, 'Predicted First Break': value,
                                   'Trace ID': trace, 'RFB': first})
    final_df = pd.DataFrame(data_frame)
    variance = final_df["Predicted First Break"] - final_df["RFB"]
    final_df["Variance"] = variance
    return final_df


def model_plot(gather, suptitle):
    y = np.linspace(1, 500, 500) * -1
    filt = []
    raw = []
    jup_pick = []
    title = ["K1", "K2", "K3"]
    ni = len(gather)
    for i in range(ni):
        filt.append(gather[i].filt)
        raw.append(gather[i].raw)
        jup_pick.append(gather[i].jup_pick)
    fig, axes = plt.subplots(nrows=1, ncols=ni)
    for i in range(ni):
        axes[i].plot(raw[i], y, color='orange', alpha=0.6)
        axes[i].plot(filt[i], y)
        state = gather[i].ml_pred
        axes[i].set_title(title[i])
        axes[i].set_xticks([])
        if jup_pick[i] != None:
            time = jup_pick[i]
            axes[i].axhline(time * -1, label='Jupyter Pick', color='red')
        if isinstance(state, np.ndarray):
            sag = axes[i].twiny()
            sag.plot(state, y, color='orange')
            sag.set_xticks([])
        if state is None:
            print('No predictions yet.')
    fig.suptitle(suptitle, fontsize=16)


# Interactive area selection — uses module-level globals shared with line_select_callback/generate_selection
X0, Y0, X1, Y1, iDs = [], [], [], [], []
filtered_X0, filtered_Y0, filtered_X1, filtered_Y1, filtered_iDs = [], [], [], [], []
area_selection = []
comp_trace = []


def area_plot(comp_trace_arg):
    global X0, Y0, iDs
    X0 = []
    Y0 = []
    iDs = []
    for i in range(len(comp_trace_arg)):
        Trace_Selection = next(t for t in comp_trace_arg if t.iD == i)
        Y0.append(Trace_Selection.rec_loc[1])
        X0.append(Trace_Selection.rec_loc[0])
        iDs.append(Trace_Selection.iD)
    fig, ax = plt.subplots()
    ax.scatter(X0, Y0)
    return ax


def line_select_callback(eclick, erelease):
    global X0, Y0, Y1, X1, filtered_X0, filtered_Y0, filtered_X1, filtered_Y1, filtered_iDs
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    mask = (X0 > min(x1, x2)) & (X0 < max(x1, x2)) & \
           (Y0 > min(y1, y2)) & (Y0 < max(y1, y2))
    mask2 = (X1 > min(x1, x2)) & (X1 < max(x1, x2)) & \
            (Y1 > min(y1, y2)) & (Y1 < max(y1, y2))
    filtered_iDs = [i for indx, i in enumerate(iDs) if mask[indx] == True]
    filtered_X0 = [i for indx, i in enumerate(X0) if mask[indx] == True]
    filtered_Y0 = [i for indx, i in enumerate(Y0) if mask[indx] == True]
    filtered_X1 = [i for indx, i in enumerate(X1) if mask2[indx] == True]
    filtered_Y1 = [i for indx, i in enumerate(Y1) if mask2[indx] == True]
    generate_selection(filtered_iDs, filtered_X1, filtered_Y1)


def generate_selection(filtered_iDs, filtered_X1, filtered_Y1):
    global area_selection, comp_trace
    area_selection = []
    for i in range(len(filtered_iDs)):
        h = filtered_iDs[i]
        Trace_Selection = next(t for t in comp_trace if t.iD == h)
        area_selection.append(Trace_Selection)
