def mean_residual_calcs(test_set):
    BPNN01_var = np.mean([int(trace.FB_Picks[1]*1000)-trace.prediction__value['BPNN01'] for trace in test_set])
    BPNN02_var = np.mean([int(trace.FB_Picks[1]*1000)-trace.prediction__value['BPNN02'] for trace in test_set])
    BPNN33_var = np.mean([int(trace.FB_Picks[1]*1000)-trace.prediction__value['BPNN33'] for trace in test_set])
    
    CVNN01_var = np.mean([int(trace.FB_Picks[1]*1000)-trace.prediction__value['CVNN01'] for trace in test_set])
    CVNN02_var = np.mean([int(trace.FB_Picks[1]*1000)-trace.prediction__value['CVNN02'] for trace in test_set])
    CVNN33_var = np.mean([int(trace.FB_Picks[1]*1000)-trace.prediction__value['CVNN33'] for trace in test_set])
    
    LSTM01_var = np.mean([int(trace.FB_Picks[1]*1000)-trace.prediction__value['LSTM01'] for trace in test_set])
    LSTM02_var = np.mean([int(trace.FB_Picks[1]*1000)-trace.prediction__value['LSTM02'] for trace in test_set])
    LSTM33_var = np.mean([int(trace.FB_Picks[1]*1000)-trace.prediction__value['LSTM33'] for trace in test_set])
    return([BPNN01_var,BPNN02_var,BPNN33_var,CVNN01_var,CVNN02_var,CVNN33_var,LSTM01_var,LSTM02_var,LSTM33_var])

def save_exp_instance(train,vali,test,models,preds,coppens):
    '''This function takes all aspects of a given model run through and saves it.
    importantly it should run immediatley after run_all_models(). After saving the
    results are returned for the purposese of plotting.'''

    model_keys = ['BPNN01','BPNN02','BPNN33','CVNN01','CVNN02','CVNN33','LSTM01','LSTM02','LSTM33']

    if len(test)!=len(preds[0]):                                            # 0.1 - Sanity Check number of predictions
        raise Exception('Missmatched number of predictions and traces.')
    if len(model_keys)!=len(preds):                                         # 0.2 - Sanity Check number of models
        raise Exception('Missmatched models and predictions.')

    for method_i in range(len(preds)):                                      # 1.1 - for each model method
        key_i = model_keys[method_i]                                        # 1.2 - select model key
        print('Starting - '+key_i)
        pred = preds[method_i]                                              # 1.3 - select list of model predictions
        for index in range(len(pred)):                                      # 1.4 - cycle through each prediction
            pred_series = pred[index].flatten()                             # 1.5 - flatten the array
            edited = []                                                     # 1.6 - create a new blank array
            for ts in range(len(pred_series)):                              # 1.7 - for each time step in that prediction series
                edited.append(max(pred_series[:ts+1]))                      # 1.8 - append the running maximum
                
            dif_ = np.diff(edited,1)                                        # 2.0 - first order diferential of the series
            area = np.trapz(dif_)                                           # 2.1 - area under curve using simple trapz rule (fast calculation)
            nrml = dif_/area                                                # 2.2 - normilised result gives us a pseudo cumulative distribution function
            fbpk = np.argmax(nrml)                                          # 2.3 - argmax shows where highest likelyhood is (ie first break pick = fbpk)
            
            trace_i = test[index]                                           # 3.0 - select trace
            trace_i.prediction_series[key_i] = pred_series                  # 3.1 - place edited prediction series in dict in trace object
            trace_i.prediction__value[key_i] = fbpk                         # 3.2 - place prediction value in dict in trace object
            # -- LOOPS OVER ALL TRACES AND ALL MODEL SETS -- 
    
    for j in range(len(test)):
        trace_i = test[j]
        trace_i.prediction_series['Coppens'] = coppens[1][j]                
        trace_i.prediction__value['Coppens'] = coppens[0][j]
        
    
    now = datetime.datetime.now()
    date_string = now.strftime("%d%m%Y_%H%M")

    dill.dump(test, open(date_string+'_Test_TRACES.p', "wb" ),pickle.HIGHEST_PROTOCOL)            # 4.0 - saves the test data 
    dill.dump(train, open(date_string+'_Train_TRACES.p', "wb" ),pickle.HIGHEST_PROTOCOL)          # 4.1 - saves the train data 
    dill.dump(vali, open(date_string+'_Vali_TRACES.p', "wb" ),pickle.HIGHEST_PROTOCOL)            # 4.2 - saves the validation data 

    for  i in range(len(models)):                                           # 4.3 - save each of the models
        dill.dump(models[i], open(date_string+model_keys[i]+"_model.p","wb"))
        
    return(test)                                                            # 4.4 - returns the modified test set.

def plot_predictions(trace_selection, comp_trace, name):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams.update({'font.size': 10})
    fig, axes = plt.subplots(len(trace_selection),figsize=(9,8), sharex=True,dpi=100)
    labels=['A','B','C']
    for j in range(len(trace_selection)): # do trace
        trace = [x for x in comp_trace if x.iD==trace_selection[j]][0]
        # prediction details
        titles = []
        predictions = []
        preds = list(trace.prediction__value.keys())
        no_plz = ['BPNN02','CVNN02','LSTM02']
        for q in range(len(preds)): # gets the info for plotting the vertical lines 
            if preds[q] not in no_plz:
                titles.append(preds[q])
                predictions.append(trace.prediction__value[preds[q]])
        # plot colors
        color = ['m','C0','C0','C1','C1','C2','C2']
        ls = ['-','-',':','-',':','-',':']
        # data sets
        titles = [ x.replace("BPNN","NN")for x in titles]
        titles = [ x.replace("CVNN","CNN")for x in titles]
        print(titles)
        y=np.linspace(1,500,500)
        a=trace.Signal_1[0]
        b=trace.Signal_1[1]
        rio=trace.FB_Picks[1]*1000
        stn_num=trace.Station
        # plot stuff
        axes[j].plot(y,a,label='Raw',color='k',alpha=0.6)
        axes[j].plot(y,b,label='Low-Pass Filter',color='k')
        axes[j].axvline(rio,label='Interpreted First Break',color='r')
        axes[j].text(480, 0.000018, labels[j], fontsize=20)
        axes[j].set_xlim((-10,510))
        axes[j].set_ylim((-0.00003,0.00003))
        axes[j].set_yticks([0.00002,0,-0.00002])
        axes[j].set_yticklabels(['+','0','-'])  
        # cycles through the predictions
        for i in range(len(ls)):
            axes[j].axvline(predictions[i],ls=ls[i],color=color[i],label=titles[i])       
    axes[len(trace_selection)-1].set_xlabel('Time (ms)')
    axes[len(trace_selection)-1].legend(bbox_to_anchor=(0.00, -0.3),loc='upper left',ncol=5)
    axes[1].set_ylabel('Amplitude')
    if len(trace_selection)==2:
        axes[0].set_ylabel('Amplitude')
    plt.gcf().subplots_adjust(bottom=0.30)
    plt.savefig('Figures/'+name+'.png')   
    
def load_csv():
    '''Loads the CSV of seismic trace data and returns a pandas Dataframe'''
    root = tk.Tk()
    root.lift()
    file_path1 = filedialog.askopenfilename(filetypes=[('.csvfiles', '.csv')],title='Select Trace File')
    root.destroy()
    seismic_df=pd.read_csv(file_path1,na_values={"rfb": 'False',"Grav_sel":'False'},dtype={"rfb": float,"repeat":str})
    seismic_df['repeat']=seismic_df['repeat'].apply(bool)
    seismic_df['rfb']=seismic_df['rfb'].fillna(0)
    seismic_df['Grav_sel']=seismic_df['Grav_sel'].fillna(0)
    return(seismic_df)

def plot_scalogram_alt(trace):
    fig, axes = plt.subplots(1,figsize=(10,4), sharex=True,dpi=500)          
    trace_i = trace
    raw = trace_i.Signal_1[0]
    fil = trace_i.Signal_1[1]
    sel_trace_iD = trace_i.iD
    first_break = trace_i.FB_Picks[1]*1000
    tim = np.linspace(1,500,500)
    
    #performs wavelet transform
    wp = pywt.WaveletPacket(raw, 'db2', 'sym', maxlevel=5)
    nodes = wp.get_level(5, order="freq")
    labels = [n.path for n in nodes]
    values = np.array([n.data for n in nodes], 'd')
    values = abs(values)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('CET',cc.CET_D1A, 256)
    
    im = plt.imshow(values, interpolation='gaussian', cmap=cmap, aspect="auto", origin="lower", extent=[0, 500, 0, len(values)])
    axes.imshow(values, interpolation='gaussian', cmap=cmap, aspect="auto", origin="lower", extent=[0, 500, 0, len(values)])
    axes.axvline(first_break,label='Interpreted First Break',color='r')
    #lt.colorbar(im,orientation='horizontal')#,cax=f)
    axes.set_ylabel('Pseudo Frequency')
    
    ax2 = axes.twinx()
    ax2.plot(tim,raw,label='Raw',color='w',alpha=0.3)
    ax2.plot(tim,fil,label='Low-Pass Filter',color='w',alpha=0.8)
    ax2.axvline(first_break,label='Interpreted First Break',color='r')
    ax2.set_yticks([0.00002,0,-0.00002])
    ax2.set_yticklabels(['+','0','-'])
    ax2.set_ylabel("Amplitude")
    axes = plt.gca()
    axes.set_xlim([0,500])
    plt.tight_layout()
    plt.savefig('Figures/Example_Scalogram_iD_'+str(sel_trace_iD)+'_alt.png')
    plt.show()


def calc_iter(seismic_df):
    '''Uses Sesimic_df to identify unique stations based on easting and northings'''
    num_iter = seismic_df.groupby(['Northing', 'Easting']).ngroups
    combo_list = seismic_df.groupby(['Northing','Easting']).mean().reset_index()
    combo_list = combo_list[['Northing','Easting']]
    return(num_iter, combo_list)

def test_coppens(listoftraces,window):
    results=[]
    lad = []
    for i in range(len(listoftraces)):
        samples=listoftraces[i].Signal_1[1].values**2
        sub_samples=[]
        for i in range(int(len(samples))):
            sub_samples.append(samples[i:i+window])
        tau_window=[np.trapz(x) for x in sub_samples]
        normiliser=[]
        for i in range(len(tau_window)):
            normilizer_i=samples[:i+5]
            normiliser.append(np.trapz(normilizer_i))
        x=np.array(tau_window)/np.array(normiliser)
        x=x[50:]#trims off first 50 channels which always trigger
        results.append(np.argmax(x))
        lad.append(x)
    return(results, lad)

def notify():
    _16 = 125
    _08 = 250
    _04 = 500
    _02 = 1000
    _01 = 2000
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

def process_fb_calc(predicted_set,predictions):
    if len(predicted_set)!=len(predictions):
        raise Exception('Missmatched number of predictions and traces.')
    pred_val=[]
    real=[]
    pred_ser=[]
    tid=[]
    for i in range(len(predicted_set)):
        predicted_set_i=predicted_set[i]
        prediction_i=predictions[i].flatten()
        predicted_set_i.prediction_series[0]=prediction_i
        edited=[]
        for j in range(len(prediction_i)):
            iter_sel=max(prediction_i[:j+1])
            edited.append(iter_sel)
        #edited[:20]=np.zeros(20)
        predicted_set_i.prediction_series[1]=edited
        dif=np.diff(edited,1)
        area=np.trapz(dif)
        normalized = dif/area
        predicted_set_i.prediction_series[2]=normalized
        predicted_set_i.FB_Picks[2]=np.argmax(normalized)
        real.append(predicted_set_i.FB_Picks[1])
        pred_val.append(predicted_set_i.FB_Picks[2])
        pred_ser.append(prediction_i)
        tid.append(predicted_set_i.iD)
    return(pred_val,real,pred_ser,tid)

def convert_series(prediction_array):
        running_max=[]
        for j in range(len(prediction_array)):
            iter_sel=max(prediction_array[:j+1])
            running_max.append(iter_sel)
        dif=np.diff(running_max,1)
        area=np.trapz(dif)
        pseudo_cdf = dif/area
        return(pseudo_cdf)

def plot_uncertainty(selection,method):
    cet_cmap = matplotlib.colors.LinearSegmentedColormap.from_list('CET',cc.CET_L3, 256)
    x0 = np.array([trace.rec_loc[0] for trace in selection])
    y0 = np.array([trace.rec_loc[1] for trace in selection])
    z1 = np.array([trace.Unc_Metrics[0] for trace in selection])
    z2 = np.array([trace.Unc_Metrics[1] for trace in selection])
    z3 = np.array([trace.Unc_Metrics[2] for trace in selection])
    xi = np.linspace(x0.min(), x0.max(), 1000)
    yi = np.linspace(y0.min(), y0.max(), 1000)
    if method=='UM1':
        zi = griddata((x0, y0), z1, (xi[None,:], yi[:,None]), method='linear')
    if method=='UM2':
        zi = griddata((x0, y0), z2, (xi[None,:], yi[:,None]), method='linear')   
    if method=='UM3':
        zi = griddata((x0, y0), z3, (xi[None,:], yi[:,None]), method='linear')
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(15, 10))
    axes.set_xlabel('Easting')
    axes.set_ylabel('Northing')     
    plt.contourf(xi,yi,zi,15,cmap=cet_cmap)
    fig.canvas.draw()
    
def visualise_classifier(trace,comp_trace,all_classifiers):
    y=np.linspace(1,500,500)
    a=trace.Signal_1[0]
    b=trace.Signal_1[1]
    Post=int(trace.FB_Picks[1]*1000)
    
    plt.rcParams["font.family"] = "Times new roman"
    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(figsize=(20, 10))
    
    raw  = plt.plot(y,a,label='Raw',color='k',alpha=0.6)
    filt = plt.plot(y,b,label='Low-Pass Filter',color='k')

    ax.set_yticks([0.00003,0,-0.00003])
    ax.set_yticklabels(['+','0','-'])
    ax.set_ylabel("Amplitude")
    ax.set_xlabel('Time (ms)')
    
    ax2 = ax.twinx()
    classif=ax2.plot(np.append([np.zeros(Post)],[np.ones(500-Post)]),'--',color='red',label='Target Series')
    
    if all_classifiers == True:
        interp=ax2.axvline(Post,label='Interperter Prediction',color='r',linewidth=2)
        pred_series = trace.prediction_series['CVNN01']
        run_max = []
        for i in range(len(pred_series)):
            if i == 0:
                run_max.append(pred_series[0])
            if i >0:
                run_max.append(max(pred_series[:i]))
        
        dif=np.diff(run_max,1)
        area=np.trapz(dif)
        normalized = dif/area
        
        cdf  =  ax2.plot(run_max,color='b',label='CDF*')
        raw_p  =  ax2.plot(pred_series,'--',color='b',label='Returned Series')
        
        pdf  =  ax2.plot(normalized,'--',label='PDF',color='orange',linewidth=2.0)
        pred = ax2.axvline(np.argmax(normalized),label='Model Prediction',color='orange',linewidth=2)
        
    ax2.set_ylabel('Classification Series',rotation=270)
    
    lns = [raw,filt,classif,interp,raw_p,cdf,pdf,pred]
    lns = [line if not isinstance(line,list) else line[0] for line in lns]    
    labs = [l.get_label() for l in lns]
    
    ax2.set_yticks([0,1])
    ax2.set_yticklabels(['0','1'])
    lgd = ax.legend(lns, labs, loc='lower center', bbox_to_anchor=(0.5, -0.4),ncol=4, fancybox=True)
    plt.xlim([125,300])
    fig.set_size_inches(11, 5)
    fig.savefig('Figures/Example Classifcation Series.png',bbox_extra_artists=[lgd],bbox_inches='tight', dpi=500)
    
    
def plot_scalogram(sel_trace):
    fig, axes = plt.subplots(2,figsize=(9,8), sharex=True,dpi=100,gridspec_kw={'height_ratios': [1,2]})
    trace_i = sel_trace       
    raw = trace_i.Signal_1[0]
    fil = trace_i.Signal_1[1]  
    first_break = trace_i.FB_Picks[1]*1000
    tim = np.linspace(1,500,500)
    sel_trace_iD = trace_i.iD
    
    #performs wavelet transform
    wp = pywt.WaveletPacket(raw, 'db2', 'sym', maxlevel=5)
    nodes = wp.get_level(5, order="freq")
    labels = [n.path for n in nodes]
    values = np.array([n.data for n in nodes], 'd')
    values = abs(values)
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('CET',cc.CET_D1A, 256)
    
    axes[0].plot(tim,raw,label='Raw',color='k',alpha=0.6)
    axes[0].plot(tim,fil,label='Low-Pass Filter',color='k')
    axes[0].axvline(first_break,label='Interpreted First Break',color='r')
    axes[0].axhline(0,label='Interpreted First Break',color='k',ls=':')
    axes[0].text(460, (max(raw)-max(raw)*0.3), 'A', fontsize=35)
    axes[0].set_yticks([0.00002,0,-0.00002])
    axes[0].set_yticklabels(['+','0','-'])
    axes[0].set_ylabel("Amplitude")
    
    im = plt.imshow(values, interpolation='gaussian', cmap=cmap, aspect="auto", origin="lower", extent=[0, 500, 0, len(values)])
    axes[1].imshow(values, interpolation='gaussian', cmap=cmap, aspect="auto", origin="lower", extent=[0, 500, 0, len(values)])
    axes[1].axvline(first_break,label='Interpreted First Break',color='r')
    plt.colorbar(im,orientation='horizontal')#,cax=f)
    axes[1].set_ylabel('Scale')
    axes[1].text(460, 25, 'B', fontsize=35,color='white')
    
    plt.tight_layout()
    plt.savefig('Figures/Example_Scalogram_iD_'+str(sel_trace_iD)+'.png')
    plt.show()

def trace_comparison(comp_trace,sel_trace_iD):    
    #bad 35,38,57,60,84,88
    #good 137,157,164
    if len(sel_trace_iD)>3:
        print('Select fewer traces')
    if len(sel_trace_iD)<=3:
        labels = ['A','B','C']
        plt.rcParams["font.family"] = "Times New Roman"
        # setting up the plot
        fig, axes = plt.subplots(len(sel_trace_iD), sharex = True, sharey = True,figsize=(9,8))
        fig.add_subplot(111, frameon = False)
        plt.tick_params(labelcolor = 'none', top = False, bottom = False, left = False, right = False)
        plt.grid(False)
        plt.xlabel("Time (ms)")
        plt.ylabel("Amplitude ")
        # plotting
        for idx, ax in enumerate(axes):
            trace_i = [x for x in comp_trace if x.iD == sel_trace_iD[idx]][0]              
            raw = trace_i.Signal_1[0]
            fil = trace_i.Signal_1[1]  
            first_break = trace_i.FB_Picks[1]*1000
            tim = np.linspace(1,500,500)
            ax.plot(tim,raw,label='Raw',color='k',alpha=0.6)
            ax.plot(tim,fil,label='Low-Pass Filter',color='k')
            ax.axvline(first_break,label='Interpreted First Break',color='r')
            ax.axhline(0,label='Interpreted First Break',color='k',ls=':')
            #ax.legend(loc='lower right')
            ax.set_xlim((100,300))
            ax.set_yticks([0.00002,0,-0.00002])
            ax.set_yticklabels(['+','0','-'])
            ax.text(290, 0.000025, labels[idx], fontsize=20)
        plt.savefig('Figures/Trace_Comparison.png',dpi=700)



def custom_sort(t):
    return(t[1])

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r',flush=True)
    # Print New Line on Complete
    if iteration == total: 
        print()

def import_traces(num2import,fb_type):
    comp_trace=list()
    seismic_df=load_csv()
    l=num2import
    if isinstance(num2import, str):
        l=len(seismic_df)
    printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    for i in range(l):
        #load a sample
        iter_sample=seismic_df.iloc[i]
        #meta
        iD=i
        Date=iter_sample['date']
        Time=iter_sample['time'] 
        Station=iter_sample['Station']  
        Repeat=iter_sample['repeat']
        Atl_fb=iter_sample['afb']  
        Rio_fb=iter_sample['rfb'] 
        Gravty=iter_sample['Grav_sel'] 
        K_Type=iter_sample['K_Type']
        
        #spatial
        xr=iter_sample['xr']
        yr=iter_sample['yr']
        zr=iter_sample['zr']
        xs=iter_sample['xs']
        ys=iter_sample['ys']
        zs=iter_sample['zs']
        
        #signals
        Raw_1=iter_sample.loc['R1_000':'R1_499']
        Raw_2=iter_sample.loc['R2_000':'R2_499']
        Flt_1=iter_sample.loc['F1_000':'F1_499']
        Flt_2=iter_sample.loc['F2_000':'F2_499']
        
        trace_obj=trace(iD,Date,Time,Station,Repeat,Atl_fb,Rio_fb,Gravty,
                 K_Type,xr,yr,zr,xs,ys,zs,Raw_1,Flt_1,Raw_2,Flt_2)
        
        comp_trace.append(trace_obj)
        printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    return(comp_trace)

def method_results(Uncertainty_method,thresh,title):#####REQUIRES UPDATES    
    # make up some randomly distributed data
    threshold_m1=Uncertainty_method[Uncertainty_method[0] < thresh]
    x = threshold_m1['Easting'].values
    y = threshold_m1['Northing'].values
    z = threshold_m1[0].values
    # define grid.
    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)
    zi = np.linspace(y.min(), y.max(), 100)
    # grid the data.
    zi = griddata((x, y), z, (xi[None,:], yi[:,None]), method='cubic')
    # contour the gridded data, plotting dots at the randomly spaced data points.
    fig, ax = plt.subplots(figsize=(8, 10))
    ax =plt.subplot(211)
    CS = plt.contour(xi,yi,zi,15,linewidths=0.5,colors='k')
    CS = plt.contourf(xi,yi,zi,15,cmap=plt.cm.jet)
    plt.colorbar()
    plt.scatter(x,y,marker='o',c='k',s=5)
    plt.title(title[0])
    ax =plt.subplot(212)
    Uncertainty_m1[0].hist(bins=100)
    plt.axvline(thresh,label='User Specified Threshold',color='r')
    plt.title(title[1])
    plt.tight_layout()
    plt.show()
    
def area_plot(comp_trace):
    global X0,Y0,X1,Y1,iDs
    X0=[]
    Y0=[]
    iDs=[]
    for i in range(len(comp_trace)):
        Trace_Selection = next(trace for trace in comp_trace if trace.iD == i) #select a signle trace using a trace iD
        Y0.append(Trace_Selection.rec_loc[1])
        X0.append(Trace_Selection.rec_loc[0])
        iDs.append(Trace_Selection.iD)
    #Y1=step_out['MGA94NORTH_Receiver'].values
    #X1=step_out['MGA94EAST_Receiver'].values
    fig, ax = plt.subplots()
    ax.scatter(X0,Y0)
    #ax.scatter(X1,Y1)
    return(ax)
    
def line_select_callback(eclick, erelease):
    global X0,Y0,Y1,X1,filtered_X0,filtered_Y0,filtered_X1,filtered_Y1,filtered_iDs, step_out_selection
    
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    mask=(X0 > min(x1,x2)) & (X0 < max(x1,x2)) &\
    (Y0 > min(y1,y2)) & (Y0 < max(y1,y2))
    mask2=(X1 > min(x1,x2)) & (X1 < max(x1,x2)) &\
    (Y1 > min(y1,y2)) & (Y1 < max(y1,y2))
    
    filtered_iDs = [i for indx,i in enumerate(iDs) if mask[indx] == True]
    filtered_X0 = [i for indx,i in enumerate(X0) if mask[indx] == True]
    filtered_Y0 = [i for indx,i in enumerate(Y0) if mask[indx] == True]
    filtered_X1 = [i for indx,i in enumerate(X1) if mask2[indx] == True]
    filtered_Y1 = [i for indx,i in enumerate(Y1) if mask2[indx] == True]
    
    generate_selection(filtered_iDs,filtered_X1,filtered_Y1)
    #step_out_selection = step_out[(step_out['MGA94EAST_Receiver'] >= min(x1,x2)) 
                                  #& (step_out['MGA94EAST_Receiver'] <= max(x1,x2))]
    #step_out_selection = step_out_selection[(step_out_selection['MGA94NORTH_Receiver'] >= min(y1,y2)) 
                                  #& (step_out_selection['MGA94NORTH_Receiver'] <= max(y1,y2))]
    
def generate_selection(filtered_iDs,filtered_X1,filtered_Y1):
    global area_selection
    global comp_trace
    #using the slected iD's generates a list of trace2 objects
    area_selection=[]
    for i in range(len(filtered_iDs)):
        h=filtered_iDs[i]
        Trace_Selection = next(trace for trace in comp_trace if trace.iD == h) #select a single trace using a trace iD
        area_selection.append(Trace_Selection)

def model_plot(gather,suptitle):
    #data series extraction
    y=np.linspace(1,500,500)*-1
    filt=[]
    raw=[]
    jup_pick=[]
    title=["K1","K2","K3"]
    ni=len(gather)
    for i in range(ni):
        filt.append(gather[i].filt)
        raw.append(gather[i].raw)
        jup_pick.append(gather[i].jup_pick)
    #plotting setup
    fig, axes = plt.subplots(nrows=1,ncols=ni)
    for i in range(ni):
        axes[i].plot(raw[i],y,color='orange',alpha=0.6)
        axes[i].plot(filt[i],y)
        state=gather[i].ml_pred
        axes[i].set_title(title[i])
        axes[i].set_xticks([])
        if jup_pick[i]!=None:
            time=jup_pick[i]
            axes[i].axhline(time*-1,label='Jupyter Pick',color='red')
        if isinstance(state, np.ndarray):
            #axes[i].set_yticks([])
            sag=axes[i].twiny()
            sag.plot(state,y,color='orange')
            sag.set_xticks([])
        if state is None:
            print('No predictions yet.')
            #axes[i].axhline(gather[i].jup_pick*-1,label='Jupyter Pick',color='green')
    fig.suptitle(suptitle, fontsize=16)

def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def keras_input(learn_set,feat_num):
    num_samples=len(learn_set)
    signal_set=[]
    target_a_set=[]
    target_b_set=[]
    for i in range(num_samples):
        training=learn_set[i].feat_space.fillna(0)
        training=training.values
        signal_sel=training[:,:feat_num]
        signal_set.append(normalize(signal_sel,axis=-1,order=2))
        target_a_set.append(training[:,35])
        target_b_set.append(training[:,36])
    signal_set=np.stack(signal_set)
    target_a_set=np.stack(target_a_set)
    target_b_set=np.stack(target_b_set)
    return(signal_set,target_a_set,target_b_set)

def plot_violin(dataframe):
    plt.rcParams.update({'font.size': 18})
    plt.rcParams["font.family"] = "Times new roman"
    fig, ax = plt.subplots(figsize=(15,8))
    dataframe = dataframe[(dataframe['Feat Space'] != '1')]
    # make a vector of color: red for the interesting group, blue for others:
    ax = sns.violinplot(x='Model', y='Variance',data=dataframe,split=True,hue="Feat Space", palette="Blues_d",scale="count",bw=.2,inner="quartile")
    ax.yaxis.grid(True)
    ax.xaxis.grid(False)
    plot_title="Distribution of Prediction Residual"
    plt.title('Distribution of Prediction Residuals')
    plt.tight_layout()
    ax.set(ylabel='Prediction Residuals (ms)')
    ax.set(xlabel='')
    ax.set_xticklabels(['Coppens','NN','CNN','LSTM'])
    plt.rcParams['axes.labelweight'] = 'normal'
    plt.savefig('Figures/Prediction_Residuals.png',dpi=700)
    return(ax)

def convert_series(prediction_array):
        running_max=[]
        for j in range(len(prediction_array)):
            iter_sel=max(prediction_array[:j+1])
            running_max.append(iter_sel)
        dif=np.diff(running_max,1)
        area=np.trapz(dif)
        pseudo_pdf = dif/area
        return(pseudo_pdf)

def confidence_calcs(pred_array,model_array):
    confidence = []
    for i in range(len(model_array)):
        prediction_series = model_array[i]
        prediction_val = pred_array[i]
        if not np.isnan(prediction_series).any():
            if prediction_val != 0 and prediction_val !=500:
                prediction_window = prediction_series[prediction_val-2:prediction_val+2]
            if prediction_val == 0:
                prediction_window = prediction_series[:5]
            if prediction_val == 500:
                prediction_window = prediction_series[495:]
            mods = np.trapz(abs(prediction_window))/np.trapz(abs(prediction_series))
        if np.isnan(prediction_series).any():
            mods =0
        confidence.append(mods)
    return(confidence)

# query test
def plot_viola(test):
    data_frame = []
    
    for i in range(len(test)):
        trace_i = test[i]
        keys = list(trace_i.prediction__value.keys())
        for key in keys:        
            model = key[:4]
            
            flag=False
            if 'Co' in key:
                flag=True
            if '01' in key:
                feats = '01'
            if '02' in key:
                feats = '02'
            if '33' in key:
                feats = '33'
                
            value = trace_i.prediction__value[key]
            trace = trace_i.iD
            first = int(trace_i.FB_Picks[1]*1000)
            
            if flag == True:        
                data_frame.append({'Model':model,'Feat Space':'01','Predicted First Break':value,'Trace ID':trace,'RFB':first})
                data_frame.append({'Model':model,'Feat Space':'33','Predicted First Break':value,'Trace ID':trace,'RFB':first})
            if flag != True:
                data_frame.append({'Model':model,'Feat Space':feats,'Predicted First Break':value,'Trace ID':trace,'RFB':first})
        
    final_df = pd.DataFrame(data_frame)   
    variance = final_df["Predicted First Break"] - final_df["RFB"]
    final_df["Variance"] = variance
    return(final_df)

### FLAGGED FOR REMOVAL


def save_predictions(predictions,test_set,feat,model):
    (pred_val1,real2,pred_ser1,tid1)=process_fb_calc(test_set,predictions)
    real1=[int(x.FB_Picks[1]*1000) for x in test_set]
    labelsssss=np.arange(1,501)
    labelsssss=[str(x) for x in labelsssss]
    for i in range(len(pred_ser1)):
        a=pred_ser1[i]
        df = pd.DataFrame(a.reshape(-1, len(a)),columns=labelsssss)
        if i==0:
            total=df
        if i !=0:
            total=total.append(df)
    all_results_long1={}
    all_results_long1['Model']=[model]*len(pred_val1)
    all_results_long1['Feat Space']=[feat]*len(pred_val1)
    all_results_long1['Predicted First Break']=pred_val1
    all_results_long1['Trace ID']=tid1
    all_results_long1['RFB']=real1
    long_DF=pd.DataFrame.from_dict(all_results_long1)
    pickle.dump(total, open(model+feat+'_Series.p', "wb" ))
    return(long_DF)

def process_fb_calc(predicted_set,predictions):
    if len(predicted_set)!=len(predictions):
        raise Exception('Missmatched number of predictions and traces.')
    pred_val=[]
    real=[]
    pred_ser=[]
    tid=[]
    for i in range(len(predicted_set)):
        predicted_set_i=predicted_set[i]
        prediction_i=predictions[i].flatten()
        predicted_set_i.prediction_series[0]=prediction_i
        edited=[]
        for j in range(len(prediction_i)):
            iter_sel=max(prediction_i[:j+1])
            edited.append(iter_sel)
        #edited[:20]=np.zeros(20)
        predicted_set_i.prediction_series[1]=edited
        dif=np.diff(edited,1)
        area=np.trapz(dif)
        normalized = dif/area
        predicted_set_i.prediction_series[2]=normalized
        predicted_set_i.FB_Picks[2]=np.argmax(normalized)
        real.append(predicted_set_i.FB_Picks[1])
        pred_val.append(predicted_set_i.FB_Picks[2])
        pred_ser.append(prediction_i)
        tid.append(predicted_set_i.iD)
    return(pred_val,real,pred_ser,tid)

