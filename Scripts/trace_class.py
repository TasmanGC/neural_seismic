class trace:
    
    def __init__(self, iD,Date,Time,Station,Repeat,Atl_fb,Rio_fb,Gravty,
                 K_type,xr,yr,zr,xs,ys,zs,Raw_1,Flt_1,Raw_2,Flt_2):
        # meta data
        self.iD = iD
        self.Date = Date
        self.Time = Time
        self.Station = Station
        self.Repeat = Repeat
        self.FB_Picks = [Atl_fb,Rio_fb,None]
        self.Gravty = Gravty
        self.K_Type = K_type
        self.Unc_Metrics=None
        self.feat_space=None
        
        #spatial
        self.rec_loc=[xr,yr,zr]
        self.sou_loc=[xs,ys,zs]
        
        # signals
        self.Signal_1=[Raw_1,Flt_1]
        if Repeat==True:
            self.Signal_2=[Raw_2,Flt_2]
        if Repeat==False:
            self.Signal_2=[None,None]
        
        self.prediction_series={'Coppens':None,
                                'BPNN01':None,
                                'BPNN02':None,
                                'BPNN33':None,
                                'CVNN01':None,
                                'CVNN02':None,
                                'CVNN33':None,
                                'LSTM01':None,
                                'LSTM02':None,
                                'LSTM33':None,
                               }
        self.prediction__value={'Coppens':None,
                                'BPNN01':None,
                                'BPNN02':None,
                                'BPNN33':None,
                                'CVNN01':None,
                                'CVNN02':None,
                                'CVNN33':None,
                                'LSTM01':None,
                                'LSTM02':None,
                                'LSTM33':None,
                               }
        self.FB_Window = None 

    def plot_comp(self,title):#Updating
        y=np.linspace(1,500,500)
        a=self.Signal_1[0]
        b=self.Signal_1[1]
        atlas=self.FB_Picks[0]*1000
        rio=self.FB_Picks[1]*1000
        stn_num=self.Station
        plt.rcParams.update({'font.size': 10})
        #title=('Station Number = '+str(stn_num)+': '+self.K_Type+': Filter Comparison')
        plt.title(title)
        plt.plot(y,a,label='Raw',color='k',alpha=0.6)
        plt.plot(y,b,label='Low-Pass Filter',color='k')
        plt.axvline(rio,label='Interpreted First Break',color='r')
        #plt.axvline(atlas,label='Atlas First Break',color='b')
        #if self.FB_Picks[2]!=None:
        #    ML_Pick=self.FB_Picks[2]
        #    plt.axvline(atlas,label='ML Pick',color='crimson')
        plt.legend(loc='lower right')
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.xlim((0,300))
        fig = plt.gcf()
        fig.set_size_inches(11, 5)
        fig.savefig('Plots/Trace_Filter_Comparison.png', dpi=100)
        
    def plot_scalo2(self,title):
        data=self.Signal_1[0]
        filt=self.Signal_1[1]
        stn_num=self.Station
        x=np.linspace(1,500,500)
        wavelet = 'db2'
        level = 5
        wp = pywt.WaveletPacket(data, wavelet, 'sym', maxlevel=level)
        order = "freq"  # "normal"
        nodes = wp.get_level(level, order=order) #automatically performs decomp
                                                 # until achieves max level
        labels = [n.path for n in nodes]
        values = np.array([n.data for n in nodes], 'd')
        values = abs(values)
        f = plt.figure()
        f.subplots_adjust(hspace=0.25, bottom=.03, left=.07, right=.97, top=.92)
        plt.subplot(3, 1, 1)
        #title=('Station Number = '+str(stn_num)+' : '+self.K_Type)
        plt.title(title)
        plt.plot(x, data, 'k',alpha=0.6,label='Raw Signal')
        plt.plot(x,filt,label='Low-Pass Filter',color='k')
        plt.xlim(0, x[-1])
        #atlas=self.FB_Picks[0]*1000
        rio=self.FB_Picks[1]*1000
        plt.axvline(rio,label='Interpreted First Break',color='r')
        #plt.axvline(atlas,label='Contractor First Break',color='b')
        plt.legend(loc=1)   
        plt.xlabel('Time (ms)')
        plt.ylabel('Amplitude')
        plt.subplot2grid((3, 1),(1,0),rowspan=2)
        #f.subplots_adjust(hspace=0.25, bottom=.03, left=.07, right=.97, top=.92)
        interpolation = 'gaussian'
        cmap = plt.cm.PiYG
        plt.title("Scalogram")
        im=plt.imshow(values, interpolation=interpolation, cmap=cmap, aspect="auto", origin="lower", extent=[0, 500, 0, len(values)])
        plt.axvline(rio,label='Interpreted First Break',color='r')
        plt.colorbar(im,orientation='horizontal')#,cax=f)
        plt.xlabel('Time (ms)')
        plt.ylabel('Pseudo Frequency')
        fig = f
        fig.set_size_inches(9,9)
        plt.tight_layout()
        plt.savefig('Final Figures/'+title+'Example_Scalogram.png')
        plt.show()
    
    
    def plot_scalo(self,state):
        data=self.Signal_1[0]
        filt=self.Signal_1[1]
        stn_num=self.Station
        x=np.linspace(1,500,500)
        wavelet = 'db2'
        level = 5
        wp = pywt.WaveletPacket(data, wavelet, 'sym', maxlevel=level)
        order = "freq"  # "normal"
        nodes = wp.get_level(level, order=order) #automatically performs decomp
                                                 # until achieves max level
        labels = [n.path for n in nodes]
        values = np.array([n.data for n in nodes], 'd')
        values = abs(values)
        if state>0:
            f = plt.figure()
            f.subplots_adjust(hspace=0.25, bottom=.03, left=.07, right=.97, top=.92)
            plt.subplot(2, 1, 1)
            title=('Station Number = '+str(stn_num)+' : '+self.K_Type)
            plt.title(title)
            plt.plot(x, data, 'k',alpha=0.6,label='Raw Signal')
            plt.plot(x,filt,label='OM - Low-Pass Filter',color='k')
            plt.xlim(0, x[-1])
            atlas=self.FB_Picks[0]*1000
            rio=self.FB_Picks[1]*-1
            plt.axvline(rio,label='Company First Break',color='r')
            plt.axvline(atlas,label='Contractor First Break',color='b')
            plt.legend(loc=1)   
            
            ax = plt.subplot(2, 1, 2)
            interpolation = 'gaussian'
            cmap = plt.cm.PiYG
            plt.title("Wavelet packet coefficients at level %d" % level)
            plt.imshow(values, interpolation=interpolation, cmap=cmap, aspect="auto", origin="lower", extent=[0, 500, 0, len(values)])
            fig = f
            fig.set_size_inches(7, 7)
            plt.tight_layout()
            plt.savefig('Plots/Trace_Wavelet_Scalogram.png')
            plt.show()
            plt.savefig('Wavelet.eps', format='eps')
            return(values)
        else:
            return(values)
        
    def calc_metrics(self):
        #Uncertainty Metric 1 - Operator Pick Disparity
        UM1=self.FB_Picks[0]-self.FB_Picks[1]
        
        #Uncertainty Metric 2 - Noise to signal ratio
        noise_it = (np.mean(np.absolute(self.Signal_1[0])))
        signl_it = (np.mean(np.absolute(self.Signal_1[1])))
        UM2 = (signl_it/noise_it)
        
        #Uncertainty Metric 3 - First Break Clarity
        UM3_sub=[None,None]
        NumberTypes = (int, float)
        if isinstance(self.FB_Picks[0],NumberTypes):
            atl = int(self.FB_Picks[0]*1000)
        if not isinstance(self.FB_Picks[0],NumberTypes):
            atl = 0
        if isinstance(self.FB_Picks[1],NumberTypes):    
            rio = int(self.FB_Picks[1]*-1)
        if not isinstance(self.FB_Picks[1],NumberTypes):    
            rio = 0
            
        atl=int(self.FB_Picks[0]*1000)
        rio=int(self.FB_Picks[1]*1000)
            
        if rio!=0:
            rio_noise_it_pre =(np.mean(np.absolute(self.Signal_1[0][rio-25:rio])))
            rio_signal_it_pre =(np.mean(np.absolute(self.Signal_1[1][rio-25:rio])))
            rio_noise_it_post =(np.mean(np.absolute(self.Signal_1[0][rio:rio+25])))
            rio_signal_it_post =(np.mean(np.absolute(self.Signal_1[1][rio:rio+25])))
            mean_rio_pre=(rio_noise_it_pre+rio_signal_it_pre)/2
            mean_rio_post=(rio_noise_it_post+rio_signal_it_post)/2
            rio_entropy=mean_rio_pre/mean_rio_post
            UM3_sub[0]=rio_entropy
            
        if atl!=0:
            atl_noise_it_pre =(np.mean(np.absolute(self.Signal_1[0][atl-25:atl])))
            atl_signal_it_pre =(np.mean(np.absolute(self.Signal_1[1][atl-25:atl])))
            atl_noise_it_post =(np.mean(np.absolute(self.Signal_1[0][atl:atl+25])))
            atl_signal_it_post =(np.mean(np.absolute(self.Signal_1[1][atl:atl+25])))
            mean_atl_pre=(atl_noise_it_pre+atl_signal_it_pre)/2
            mean_atl_post=(atl_noise_it_post+atl_signal_it_post)/2
            atl_entropy=mean_atl_pre/mean_atl_post
            UM3_sub[1]=atl_entropy
        self.Unc_Metrics=[UM1,UM2,UM3_sub]
    
    def gen_feat_space(self):
        """This function generates a feature space for binary classification.
        input(trace_object)+input(int)-->output(pandas)"""
        ### Names all the columns note if adding new variable please add to this list
        column_headers=['Raw','Filt','WC1','WC2','WC3','WC4','WC5','WC6',
                        'WC7','WC8','WC9','WC10','WC11','WC12', 'WC13',
                        'WC14','WC15','WC16','WC17','WC18','WC19', 'WC20',
                        'WC21','WC22','WC23','WC24','WC25','WC26', 'WC27',
                        'WC28','WC29','WC30','WC31','WC32','Pred','Rio_FB','Atlas_FB']
        
        ### generates the dataframe
        feat_space=pd.DataFrame(data=None, columns=column_headers, index=range(0,500))
        ### adds the raw and filt column values
        raw=self.Signal_1[0]
        raw=np.asarray(raw)
        feat_space["Raw"]=raw
        filt=self.Signal_1[1]
        filt=np.asarray(filt)
        feat_space["Filt"]=filt
        
        data=raw
        wavelet = 'db2'
        level = 5
        wp = pywt.WaveletPacket(data, wavelet, 'sym', maxlevel=level)
        order = "freq"  # "normal"
        nodes = wp.get_level(level, order=order)
        labels = [n.path for n in nodes]
        values = np.array([n.data for n in nodes], 'd')
        values = abs(values)
        values=self.plot_scalo(0)
        num_lev,lev_len = values.shape
        ### uses radial basis function to process the values output by DWT 
        for a in range(num_lev):
            #for each coefiecent level
            Y_val=[]
            values_iter=values[a]
            Y_val.extend(values_iter)
            X_val=[]
            for b in range(lev_len):
                X=int((b)*(500/(lev_len)))
                X_val.append(X)
            #interpolates between discrete values of DWT using radial basis function
            xnew = np.linspace(0, 472, num=472, endpoint=True)
            rbf = Rbf(X_val, Y_val)
            fi = rbf(xnew)
            extrap_rbf=interp([500], xnew, fi)
            X_val_2=[]
            X_val_2.extend(X_val[:])
            X_val_2.append(500)
            Y_val_rbf=[]
            Y_val_rbf.extend(Y_val[:])
            Y_val_rbf.append(extrap_rbf[0])
            xfinal = np.linspace(0, 500, num=500, endpoint=True)
            rbf = Rbf(X_val_2, Y_val_rbf)
            #interpolated coeficient for given level           
            coeficient_values = rbf(xfinal)
            feat_space[column_headers[a+2]]=coeficient_values
        
        ### uses the operator first-break pick to generate classifier
        Rio_fb=int(self.FB_Picks[1]*1000)
        Atlas_fb=int(self.FB_Picks[0]*1000)
        
        if Rio_fb!=None:
            ones=np.ones(((500-Rio_fb),), dtype=np.int)
            zeros=np.zeros((Rio_fb,), dtype=np.int)
            method2=np.append(zeros,ones)
            feat_space[column_headers[35]]=method2
        if Atlas_fb!=None:
            ones=np.ones(((500-Atlas_fb),), dtype=np.int)
            zeros=np.zeros((Atlas_fb,), dtype=np.int)
            method2=np.append(zeros,ones)
            feat_space[column_headers[36]]=method2
            
        feat_space=(feat_space-feat_space.min())/(feat_space.max()-feat_space.min())
        
        self.feat_space=feat_space
        
    
    
        