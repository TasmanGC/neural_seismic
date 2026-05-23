# Generator for series vale prediction
def series_data_gen(batch_T_obj,feat_num):
    samples_per_epoch=len(batch_T_obj)
    counter=0
    while 1:
        iter_trace=batch_T_obj[counter]
        iter_trace.gen_feat_space()
        training=iter_trace.feat_space.fillna(0)
        training=training.values
        signal_sel=training[:,:feat_num]
        #signal_sel=normalize(signal_sel,axis=-1,order=2)
        target=training[:,35]
        signal_sel=np.stack(signal_sel)
        shape=signal_sel.shape
        signal_sel=signal_sel.reshape((1,shape[0],shape[1]))
        target=np.stack(target)
        target=target.reshape((1,500,1))
        counter +=1
        yield(signal_sel,target)
        if counter>=samples_per_epoch:
            counter=0
            
            
# BPNN MODEL MAKERS
def BPNN_2(hidden_neuron,n_hidden,optim):
    modelexp = Sequential()
    modelexp.add(Dense(50, input_shape=(500, 2),activation=None))
    for i in range(n_hidden):
        modelexp.add(Dense(hidden_neuron))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)
def BPNN_1(hidden_neuron,n_hidden,optim):
    modelexp = Sequential()
    modelexp.add(Dense(50, input_shape=(500, 1),activation=None))
    for i in range(n_hidden):
        modelexp.add(Dense(hidden_neuron))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)

def BPNN_33(hidden_neuron,n_hidden,optim):
    modelexp = Sequential()
    modelexp.add(Dense(50, input_shape=(500, 33),activation=None))
    for i in range(n_hidden):
        modelexp.add(Dense(hidden_neuron))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)

# CONV1D MODEL MAKERS
def CONV_1(hidden_neuron,n_hidden,optim):    
    modelexp = Sequential()
    modelexp.add(Conv1D(10,kernel_size=20,padding="same", input_shape=(500,1),activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500,1)))
    modelexp.add(Conv1D(50,kernel_size=20,padding="same",activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500,1)))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)

def CONV_2(hidden_neuron,n_hidden,optim):    
    modelexp = Sequential()
    modelexp.add(Conv1D(10,kernel_size=20,padding="same", input_shape=(500,2),activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500,1)))
    modelexp.add(Conv1D(10,kernel_size=20,padding="same",activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500,1)))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)

def CONV_33(hidden_neuron,n_hidden,optim):     
    modelexp = Sequential()
    modelexp.add(Conv1D(10,kernel_size=20,padding="same", input_shape=(500,33),activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500,1)))
    modelexp.add(Conv1D(50,kernel_size=20,padding="same",activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500,1)))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)

def LSTM_conv_33(hidden_neuron,n_hidden,optim):     
    modelexp = Sequential()
    modelexp.add(LSTM(10, input_shape=(500, 33),return_sequences=True,activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500,1)))
    modelexp.add(LSTM(10, input_shape=(500, 33),return_sequences=True,activation=None))
    modelexp.add(MaxPooling1D(pool_size=20))
    modelexp.add(Dense(20))
    modelexp.add(Flatten())
    modelexp.add(Reshape((500,1)))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)

# LSTM MODEL MAKERS
def LSTM_1(hidden_neuron,n_hidden,optim):    
    modelexp = Sequential()
    modelexp.add(LSTM(20, input_shape=(500, 1),return_sequences=True,activation=None,unit_forget_bias=True))
    for i in range(n_hidden):
        modelexp.add(LSTM(hidden_neuron,input_shape=(500, 2),return_sequences=True,unit_forget_bias=True))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)

def LSTM_2(hidden_neuron,n_hidden,optim):    
    modelexp = Sequential()
    modelexp.add(LSTM(20, input_shape=(500, 2),return_sequences=True,activation=None,unit_forget_bias=True))
    for i in range(n_hidden):
        modelexp.add(LSTM(hidden_neuron,input_shape=(500, 2),return_sequences=True,unit_forget_bias=True))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)
    
def LSTM_33(hidden_neuron,n_hidden,optim):    
    modelexp = Sequential()
    modelexp.add(LSTM(20, input_shape=(500, 33),return_sequences=True,activation=None,unit_forget_bias=True))
    for i in range(n_hidden):
        modelexp.add(LSTM(hidden_neuron,input_shape=(500, 2),return_sequences=True,unit_forget_bias=True))
    modelexp.add(Dense(1))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)

#class EarlyStoppingThresh(Callback):
#    def __init__(self, monitor='acc', value=0.00001, verbose=0):
#        super(Callback, self).__init__()
#        self.monitor = monitor
#        self.value = value
#        self.verbose = verbose
#
#    def on_epoch_end(self, epoch, logs={}):
#        current = logs.get(self.monitor)
#        if current is None:
#            warnings.warn("Selected metric %s , is not available!" % self.monitor, RuntimeWarning)
#
#        if current > self.value:
#            if self.verbose > 0:
#                print("Epoch %05d: early stopping THR" % epoch)
#            self.model.stop_training = True
#            
            
            
### LSTM BUT WITH POOLING
def LSTMPOOL_2(hidden_neuron,optim):    
    modelexp = Sequential()
    modelexp.add(LSTM(20, input_shape=(500, 2),return_sequences=True,activation=None))
    modelexp.add(LSTM(20,unit_forget_bias=True,return_sequences=True))
    modelexp.add(GlobalAveragePooling1D('channels_first'))
    modelexp.add(Reshape((500,1)))
    modelexp.add(LSTM(20,unit_forget_bias=True,return_sequences=True))
    modelexp.add(GlobalAveragePooling1D('channels_first'))
    modelexp.add(Reshape((500,1)))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)

def LSTMPOOL_33(hidden_neuron,optim):    
    modelexp = Sequential()
    modelexp.add(LSTM(20, input_shape=(500, 33),return_sequences=True,activation=None))
    modelexp.add(LSTM(20,unit_forget_bias=True,return_sequences=True))
    modelexp.add(GlobalAveragePooling1D('channels_first'))
    modelexp.add(Reshape((500,1)))
    modelexp.add(LSTM(20,unit_forget_bias=True,return_sequences=True))
    modelexp.add(GlobalAveragePooling1D('channels_first'))
    modelexp.add(Reshape((500,1)))
    modelexp.compile(loss='mean_absolute_error',optimizer=optim,metrics=['accuracy'])
    return(modelexp)


# runs everything

def run_all_models(train,vali,test):
    # standard neural networks
    BP_NN_1 = BPNN_1(10,10,'Adamax')
    BP_NN_1.summary()
    BP_01_hist = BP_NN_1.fit_generator(series_data_gen(train,1),steps_per_epoch=100,epochs=len(train)/100)
    #,validation_data=series_data_gen(vali,1),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 10 mins for 7500 samples
    BP_NN_1_PRED = BP_NN_1.predict_generator(series_data_gen(test,1),steps=len(test)) # takes <5 mins
    
    BP_NN_2 = BPNN_2(10,10,'Adamax')
    BP_NN_2.summary()
    BP_02_hist=BP_NN_2.fit_generator(series_data_gen(train,2),steps_per_epoch=100,epochs=len(train)/100)
    #,validation_data=series_data_gen(vali,2),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 10 mins for 7500 samples
    BP_NN_2_PRED = BP_NN_2.predict_generator(series_data_gen(test,2),steps=len(test)) # takes <5 mins
    
    BP_NN_33 = BPNN_33(10,10,'Adamax')
    BP_NN_33.summary()
    BP_33_hist=BP_NN_33.fit_generator(series_data_gen(train,33),steps_per_epoch=100,epochs=len(train)/100)
    #,validation_data=series_data_gen(vali,33),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 10 mins for 7500 samples
    BP_NN_33_PRED = BP_NN_33.predict_generator(series_data_gen(test,33),steps=len(test)) # takes <5 mins

    # convolutional models
    CV_NN_1 = CONV_1(10,10,'Adamax')
    CV_NN_1.summary()
    CV_NN_1_hist = CV_NN_1.fit_generator(series_data_gen(train,1),steps_per_epoch=100,epochs=len(train)/100)
    #,validation_data=series_data_gen(vali,1),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 15 mins for 7500 samples DOESNT WORK
    CV_NN_1_PRED = CV_NN_1.predict_generator(series_data_gen(test,1),steps=len(test))
    
    CV_NN_2 = CONV_2(10,10,'Adamax')
    CV_NN_2.summary()
    CV_NN_2_hist = CV_NN_2.fit_generator(series_data_gen(train,2),steps_per_epoch=100,epochs=len(train)/100)
    #,validation_data=series_data_gen(vali,2),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 15 mins for 7500 samples DOESNT WORK
    CV_NN_2_PRED = CV_NN_2.predict_generator(series_data_gen(test,2),steps=len(test))
    
    CV_NN_33 = CONV_33(10,10,'Adamax')
    CV_NN_33.summary()
    CV_NN_33_hist = CV_NN_33.fit_generator(series_data_gen(train,33),steps_per_epoch=100,epochs=len(train)/100)
    #,validation_data=series_data_gen(vali,33),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 15 mins for 7500 samples DOESNT WORK
    CV_NN_33_PRED = CV_NN_33.predict_generator(series_data_gen(test,33),steps=len(test))
    
    # LSTM
    LS_TM_1 = LSTM_1(10,10,'Adamax')
    LS_TM_1.summary()
    LS_TM_1_hist = LS_TM_1.fit_generator(series_data_gen(train,1),steps_per_epoch=100,epochs=len(train)/100)
    #,validation_data=series_data_gen(vali,1),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 40 mins for 7500 samples
    LS_TM_1_PRED = LS_TM_1.predict_generator(series_data_gen(test,1),steps=len(test))
    
    LS_TM_2 = LSTM_2(10,10,'Adamax')
    LS_TM_2.summary()
    LS_TM_2_hist = LS_TM_2.fit_generator(series_data_gen(train,2),steps_per_epoch=100,epochs=len(train)/100)
    #,validation_data=series_data_gen(vali,2),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 40 mins for 7500 samples
    LS_TM_2_PRED = LS_TM_2.predict_generator(series_data_gen(test,2),steps=len(test))
    
    LS_TM_33 = LSTM_33(10,10,'Adamax')
    LS_TM_33.summary()
    LS_TM_33_hist = LS_TM_33.fit_generator(series_data_gen(train,33),steps_per_epoch=100,epochs=len(train)/100)
    #,validation_data=series_data_gen(vali,33),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60])
    LS_TM_33_PRED = LS_TM_33.predict_generator(series_data_gen(test,33),steps=len(test))
   
    # these are the models which after running are fully trained
    models = [BP_NN_1, BP_NN_2, BP_NN_33,
              CV_NN_1, CV_NN_2, CV_NN_33,
              LS_TM_1, LS_TM_2, LS_TM_33]
    # all predictions on input test data
    predictions = [BP_NN_1_PRED, BP_NN_2_PRED,BP_NN_33_PRED,
                   CV_NN_1_PRED, CV_NN_2_PRED,CV_NN_33_PRED, 
                   LS_TM_1_PRED, LS_TM_2_PRED,LS_TM_33_PRED]
    return(models,predictions)



def run_all_models_consis(train,vali,test):
    # standard neural networks
    BP_NN_1 = BPNN_1(10,1,'Adamax')
    BP_NN_1.summary()
    BP_01_hist = BP_NN_1.fit_generator(series_data_gen(train,1),steps_per_epoch=100,epochs=5)
    #,validation_data=series_data_gen(vali,1),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 10 mins for 7500 samples
    BP_NN_1_PRED = BP_NN_1.predict_generator(series_data_gen(test,1),steps=len(test)) # takes <5 mins
    
    BP_NN_2 = BPNN_2(10,1,'Adamax')
    BP_NN_2.summary()
    BP_02_hist=BP_NN_2.fit_generator(series_data_gen(train,2),steps_per_epoch=100,epochs=5)
    #,validation_data=series_data_gen(vali,2),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 10 mins for 7500 samples
    BP_NN_2_PRED = BP_NN_2.predict_generator(series_data_gen(test,2),steps=len(test)) # takes <5 mins
    
    BP_NN_33 = BPNN_33(10,1,'Adamax')
    BP_NN_33.summary()
    BP_33_hist=BP_NN_33.fit_generator(series_data_gen(train,33),steps_per_epoch=100,epochs=5)
    #,validation_data=series_data_gen(vali,33),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 10 mins for 7500 samples
    BP_NN_33_PRED = BP_NN_33.predict_generator(series_data_gen(test,33),steps=len(test)) # takes <5 mins

    # convolutional models
    CV_NN_1 = CONV_1(10,1,'Adamax')
    CV_NN_1.summary()
    CV_NN_1_hist = CV_NN_1.fit_generator(series_data_gen(train,1),steps_per_epoch=100,epochs=5)
    #,validation_data=series_data_gen(vali,1),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 15 mins for 7500 samples DOESNT WORK
    CV_NN_1_PRED = CV_NN_1.predict_generator(series_data_gen(test,1),steps=len(test))
    
    CV_NN_2 = CONV_2(10,1,'Adamax')
    CV_NN_2.summary()
    CV_NN_2_hist = CV_NN_2.fit_generator(series_data_gen(train,2),steps_per_epoch=100,epochs=5)
    #,validation_data=series_data_gen(vali,2),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 15 mins for 7500 samples DOESNT WORK
    CV_NN_2_PRED = CV_NN_2.predict_generator(series_data_gen(test,2),steps=len(test))
    
    CV_NN_33 = CONV_33(10,1,'Adamax')
    CV_NN_33.summary()
    CV_NN_33_hist = CV_NN_33.fit_generator(series_data_gen(train,33),steps_per_epoch=100,epochs=5)
    #,validation_data=series_data_gen(vali,33),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 15 mins for 7500 samples DOESNT WORK
    CV_NN_33_PRED = CV_NN_33.predict_generator(series_data_gen(test,33),steps=len(test))
    
    # LSTM
    LS_TM_1 = LSTM_1(10,1,'Adamax')
    LS_TM_1.summary()
    LS_TM_1_hist = LS_TM_1.fit_generator(series_data_gen(train,1),steps_per_epoch=100,epochs=5)
    #,validation_data=series_data_gen(vali,1),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 40 mins for 7500 samples
    LS_TM_1_PRED = LS_TM_1.predict_generator(series_data_gen(test,1),steps=len(test))
    
    LS_TM_2 = LSTM_2(10,1,'Adamax')
    LS_TM_2.summary()
    LS_TM_2_hist = LS_TM_2.fit_generator(series_data_gen(train,2),steps_per_epoch=100,epochs=5)
    #,validation_data=series_data_gen(vali,2),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60]) # takes about 40 mins for 7500 samples
    LS_TM_2_PRED = LS_TM_2.predict_generator(series_data_gen(test,2),steps=len(test))
    
    LS_TM_33 = LSTM_33(10,1,'Adamax')
    LS_TM_33.summary()
    LS_TM_33_hist = LS_TM_33.fit_generator(series_data_gen(train,33),steps_per_epoch=100,epochs=5)
    #,validation_data=series_data_gen(vali,33),validation_steps=len(vali))#,validation_freq=[5,10,15,20,25,30,40,50,60])
    LS_TM_33_PRED = LS_TM_33.predict_generator(series_data_gen(test,33),steps=len(test))
   
    # these are the models which after running are fully trained
    models = [BP_NN_1, BP_NN_2, BP_NN_33,
              CV_NN_1, CV_NN_2, CV_NN_33,
              LS_TM_1, LS_TM_2, LS_TM_33]
    # all predictions on input test data
    predictions = [BP_NN_1_PRED, BP_NN_2_PRED,BP_NN_33_PRED,
                   CV_NN_1_PRED, CV_NN_2_PRED,CV_NN_33_PRED, 
                   LS_TM_1_PRED, LS_TM_2_PRED,LS_TM_33_PRED]
    return(models,predictions)
