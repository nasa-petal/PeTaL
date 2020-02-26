from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import ast
import numpy as np
import pandas as pd 
import collections
import os 
import itertools
import json
# import keras_loss_track as LossTrack
import csv
from shutil import copyfile
# example of training a final classification model
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import model_from_json

# Create dictionary with columnTypes 
OUTPUTS = []
INPUTS = []
dropColumns = []
for i in range(0,1020,20):
    OUTPUTS.append("y_ss_" + str(int(i)))
for i in range(0,1020,20):
    OUTPUTS.append("y_ps_" + str(int(i)))
for i in range(0,1020,20):
    INPUTS.append("cp_ss_" + str(int(i)))
for i in range(0,1020,20):
    INPUTS.append("cp_ps_" + str(int(i)))
INPUTS.append("Re")
INPUTS.append("Ncrit") # FEATURES
INPUTS.append("alpha")
OUTPUTS.append("Cl") # LABELS
OUTPUTS.append("Cd")
OUTPUTS.append('Cdp')
OUTPUTS.append("Cm")

COLUMNS = list()
COLUMNS.append('AirfoilName')
COLUMNS.extend(OUTPUTS) # Features 
COLUMNS.extend(INPUTS)
# Used to handle json errors
def change_type(o):
    if isinstance(o, np.int64): 
        return int(o)  
    return o

##define base model
def base_model(input_dim,neurons,outputs):
    model = Sequential()
    model.add(Dense(neurons[0], input_dim=input_dim, kernel_initializer='normal', activation='relu'))
    for i in range(1,len(neurons)):
        model.add(Dense(neurons[i], kernel_initializer='normal', activation='relu'))        
    
    model.add(Dense(outputs, kernel_initializer='normal')) 
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    return model  

def evaluate_model(df, neurons):
    neuron_name = '_'.join(map(str, neurons))

    df.isnull().values.any()
    [training_set, test_set] = train_test_split(df, test_size=0.3)
    [validation_set, test_set] = train_test_split(test_set, test_size=0.4)

    x_train = training_set.drop(columns=OUTPUTS)
    y_train = training_set.drop(columns=INPUTS)

    x_test = test_set.drop(columns=OUTPUTS)
    y_test = test_set.drop(columns=INPUTS)

    x_validation = validation_set.drop(columns=OUTPUTS)
    y_validation = validation_set.drop(columns=INPUTS)        

    # l_track=LossTrack.Histories()
    model_cp = base_model(len(x_train.columns),neurons,114)
    history_callback = model_cp.fit(x_train,y_train,
        batch_size=8196, epochs=20,
        verbose=1,validation_data=(x_test,y_test))
    loss = history_callback.history['loss']
    val_loss = history_callback.history['val_loss']

    # Save the loss to csv
    if os.path.exists('loss.csv'):
        csv_input = pd.read_csv('loss.csv')
        csv_input[neuron_name] = loss
    else:
        test = {neuron_name:loss}
        csv_input = pd.DataFrame.from_dict(test)
    csv_input.to_csv('loss.csv', index=False)
    
    if os.path.exists('val_loss.csv'):
        csv_input = pd.read_csv('val_loss.csv')
        csv_input[neuron_name] = val_loss
    else:
        test = {neuron_name:val_loss}
        csv_input = pd.DataFrame.from_dict(test)
    csv_input.to_csv('val_loss.csv', index=False)

    # serialize model to JSON
    save_name='model/keras_cp_base_{0}'.format(neuron_name)
    model_json = model_cp.to_json()
    with open(save_name + ".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model_cp.save_weights(save_name + ".h5")
    print("Saved model to disk")

    y_pred = model_cp.predict(x_validation) 
    print(model_cp.summary())

    csv_columns = y_test.columns.values
    csv_columns = np.append(csv_columns, 'neuron_name')

    # Mean Analysis
    if not os.path.exists('neuron_mean_analysis.csv'): 
        with open('neuron_mean_analysis.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()

    with open('neuron_mean_analysis.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        dic = {}
        for indx in range(len(y_test.columns)):
            # print("mean squared error {0}: {1:0.5f}".format(y_test.columns[indx], mean_squared_error(y_validation[y_test.columns[indx]],y_pred[:,indx])))   
            dic[y_test.columns[indx]] = mean_squared_error(y_validation[y_test.columns[indx]],y_pred[:,indx])
        dic['neuron_name'] = neuron_name
        writer.writerow(dic)
    
    # Mean Analysis
    if not os.path.exists('neuron_variance_analysis.csv'): 
        with open('neuron_variance_analysis.csv', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
    
    with open('neuron_variance_analysis.csv', 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        dic = {}
        for indx in range(len(y_test.columns)):
            # print("variance score {0}: {1:0.5f}".format(y_test.columns[indx], r2_score(y_validation[y_test.columns[indx]],y_pred[:,indx])))
            dic[y_test.columns[indx]] = r2_score(y_validation[y_test.columns[indx]],y_pred[:,indx])
        dic['neuron_name'] = neuron_name
        writer.writerow(dic)


def load_model(model_name,x_test,y_test=None):
    json_file = open(model_name + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(model_name,".h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    if (y_test):
        score = loaded_model.evaluate(X, Y, verbose=0)
        print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))        
    y_pred = loaded_model.predict(x_test)

def main(unused_argv):
    h5_file = 'storage.h5'

    df = pd.read_hdf(h5_file, 'data')

    print('Num Columns %s' % len(df.columns))
    df = df.drop(columns=['AirfoilName'])

    print('Num Columns %s' % len(df.columns))
    # print('DataFrame Shape: %s' % df.shape)

    # NORMALIZE ALL COLUMNS BY the DATASET
    min_max_scaler = MinMaxScaler()
    min_max_feature = {}
    COLUMNS.remove("AirfoilName")
    for col in COLUMNS:
        min_F=change_type(df[[col]].min()[0])
        max_F=change_type(df[[col]].max()[0])
        
        min_max_feature[col] = [min_F,max_F]
        feat_scaled = min_max_scaler.fit_transform(df[[col]].values.astype(float))
        df[[col]] = feat_scaled
    # Save the min and max for rescale later
    with open(h5_file.replace('.h5','.json'), 'w') as f:
        json.dump(min_max_feature, f)

    neurons_3_layer=[[16,16,16],[32,32,32],[64,64,64],[128,128,128]]
    neurons_4_layer=[[16,16,16,16],[32,32,32,32],[64,64,64,64],[128,128,128,128]]
    neurons_5_layer=[[16,16,16,16,16],[32,32,32,32,32],[64,64,64,64,64],[128,128,128,128,128]]
    neurons_6_layer=[[16,16,16,16,16,16],[32,32,32,32,32,32],[64,64,64,64,64,64],[128,128,128,128,128,128]]
    neurons_12_layer=[[16,16,16,16,16,16,16,16,16,16,16,16],[32,32,32,32,32,32,32,32,32,32,32,32],[64,64,64,64,64,64,64,64,64,64,64,64],[128,128,128,128,128,128,128,128,128,128,128,128]]


    for neuron in neurons_3_layer:
        evaluate_model(df,neuron)
    for neuron in neurons_4_layer:
        evaluate_model(df,neuron)
    for neuron in neurons_5_layer:
        evaluate_model(df,neuron)
    for neuron in neurons_6_layer:
        evaluate_model(df,neuron)
    for neuron in neurons_12_layer:
        evaluate_model(df,neuron)
main(None)
# if __name__ == "__main__":

#     tf.logging.set_verbosity(tf.logging.INFO)
#     with tf.Session() as sess:        
#         tf.app.run()

    
