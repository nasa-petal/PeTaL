import numpy as np
import pandas as pd 

import torch
import torch.nn as nn
import torch.nn.functional as F

class AirfoilModel(nn.Module):
    def __init__(self, inputs, outputs, hidden=1, width=100):
        nn.Module.__init__(self)
        self.fc1 = nn.SELU(nn.Linear(inputs, width))
        self.fc2 = nn.SELU(nn.Linear(width, width))
        self.fc3 = nn.Linear(width, outputs)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(hidden):
            x = self.fc2(x)
        x = self.fc3(x)
        return x

    # l_track=LossTrack.Histories()
    # model_cp = base_model(len(x_train.columns),neurons,114)
    # history_callback = model_cp.fit(x_train,y_train,
    #     batch_size=8196, epochs=20,
    #     verbose=1,validation_data=(x_test,y_test))
    # loss = history_callback.history['loss']
    # val_loss = history_callback.history['val_loss']

    # # Save the loss to csv
    # if os.path.exists('loss.csv'):
    #     csv_input = pd.read_csv('loss.csv')
    #     csv_input[neuron_name] = loss
    # else:
    #     test = {neuron_name:loss}
    #     csv_input = pd.DataFrame.from_dict(test)
    # csv_input.to_csv('loss.csv', index=False)
    # 
    # if os.path.exists('val_loss.csv'):
    #     csv_input = pd.read_csv('val_loss.csv')
    #     csv_input[neuron_name] = val_loss
    # else:
    #     test = {neuron_name:val_loss}
    #     csv_input = pd.DataFrame.from_dict(test)
    # csv_input.to_csv('val_loss.csv', index=False)

    # # serialize model to JSON
    # save_name='model/keras_cp_base_{0}'.format(neuron_name)
    # model_json = model_cp.to_json()
    # with open(save_name + ".json", "w") as json_file:
    #     json_file.write(model_json)

    # # serialize weights to HDF5
    # model_cp.save_weights(save_name + ".h5")
    # print("Saved model to disk")

    # y_pred = model_cp.predict(x_validation) 
    # print(model_cp.summary())

    # csv_columns = y_test.columns.values
    # csv_columns = np.append(csv_columns, 'neuron_name')

    # # Mean Analysis
    # if not os.path.exists('neuron_mean_analysis.csv'): 
    #     with open('neuron_mean_analysis.csv', 'w', newline='') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #         writer.writeheader()

    # with open('neuron_mean_analysis.csv', 'a', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #     dic = {}
    #     for indx in range(len(y_test.columns)):
    #         # print("mean squared error {0}: {1:0.5f}".format(y_test.columns[indx], mean_squared_error(y_validation[y_test.columns[indx]],y_pred[:,indx])))   
    #         dic[y_test.columns[indx]] = mean_squared_error(y_validation[y_test.columns[indx]],y_pred[:,indx])
    #     dic['neuron_name'] = neuron_name
    #     writer.writerow(dic)
    # 
    # # Mean Analysis
    # if not os.path.exists('neuron_variance_analysis.csv'): 
    #     with open('neuron_variance_analysis.csv', 'w', newline='') as csvfile:
    #         writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #         writer.writeheader()
    # 
    # with open('neuron_variance_analysis.csv', 'a', newline='') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
    #     dic = {}
    #     for indx in range(len(y_test.columns)):
    #         # print("variance score {0}: {1:0.5f}".format(y_test.columns[indx], r2_score(y_validation[y_test.columns[indx]],y_pred[:,indx])))
    #         dic[y_test.columns[indx]] = r2_score(y_validation[y_test.columns[indx]],y_pred[:,indx])
    #     dic['neuron_name'] = neuron_name
    #     writer.writerow(dic)


# def load_model(model_name,x_test,y_test=None):
#     json_file = open(model_name + '.json', 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     loaded_model = model_from_json(loaded_model_json)
#     # load weights into new model
#     loaded_model.load_weights(model_name,".h5")
#     print("Loaded model from disk")
# 
#     # evaluate loaded model on test data
#     loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
#     if (y_test):
#         score = loaded_model.evaluate(X, Y, verbose=0)
#         print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))        
#     y_pred = loaded_model.predict(x_test)

def main():

    # # NORMALIZE ALL COLUMNS BY the DATASET
    # min_max_scaler = MinMaxScaler()
    # min_max_feature = {}
    # COLUMNS.remove("AirfoilName")
    # for col in COLUMNS:
    #     min_F=change_type(df[[col]].min()[0])
    #     max_F=change_type(df[[col]].max()[0])
    #     
    #     min_max_feature[col] = [min_F,max_F]
    #     feat_scaled = min_max_scaler.fit_transform(df[[col]].values.astype(float))
    #     df[[col]] = feat_scaled
    # # Save the min and max for rescale later
    # with open(h5_file.replace('.h5','.json'), 'w') as f:
    #     json.dump(min_max_feature, f)
    # model.compile(loss='mean_squared_error', optimizer = 'adam')
if __name__ == '__main__':
    main()
