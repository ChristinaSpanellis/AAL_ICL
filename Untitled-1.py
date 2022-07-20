
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
import pandas
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import numpy as np

dataset = read_csv('datasets/hh101/hh101.csv', names=["Sensor",1,2,"Value","Type"])
dataset.drop(columns={1,2,"Type"}, inplace=True)
dataset["Value"] = dataset["Value"].replace({"ON" : 1.0, "OFF" : 0.0})
dataset["Value"] = dataset["Value"].replace({"ABSENT" : 1.0, "PRESENT" : 0.0})
dataset["Value"] = dataset["Value"].replace({"OPEN" : 1.0, "CLOSE" : 0.0})
print(dataset)

count = 0
sensor_id_mapping = {}
for sensor in dataset["Sensor"].values:
    if sensor not in sensor_id_mapping:
        sensor_id_mapping[sensor] = count
        count+=1
# # transform sensorids from text into numbers to make life easier
# dataset["SensorId"].replace(replacement_mapping_dict, inplace=True)
# # drop the event labelling as we aren't predicting activities
# dataset.drop('Event', axis=1, inplace=True)

data = []

starting_date_time = datetime.strptime(dataset.index[0], '%Y-%m-%d %H:%M:%S.%f')
starting_date_time = starting_date_time.replace(microsecond=0)
# segment the data into periods of 30 seconds
# print(starting_date_time)
sensor_vals = [0.0] * len(sensor_id_mapping)
count = 0
for i, row in dataset.iterrows():
    curr_date_time =  datetime.strptime(i, '%Y-%m-%d %H:%M:%S.%f')
    curr_date_time = curr_date_time.replace(microsecond=0)
    if (count >= 19):
        values = [starting_date_time.strftime("%m-%d-%Y %H:%M:%S")]
        values.extend(sensor_vals)
        data.append(values)
        starting_date_time = curr_date_time
        sensor_vals = [0.0] * len(sensor_id_mapping)
        count = 0
    count+=1
    sensor_vals[sensor_id_mapping[row["Sensor"]]] = float(row["Value"])

columns = [i for i in range (0,len(sensor_id_mapping))]
final_columns = ["Time"]
final_columns.extend(columns)
new_data = pandas.DataFrame.from_records(data, columns=final_columns)      
print(new_data.head)

values = new_data.values
# groups = []
# empty_columns = []
# for i in range (0,len(sensor_id_mapping)):
#     groups.append(i)
#     if ((new_data[i] == new_data[i][0]).all()):
#         empty_columns.append(i)
# i = 1
# pyplot.figure()
# for group in groups:
#     pyplot.subplot(len(groups), 1, i)
#     pyplot.plot(values[:, group])
#     pyplot.title(new_data.columns[group], y=0.5, loc='right')
#     i += 1
# pyplot.show()
# # print("Irrelevant columns are: ", empty_columns)


# Remove empty columns from the data and plot resulting traces

# %%
reversed_mapping = {y: x for x, y in sensor_id_mapping.items()}
changed_legend = new_data.rename(columns = reversed_mapping)
for column in empty_columns:
    reversed_mapping.pop(column)

changed_legend.plot(subplots=True, figsize=(40,40))
pyplot.tight_layout()
pyplot.savefig("plots/CleanedData.png", format="png", dpi=1200)

# now create a scatter plot of the data at each time step

# %%
changed_legend.plot(figsize=(30,30))
pyplot.tight_layout()
pyplot.savefig("plots/idk.png", format="png", dpi=1200)

# Now need to transform the data into a sequence form so that we have predicted labels for each row of data

# %%
df = changed_legend.copy()
df.drop(columns="Time", inplace=True)
# scale values
values = df.values
min_max_scaler = MinMaxScaler()
scaled_values = min_max_scaler.fit_transform(values)
normalized_df = pandas.DataFrame(scaled_values)
n_vars = len(df.columns)
cols, names = list(), list()
# input sequence (t-n, ... t-1)
for i in range(1, 0, -1):
    cols.append(df.shift(i))
    names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
# forecast sequence (t, t+1, ... t+n)
for i in range(0, 1):
    cols.append(df.shift(-i))
    if i == 0:
        names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
    else:
        names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
# put it all together
agg = concat(cols, axis=1)
agg.columns = names
# drop rows with NaN values
agg.dropna(inplace=True)
print(agg.head)

# Now we need to separate into test and train data

# %%
values = agg.values
train_split = int(0.8 * len(values))

train = values[:train_split, :]
test = values[train_split:, :]

features = 58
train_x, train_y = train[:, :features], train[:, features:]
test_x, test_y = test[:, :features], test[:, features:]
train_x = np.asarray(train_x).astype(np.float32)
train_y = np.asarray(train_y).astype(np.float32)
test_y = np.asarray(test_y).astype(np.float32)
test_x = np.asarray(test_x).astype(np.float32)
train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))


# Now create the network

# %%
# design network
model = Sequential()
model.add(LSTM(300, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dense(features))
model.compile(loss='mean_squared_error', optimizer='adam')
# fit network
history = model.fit(train_x, train_y, epochs=100, batch_size=72, validation_data=(test_x, test_y), verbose=2, shuffle=False)
# plot history
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='test')
pyplot.legend()
pyplot.show()

# Now test the model

# 

# %%
from numpy import concatenate
from math import sqrt
from sklearn.metrics import mean_squared_error

# make a prediction
yhat = model.predict(test_x)
test_x = test_x.reshape((test_x.shape[0], 23))
print(yhat)
# # calculate RMSE
rmse = sqrt(mean_squared_error(test_y, yhat))
print('Test RMSE: %.3f' % rmse)


# # %%
# # Try generating some anomalous data
# # One idea is to just scale data to be bigger than 1, in theory this is an anomaly but isn't really in the expected range of input...
# # https://github.com/tirthajyoti/Synthetic-data-gen/blob/master/Notebooks/Time%20series%20synthesis%20with%20anomaly.ipynb
# anomaly_frac = 0.2
# one_sided = False
# new_arr = agg.copy()
# arr_min = new_arr.min()
# arr_max = new_arr.max()
# no_anomalies = int(new_arr.size * anomaly_frac)
# idx_list = np.random.choice(a=new_arr.size , size=no_anomalies, replace=False)
# for idx in idx_list:
#     if one_sided:
#         new_arr[idx] = self.loc + np.random.uniform(
#             low=arr_min, high=anomaly_scale * (arr_max - arr_min)
#         )
#     else:
#         new_arr[idx] = self.loc + np.random.uniform(
#             low=-anomaly_scale * (arr_max - arr_min),
#             high=anomaly_scale * (arr_max - arr_min),
#         )
# self.anomalized_data = new_arr
# self._anomaly_flag_ = True

# if return_df:
#     df = pd.DataFrame(
#         {"time": self.time_arr, "anomaly_data": self.anomalized_data}
#     )
#     return df
# else:
#     return self.anomalized_data


