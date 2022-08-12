# %% [markdown]
# 
# # MSc Individual Project: Anomaly Detection for Assisted Independent Living
# 
# Author: Christina Spanellis
#  
# Sections 1-4 of this notebook define the necessary method definitions and constants for this project.
# 
# Section 5 contains code to build and test the system
# 
# ## Sections
# ### 1. [Data preparation and pre-processing](#section1)
# ### 2. [Anomalous Data Generation Module](#section2)
# ### 3. [Prediction Module](#section3)
# ### 4. [Anomaly Detection Module](#section4)
# ### 5. [Running the system](#section5)

# %%
# import the necessary packages for the project
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
import pandas
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from keras import models
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf
from keras.models import Sequential
from keras import callbacks
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import seaborn as sns
import keras_tuner as kt
import numpy as np
# from keras.metrics import mean_squared_error
from numpy import sqrt
import os.path

# %% [markdown]
# <a id='section1'></a>
# ## Section 1: Data preparation and pre-processing

# %% [markdown]
# Two different data sets had to be cleaned and prepared to be in a suitable format for the prediction module. 
# 
# Data set 1: CASAS HH101
# 
# Data set 2: CASAS HH102
# 
# The below definitions define the constants and logic needed for loading and pre-processing. 

# %%
## CONSTANTS AND GLOBALS DEFINITION

SENSOR_EVENT_WINDOW_SIZE = 19
HH101_PREDICTION_SENSORS = ["M003", "LS002", "M004", "M005", "LS005", "MA015", "M012", "M010"]
HH102_PREDICTION_SENSORS = ["M007", "T105", "LS008", "M004", "M002", "LS010", "MA009", "M018", "LS002"]
NUMBER_IN_FEATURES = None
DATASET = None
NUMBER_PREDICTIONS = None
ENABLE_PLOTS = 0

# %%
# load the hh101 data set
def load_hh101():
    # load the data set
    hh101 = read_csv('datasets/hh101/hh101.csv', names=["Sensor",1,2,"Value","Type"])
    hh101.drop(columns={1,2,"Type"}, inplace=True)
    # replace string values and drop unwanted sensor readings
    hh101 = hh101[hh101["Sensor"].str.contains("BAT") == False]
    hh101["Value"] = hh101["Value"].replace({"ON" : 1.0, "OFF" : 0.0})
    hh101["Value"] = hh101["Value"].replace({"ABSENT" : 1.0, "PRESENT" : 0.0})
    hh101["Value"] = hh101["Value"].replace({"OPEN" : 1.0, "CLOSE" : 0.0})
    # creating a mapping of the sensor names to keep track of them
    count = 0
    hh101_sensor_id_mapping = {}
    for sensor in hh101["Sensor"].values:
        if sensor not in hh101_sensor_id_mapping:
            hh101_sensor_id_mapping[sensor] = count
            count+=1
    hh101_reversed_mapping = {y: x for x, y in hh101_sensor_id_mapping.items()}
    return (hh101, hh101_sensor_id_mapping, hh101_reversed_mapping)

# load the hh102 dataset 
def load_hh102():
    # load the data set
    hh102 = read_csv('datasets/hh102/hh102.csv', names=["Sensor",1,2,"Value","Type"])
    hh102.drop(columns={1,2,"Type"}, inplace=True)
    # replace string values and drop unwanted sensor readings
    hh102 = hh102[hh102["Sensor"].str.contains("BAT") == False]
    hh102["Value"] = hh102["Value"].replace({"ON" : 1.0, "OFF" : 0.0})
    hh102["Value"] = hh102["Value"].replace({"ABSENT" : 1.0, "PRESENT" : 0.0})
    hh102["Value"] = hh102["Value"].replace({"OPEN" : 1.0, "CLOSE" : 0.0})
    # creating a mapping of the sensor names to keep track of them
    count = 0
    hh102_sensor_id_mapping = {}
    for sensor in hh102["Sensor"].values:
        if sensor not in hh102_sensor_id_mapping:
            hh102_sensor_id_mapping[sensor] = count
            count+=1
    hh102_reversed_mapping = {y: x for x, y in hh102_sensor_id_mapping.items()}
    return (hh102, hh102_sensor_id_mapping, hh102_reversed_mapping)


# %%

def remove_sensors(dataset, sensors, mapping):
    for i in range (len(sensors)):
        sensors[i] = mapping[sensors[i]]
    return dataset.drop(columns=sensors)
     

# %%
from sklearn.decomposition import PCA
def pca_decomp(dataset, reversed_mapping):
    dataset = dataset.copy()
    min_max_scaler = MinMaxScaler()
    dataset[dataset.columns] = min_max_scaler.fit_transform(dataset)
    pca = PCA(0.8)
    components = pca.fit_transform(dataset)
    n_pcs= pca.n_components_ # get number of component
    # get the index of the most important feature on EACH component
    most_important = [np.abs(pca.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = dataset.columns
    # get the most important feature names
    most_important_features = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    most_important_features = [i for n, i in enumerate(most_important_features) if i not in most_important_features[:n]] 
    if (ENABLE_PLOTS):
        if (check_stationary(dataset, most_important_features)):
            print("All selected features stationary")
        check_autocorrelation(dataset, most_important_features, reversed_mapping)
        features = range(pca.n_components_)
        _ = pyplot.figure(figsize=(15, 5))
        _ = pyplot.bar(features, pca.explained_variance_)
        _ = pyplot.xlabel('PCA feature')
        _ = pyplot.ylabel('Variance')
        _ = pyplot.xticks(features)
        _ = pyplot.title("Importance of the Principal Components based on inertia")
        pyplot.savefig("data_plots/"+DATASET +"/pca")
        pyplot.close()
    for i in range(len(most_important_features)):
        most_important_features[i] = reversed_mapping[most_important_features[i]]
    return most_important_features



# %%
def check_stationary(pca, n_components):
    for column in n_components:
        result = adfuller(pca[column])[1]
        if result > 0.05:
            return False
    return True

# %%
def check_autocorrelation(dataset, most_important_features, reversed_mapping):
    axs = None
    if (len(most_important_features)  % 2 == 0):
        fig, axs = pyplot.subplots(int(len(most_important_features) / 2),2,  sharex=True, sharey=True, figsize=(11,8))
    else:
        fig, axs = pyplot.subplots(int(len(most_important_features) / 2)+1,2,  sharex=True, sharey=True, figsize=(11,8))

    axis = axs.flat
    for i, column in enumerate(most_important_features):
        plot_acf(dataset[column], lags=20, ax = axis[i], alpha=0.05, title="Sensor " + reversed_mapping[column]  +" autocorrelation")
    if (len(axis) > len(most_important_features)):
        fig.delaxes(axis[-1])
    
    fig.supxlabel("Lags")
    fig.supylabel("ACF")
    pyplot.tight_layout()
    pyplot.savefig("data_plots/" + DATASET + "/autocorrelation")

# %% [markdown]
# This method transforms the data set into a format where the columns represent the various sensor values and segment the data into 20 event sensor windows.
# 
# This means that each row in the data set represents the activations for the previous 20 sensor event activations

# %%
def transform_data(dataset, sensor_id_mapping):
    if os.path.isfile("datasets/"  + DATASET +"/transformed_data.csv"):
        dataframe = pandas.read_csv("datasets/"  + DATASET +"/transformed_data.csv", index_col="Time")
        columns = [i for i in range (0,len(sensor_id_mapping))]
        dataframe.columns = columns
        return dataframe
    else:
        data = []
        starting_date_time = datetime.strptime(dataset.index[0], '%Y-%m-%d %H:%M:%S.%f')
        starting_date_time = starting_date_time.replace(microsecond=0)
        sensor_vals = [0.0] * len(sensor_id_mapping)
        event_count = 0 ## counter used to segment the data into sensor event windows
        for i, row in dataset.iterrows():
            curr_date_time =  datetime.strptime(i, '%Y-%m-%d %H:%M:%S.%f')
            curr_date_time = curr_date_time.replace(microsecond=0)
            if (event_count >= SENSOR_EVENT_WINDOW_SIZE):
                values = [starting_date_time.strftime("%m-%d-%Y %H:%M:%S")]
                values.extend(sensor_vals)
                data.append(values)
                starting_date_time = curr_date_time
                sensor_vals = [0.0] * len(sensor_id_mapping)
                event_count = 0
            event_count +=1
            if "D" in row["Sensor"] or "M" in row["Sensor"]:
                if sensor_vals[sensor_id_mapping[row["Sensor"]]] == 0.0:
                    sensor_vals[sensor_id_mapping[row["Sensor"]]] = 1.0
                else:
                    sensor_vals[sensor_id_mapping[row["Sensor"]]] += 1.0
            else:
                    sensor_vals[sensor_id_mapping[row["Sensor"]]] = row["Value"]

        columns = [i for i in range (0,len(sensor_id_mapping))]
        final_columns = ["Time"]
        final_columns.extend(columns)
        # set the index of the dataframe to be the time column
        new_data = pandas.DataFrame.from_records(data, columns=final_columns)
        new_data["Time"] = pandas.to_datetime(new_data["Time"], format='%m-%d-%Y %H:%M:%S')
        new_data = new_data.set_index("Time")
        new_data.to_csv("datasets/"  + DATASET +"/transformed_data.csv")
        return new_data   


# %% [markdown]
# This method plots all the data (non-normalised) after formatting

# %%
def plot_cleaned_data(data, reversed_mapping):
    changed_legend = data.rename(columns = reversed_mapping)
    ax = changed_legend.plot(subplots=True,figsize=(12,24), sharey=True, ylabel="Sensor Value")
    pyplot.tight_layout()
    pyplot.savefig("data_plots/" + DATASET + "/cleaned_data", dpi=1200)

# %% [markdown]
# This method transforms the series into a format suitable for a supervised learning problem.
# 
# Eg. [Sensor1(t-1), Sensor2(t-1), Sensor1(t), Sensor2(t)]
# 
# Where readings at t-1 represent the sensor activations for the period before the activations at t.
# 
# Sensor activations at time t then become the ground truth values for activations at t-1 in the prediction module

# %%
def transform_series_to_supervised(new_data, prediction_sensors, important_features, reversed_mapping):
    if os.path.isfile("datasets/"  + DATASET +"/final_dataset.csv"):
        columns = [i for i in range (0,len(reversed_mapping))]
        return (pandas.read_csv("datasets/"  + DATASET +"/final_dataset.csv", index_col=[0]), pandas.read_csv("datasets/"  + DATASET +"/normalised_dataset.csv",  names=columns))
    else:
        df = new_data.copy()
        # scale values
        values = df.values
        min_max_scaler = MinMaxScaler()
        scaled_values = min_max_scaler.fit_transform(values)
        normalized_df = pandas.DataFrame(scaled_values)
        normalized_df.to_csv("datasets/"  + DATASET +"/normalised_dataset.csv")
        n_vars = len(normalized_df.columns)
        time_Series = normalized_df.copy().set_index(df.index)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(1, 0, -1):
            sequence = normalized_df.shift(i)
            sequence = sequence.rename(columns=reversed_mapping)
            sequence = sequence[important_features]
            cols.append(sequence)
            names += [('%s(t-%d)' % (label, i)) for label in sequence.columns]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, 1):
            sequence = normalized_df.shift(-i)
            sequence = sequence.rename(columns=reversed_mapping)
            sequence = sequence[prediction_sensors]
            cols.append(sequence.shift(-i))
            names += [('%s(t)' % (label)) for label in sequence.columns]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        agg.dropna(inplace=True)
        agg.to_csv("datasets/"  + DATASET +"/final_dataset.csv")
        return (agg, normalized_df)

# %% [markdown]
# This method plots the activations for the sensors in the data set, grouped into figures by sensor type.

# %%
def plot_normalised_sensor_activations(normalized_df, reversed_mapping):
    changed_legend = normalized_df.rename(columns = reversed_mapping)
    doors =[]
    lights = []
    temp = []
    motion = []
    for key in reversed_mapping:
        if "D" in reversed_mapping[key]:
            doors.append(key)
        elif "L" in reversed_mapping[key]:
            lights.append(key)
        elif "T1" in reversed_mapping[key]:
            temp.append(key)
        else:
            motion.append(key)
    sensors = [doors, lights, temp, motion]
    names = ["Door Sensors", "Light Sensors", "Temperature Sensors", "Motion Sensors"]
    for i, sensor in enumerate(sensors):
        if "Door" or "Temperature" in names[i]:
            figsize = (11,4)
        if (len(sensor) % 2 == 0):
            fig, axs = pyplot.subplots(int(len(sensor) / 2),2,  sharex=True, sharey=True, figsize=(11,8))
        else:
            fig, axs = pyplot.subplots(int(len(sensor) / 2)+1,2,  sharex=True, sharey=True, figsize=(11,4))

        count = 0
        for ax in axs.flat:
            if (count < len(sensor)):
                ax.plot(changed_legend[reversed_mapping[sensor[count]]])
                ax.set_title(reversed_mapping[sensor[count]])
                ax.set_xticks([0, 10000, 20000, 30000, 40000, 50000])
                ax.set_yticks([0.0, 0.5, 1.0])
                count += 1
            else:
                fig.delaxes(ax)
        for ax in axs.flat:
            ax.label_outer()
        fig.suptitle(names[i])
        fig.supxlabel("SEW")
        fig.supylabel("Sensor Value")
        pyplot.tight_layout()
        pyplot.savefig("data_plots/" + DATASET + "/" +names[i],dpi=1200)

# %% [markdown]
# Discover the features with the highest correlations

# %%
def find_correlations(dataset, reversed_mapping):
    min_max_scaler = MinMaxScaler()
    scaled_values = min_max_scaler.fit_transform(dataset.values)
    normalized_df = pandas.DataFrame(scaled_values)
    normalized_df = normalized_df.rename(columns = reversed_mapping)
    corrmat = normalized_df.corr(method='pearson', min_periods=100)
    corrmat = np.abs(corrmat)
    sns.set(context="paper", font="monospace")
    f, ax = pyplot.subplots(figsize=(12, 9))
    sns.heatmap(corrmat, vmax=0.8, square=True, xticklabels = True, yticklabels = True)
    pyplot.title(DATASET.upper() + " Pearson correlation values between sensors (absolute valued).")
    pyplot.xlabel("Sensor ID")
    pyplot.ylabel("Sensor ID")
    pyplot.tight_layout()
    pyplot.savefig("data_plots/" + DATASET + "/correlations")
    triangluar_corrmat = np.triu(corrmat, k=1)
    values = np.where(triangluar_corrmat >= 0.6)
    values = list(zip(values[0], values[1]))
    for x in range (len(values)):
        values[x] = (reversed_mapping[values[x][0]], reversed_mapping[values[x][1]], triangluar_corrmat[values[x][0]][values[x][1]])
    return values 
    

# %% [markdown]
# The data needs to be separated out into training (70%), testing (20%) and anomalous portions (10%).
# 
# The anomalous portion of the data is held back for synthetic anomaly injection to later be used to test the AD system

# %%
def create_train_test_split(dataset):
    values = dataset.values
    train_split = int(0.7 * len(values))
    anomaly_split = int(0.9 * len(values))
    train = values[:train_split, :]
    test = values[train_split:anomaly_split, :]
    anomalies = values[anomaly_split:, :]

    train_x, train_y = train[:, :NUMBER_IN_FEATURES], train[:, NUMBER_IN_FEATURES:]

    test_x, test_y = test[:, :NUMBER_IN_FEATURES], test[:, NUMBER_IN_FEATURES:]

    anomaly_x, anomaly_y = anomalies[:, :NUMBER_IN_FEATURES], anomalies[:, NUMBER_IN_FEATURES:]

    train_x = np.asarray(train_x).astype(np.float32)
    train_y = np.asarray(train_y).astype(np.float32)
    test_y = np.asarray(test_y).astype(np.float32)
    test_x = np.asarray(test_x).astype(np.float32)
    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
    return (train_x, train_y, test_x, test_y, anomaly_x, anomaly_y)


# %% [markdown]
# <a id='section3'></a>
# 
# ## Section 3: Prediction Module

# %% [markdown]
# The following section contains the logic to train models for the Prediction Module

# %% [markdown]
# A method to save trained models and plot their loss (used for experimentation)

# %%
def save_model_and_plot(model, history):
    model.save("best_models/" + DATASET +"/best_model", history)
    if (ENABLE_PLOTS):
        pyplot.plot(history.history['loss'], label='Loss')
        pyplot.plot(history.history['val_loss'], label='Validation Loss')
        pyplot.xlabel("Epochs")
        pyplot.ylabel("Loss")
        pyplot.title("Training and validation loss of final "+ DATASET.upper() +" model.")
        pyplot.tight_layout()
        pyplot.legend()
        pyplot.savefig("best_models/"+ DATASET +"/best_model")

# %% [markdown]
# This creates a model suitable for hyper parameter tuning in in keras

# %%
def build_model(hp):
    # design network
    model = Sequential()
    model.add(LSTM(hp.Int('input_lstm_layer', min_value = 50, max_value = 500, step = 50), input_shape=(train_x.shape[1], train_x.shape[2]), return_sequences = True))
    model.add(LSTM(hp.Int('final_lstm_layer', min_value = 50, max_value = 500, step = 50)))
    model.add(Dropout(hp.Float('Dropout_rate',min_value=0,max_value=0.5,step=0.1)))
    model.add(Dense(NUMBER_PREDICTIONS, activation = hp.Choice('dense_activation', values=['sigmoid'],default='sigmoid')))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mse'])
    return model

# %%
def create_baseline():
    model = Sequential()
    model.add(LSTM(20, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(NUMBER_PREDICTIONS, activation = 'sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics = ['mse'])
    return model

# %% [markdown]
# Create and use the keras tuner

# %%
def tune_model(x_train, y_train):

    tuner = kt.BayesianOptimization(
        build_model,
        objective='mse',
        max_trials=20,
        directory="tensorflow/"+DATASET+"/",
        project_name="models",
        overwrite = False
    )
    tuner.search(
        x_train,
        y_train,
        batch_size = 128,
        validation_split=0.2,
        epochs = 500,
        callbacks=[callbacks.TensorBoard(log_dir="/tmp/tb_logs/"+DATASET, histogram_freq=1), callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=30), callbacks.ModelCheckpoint('best_model_es.h5',monitor='val_loss',mode='min',save_best_only=True)],

    )
    return tuner

# %%
def train_baseline(model, train_x, train_y, test_x, test_y):
    history = model.fit(train_x, train_y, epochs=80, validation_data=(test_x, test_y))
    # save_model_and_plot(model, history)
    pyplot.plot(history.history['loss'], label='Loss')
    pyplot.plot(history.history['val_loss'], label='Validation Loss')
    pyplot.xlabel("Epochs")
    pyplot.ylabel("Loss")
    pyplot.title("Training and validation loss of final "+ DATASET.upper() +" model.")
    pyplot.tight_layout()
    pyplot.legend()
    pyplot.savefig("best_models/"+ DATASET +"/baseline_best_model")
    return model, history


# %% [markdown]
# Train new model with best hyper-parameters

# %%
# def get_final_model(train_x, train_y, test_x, test_y):
#     model = None
#     history = None
#     filename = "best_models/" + DATASET + "/best_model"
#     if os.path.isdir(filename):
#         model = models.load_model(filename)
#     else:
#         tuner = tune_model(train_x, train_y, "hh101")
#         model = train_final_model(train_x, train_y, test_x, test_y, tuner)
#     return model

# def train_final_model(train_x, train_y, test_x, test_y, tuner):
#     model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
#     history = model.fit(train_x, train_y, epochs=1000, validation_data=(test_x, test_y))
#     save_model_and_plot(model, history)
#     return model

def get_final_model(train_x, train_y, test_x, test_y):
    model = None
    history = None
    tuner = tune_model(train_x, train_y)
    model, history = train_final_model(train_x, train_y, test_x, test_y, tuner)
    save_model_and_plot(model, history)

    # filename = "best_models/" + DATASET + "/best_model"
    # if os.path.isdir(filename):
    #     model = models.load_model(filename)
    # else:
    #     tuner = tune_model(train_x, train_y, "hh101")
        # model = train_final_model(train_x, train_y, test_x, test_y, tuner)
    return model, history

def train_final_model(train_x, train_y, test_x, test_y, tuner):
    model = tuner.hypermodel.build(tuner.get_best_hyperparameters()[0])
    history = model.fit(train_x, train_y, epochs=80, validation_data=(test_x, test_y))
    # save_model_and_plot(model, history)
    return model, history


# %%
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
def predict(anomaly_x, model, anomaly_y):
    predictions = []
    errors = []
    sensor_errors = []
    r2_s = []
    for i in range (len(anomaly_x)):
        input = anomaly_x[i]
        ground_truth = anomaly_y[i]
        input = input.reshape((1, 1, input.shape[0]))
        pred = model.predict(input, batch_size=1, verbose=0)
        pred = pred[0]
        predictions.append(pred)
        sensor_errors_temp = []
        for i in range (len(ground_truth)):
            sensor_errors_temp.append(mean_squared_error([ground_truth[i]], [pred[i]]))
        sensor_errors.append(sensor_errors_temp)
        errors.append(mean_squared_error(ground_truth, pred))
        r2_s.append(r2_score(ground_truth, pred))
    predictions = np.array(predictions)
    pred_sens = HH102_PREDICTION_SENSORS
    if DATASET == "hh101":
        pred_sens = HH101_PREDICTION_SENSORS
    df = pandas.DataFrame(data = sensor_errors, columns=pred_sens)        
    # return predictions.reshape(predictions.shape[0], predictions.shape[2]), errors
    return errors, r2_s, predictions, df

# %% [markdown]
# <a id='section2'></a>
# ## Section 2: Anomamlous Data Generation Module

# %% [markdown]
# The following section contains the logic for the Anomalous Data Generation Module

# %% [markdown]
# This method takes the test data and inserts anomalies into a specified fraction of the data

# %%

def get_stats(dataset):
    return (dataset.mean(axis=0), dataset.min(axis=0), dataset.max(axis=0))

def generate_anomalous_data(stats, anomaly_x, anomaly_y, reversed_mapping):
    anomaly_split = 1000
    means = stats[0]
    mins = stats[1]
    maxs = stats[2]
    increase_mag = True
    number_anomalies = 200
    # now need to randomly select one row of data to to alter, so as to not alter the underlying sequence
    anomaly_scale = 1.1
    # types of generated anomalies?
    # - random
    # - intentional anomalies
    #   - swap anomalies
    #   - activate/deactivate anomalies
    # 1. random anomalies
    random_anomaly_x = anomaly_x[:anomaly_split, :]
    random_actual_y = anomaly_y[:anomaly_split, :]
    random_anomaly_y, random_rows = generate_random_anomaly(anomaly_y[:anomaly_split, :], number_anomalies, anomaly_scale, maxs, mins, means,reversed_mapping)

    # 2. intentional anomalies
    # # 2.1. one simple anomaly is all sensor values = 0
    # zero_anomaly_x = anomaly_x[anomaly_split:anomaly_split*2, :]
    # zero_anomaly_y, zero_rows = generate_intentional_anomaly(anomaly_y[anomaly_split:anomaly_split*2, :], number_anomalies, anomaly_scale, maxs, mins, "zero", reversed_mapping)

    # # 2.2 activate some portion of non-active sensors
    activate_anomaly_x = anomaly_x[anomaly_split*2:anomaly_split*3, :]
    activate_anomaly_y, activate_rows = generate_intentional_anomaly(anomaly_y[anomaly_split*2:anomaly_split*3, :], number_anomalies, anomaly_scale, maxs, mins, means,"activate", reversed_mapping)

    # deactivate_anomaly_x = anomaly_x[anomaly_split*3:anomaly_split*4, :]
    # # # 2.3 de-activate active, activate portion of non-active
    # deactivate_anomaly_y, deactivate_rows = generate_intentional_anomaly(anomaly_y[anomaly_split*3:anomaly_split*4, :], number_anomalies, anomaly_scale, maxs, mins, "deactivate", reversed_mapping)
    return (random_anomaly_x, random_anomaly_y, random_rows, random_actual_y),(activate_anomaly_x, activate_anomaly_y, activate_rows)


# %%

def generate_random_anomaly(anomaly_y, number_anomalies, anomaly_scale, maxs, mins, means, reversed_mapping):
    row_ids = np.random.choice(anomaly_y.shape[0], size=number_anomalies, replace=False)
    random_anomalous_y = pandas.DataFrame(anomaly_y)
    random_anomalous_y_copy = random_anomalous_y.copy()
    for row_id in row_ids:
        new_data = [0.0] * len(random_anomalous_y.columns)
        for y in range(len(new_data)):
            # if (np.random.ranf() < 0.5):
                new_data[y] = means[y] + np.random.uniform(low=random_anomalous_y.loc[row_id][y], high=anomaly_scale * (maxs[y] - mins[y]))
        random_anomalous_y.loc[row_id] = new_data
    plot_anomalsed_data(row_ids, random_anomalous_y, random_anomalous_y_copy, "random_anomaly", reversed_mapping)
    return random_anomalous_y.values, row_ids

# %%
def generate_intentional_anomaly(anomaly_y, number_anomalies, anomaly_scale, maxs, mins, means, type, reversed_mapping):
    row_ids = np.random.choice(anomaly_y.shape[0], size=number_anomalies, replace=False)
    intentional_anomalous_y = pandas.DataFrame(anomaly_y)
    intentional_anomalous_y_copy = intentional_anomalous_y.copy()
    for row_id in row_ids:
        new_data = [0.0] * len(intentional_anomalous_y.columns)
        for y in range(len(new_data)):
            if type=="deactivate":
                if (intentional_anomalous_y.iloc[row_id][y] != 0.0):
                    new_data[y] = np.random.uniform(low=mins[y], high=(maxs[y] - mins[y]))
            elif type=="activate":
                if (intentional_anomalous_y.iloc[row_id][y] == 0.0):
                    new_data[y] = means[y] + np.random.uniform(low=mins[y], high= (maxs[y] - mins[y]))
        intentional_anomalous_y.loc[row_id] = new_data
    plot_anomalsed_data(row_ids, intentional_anomalous_y, intentional_anomalous_y_copy, "intentional_anomaly", reversed_mapping)
    return intentional_anomalous_y.values, row_ids

# %%
def plot_anomalsed_data(row_ids, dataset, dataset_copy, type, reversed_mapping):
    if (ENABLE_PLOTS):
        colors = None
        if (DATASET=="hh101"):
            colors = ["cornflowerblue","lightsteelblue","mediumblue","blue","slateblue","navy","royalblue", "dodgerblue"]
        else:
            colors = ["cornflowerblue","lightsteelblue","mediumblue","blue","slateblue","navy","royalblue", "dodgerblue","cyan"]
        fig, (ax1, ax2) = pyplot.subplots(2,1, figsize=(17,12), dpi=300, sharey=True)
        # pyplot.setp((ax1,ax2), xticks=[0,20,40,60,80,100],yticks=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax1.set_xlabel('SEW')
        ax1.set_ylabel('Sensor Value')
        ax2.set_xlabel('SEW')
        ax2.set_ylabel('Sensor Value')
        fig.suptitle("Normal data vs data with random synthetic anomaly injection")
        dataset_copy.rename(columns=reversed_mapping, inplace=True)
        dataset_copy.plot(color=colors, ax=ax1)
        dataset.rename(columns=reversed_mapping, inplace=True)
        dataset.plot(color=colors, ax=ax2)
        for row in row_ids:
            ax2.axvspan(row, row, color='red', alpha=0.5)
        pyplot.tight_layout()
        pyplot.savefig("data_plots/" + DATASET + "/" + type)


# %% [markdown]
# <a id='section4'></a>
# ## Section 4: Anomaly Detection Module

# %% [markdown]
# This method estimates the likelihood of an anomaly occuring by comparing the predicted value to the ground truth

# %%
def detect_anomaly(anomaly_y, predictions, type, actual):
    anomaly_scores = np.empty((predictions.shape[0]))
    detected_anomalies = []
    correctly_detected_x = []
    correctly_detected_y = []
    for i, prediction in enumerate(predictions):
        anomaly_probability = 0.0
        for x, value in enumerate(prediction):
            anomaly_probability+=abs(value - anomaly_y[1][i][x])
        # print("Row ", i, " probability of anomaly: ", max(anomaly_probability))
        anomaly_scores[i] = anomaly_probability / len(prediction)
    min_max_scaler = MinMaxScaler()
    anomaly_scores = anomaly_scores.reshape(-1, 1)
    anomaly_scores = min_max_scaler.fit_transform(anomaly_scores)
    for i in range(len(anomaly_scores)):
        if (anomaly_scores[i] > 0.5):
            if i in actual:
                correctly_detected_x.append(i)
                correctly_detected_y.append(anomaly_scores[i])
            else:
                detected_anomalies.append(i)
    if (ENABLE_PLOTS):
        fig, ax1 = pyplot.subplots(figsize=(20,8))
        ax1.plot(anomaly_scores, '-p', markevery=actual, c='blue', mfc='red',label='anomaly score', mec='red')
        ax1.plot(anomaly_scores, '-p', markevery=correctly_detected_x, c='blue', mfc='green',label='anomaly score', mec='green')
        ax1.plot(anomaly_scores, '-p', markevery=detected_anomalies, c='blue', mfc='orange',label='anomaly score', mec='orange')
        fig.suptitle("Anomaly scores for " + type + " anomaly injection")
        fig.savefig("anomaly_plots/" + DATASET + "/" + type)
    print(len(correctly_detected_x))
    return anomaly_scores, detected_anomalies

# %%
def plot_anomaly_scores(anomalies, detected, type):
    pyplot.plot(anomalies, '-p', markevery=detected, c='blue', mfc='red',label='anomaly score', mec='red', title="Anomaly scores for " + type + " anomaly injection")


# %% [markdown]
# <a id='section5'></a>
# ## Section 5: Running the system

# %% [markdown]
# This section contains the code to run the system on data sets 1 and 2 (note these runs use the final model found for the Prediction Module during hyper parameter tuning)
# 
# 1. [HH101](#hh101)
# 2. [HH102](#hh102)

# %% [markdown]
# <a id='hh101'></a>
# 
# ### HH101 

# %%
DATASET= "hh101"
ENABLE_PLOTS = 0
# Data pre-processing 
hh101, hh101_sensor_id_mapping, hh101_reversed_mapping = load_hh101()
hh101_drop = ["D002", "MA016", "LS010", "LS009", "LS013", "LS006", "M006", "MA013", "LS015","T101", "T102", "T103", "T104", "T105"]
hh101 = transform_data(hh101, hh101_sensor_id_mapping) 
# Data pre-processing

# Feature Selection
most_important_features = pca_decomp(hh101, hh101_reversed_mapping)
# Feature Selection

# Set up data for prediction module
hh101_data, plot_data = transform_series_to_supervised(hh101, HH101_PREDICTION_SENSORS, most_important_features, hh101_reversed_mapping)
NUMBER_IN_FEATURES = len(most_important_features)
NUMBER_PREDICTIONS = len(HH101_PREDICTION_SENSORS)
train_x, train_y, test_x, test_y, anomaly_x, anomaly_y = create_train_test_split(hh101_data)
# Set up data for prediction module

# Create various data plots
if (ENABLE_PLOTS):
    plot_normalised_sensor_activations(plot_data, hh101_reversed_mapping)
    plot_cleaned_data(hh101, hh101_reversed_mapping)
    find_correlations(plot_data, hh101_reversed_mapping)
# Create various data plots

# Find best hyper-params and train final prediction model, or load best model from file
hh101_final_model, hh101_history = get_final_model(train_x, train_y, test_x, test_y)
# Find best hyper-params and train final prediction model, or load best model from file

hh101_baseline, hh101_baseline_history = train_baseline(create_baseline(), train_x, train_y, test_x, test_y)
# hh101_final_pred_error, hh101_r2_s, predictions, df = predict(anomaly_x, hh101_final_model, anomaly_y)
# Inject anomalies into hold-out data
# (random_anomaly_x, random_anomaly_y) = generate_anomalous_data(get_stats(np.row_stack((test_y, train_y, anomaly_y))), anomaly_x, anomaly_y)
# Inject anomalies into hold-out data


# %% [markdown]
# Load the best model and make predictions

# %%
print(hh101_r2_s)
# from numpy.random import seed
# seed(1)
# ENABLE_PLOTS = 1

# random_anomaly, activate_anomaly = generate_anomalous_data(get_stats(np.row_stack((test_y, train_y, anomaly_y))), anomaly_x, anomaly_y, hh101_reversed_mapping)
# # random_anomaly = anomaly_x[:100], anomaly_y[:100], [0], anomaly_y[:100]
# detect_anomalies(random_anomaly, activate_anomaly,  final_model)
# DATASET="hh101"

# hh101_error, hh101_r2s, hh101_predictions, sensor_errors = predict(anomaly_x, final_model, anomaly_y)
# b = sns.boxplot(data = sensor_errors, showfliers=False, whis=1.5)
# b.set_ylabel("MSE", fontsize=12)
# b.set_xlabel("Sensor", fontsize=12)
# b.set_title("MSE per sensor in " + DATASET +" model", fontsize=14)
# # # b.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16])
# sns.set(rc = {'figure.figsize':(15,12)})
# f = b.get_figure()
# f.savefig("data_plots/" + DATASET+"/sensor_mse")
# sns.despine(offset = 5, trim = True)
# pyplot.boxplot(hh101_error, showfliers=False)
# predictions = predict(random_anomaly_x, final_model)
# anomalies = detect_anomaly(random_anomaly_y, predictions)
# detected = []
# for i, anomaly in enumerate(anomalies):
#     if anomaly > 0.6:
#         detected.append(i)
# pyplot.plot(anomalies, '-p', markevery=detected, c='blue', mfc='red',label='anomaly score', mec='red')

# # plt.plot(range(10))
# # plt.axvspan(3, 6, color='red', alpha=0.5)
# # plt.show()

# %%


# %%
# %load_ext tensorboard
# %tensorboard --logdir /Users/christinaspanellis/Desktop/MAC/AAL_ICL/tensorflow/hh102/logs/

# %%
# plot_cleaned_data(hh101, hh101_reversed_mapping, "hh101", "/cleaned_data.png")
# plot_normalised_sensor_activations("hh101", plot_data, hh101_reversed_mapping)

# %% [markdown]
# <a id='hh102'></a>
# 
# ### HH102

# %% [markdown]
# 

# %%
DATASET= "hh102"
ENABLE_PLOTS = 0
# Data pre-processing 
print("Processing dataset...")
hh102, hh102_sensor_id_mapping, hh102_reversed_mapping = load_hh102()
hh102_drop = ["LS013", "LS006", "LS011", "M011", "MA010", "LS012", "LS015", "LS009", "LS023", "T101", "T102", "T103", "T104"]
hh102 = transform_data(hh102, hh102_sensor_id_mapping) 
print("Dataset processed")
# Data pre-processing

# Feature Selection
print("Selecting features...")
most_important_features = pca_decomp(hh102, hh102_reversed_mapping)
# Feature Selection
print("Features selected")

# Set up data for prediction module
print("Transforming data to supervised format...")
hh102_data, plot_data = transform_series_to_supervised(hh102, HH102_PREDICTION_SENSORS, most_important_features, hh102_reversed_mapping)
NUMBER_IN_FEATURES = len(most_important_features)
NUMBER_PREDICTIONS = len(HH102_PREDICTION_SENSORS)
train_x, train_y, test_x, test_y, anomaly_x, anomaly_y = create_train_test_split(hh102_data)
# Set up data for prediction module
print("Data transformed")

# Create various data plots
if (ENABLE_PLOTS):
    print("Plotting graphs..")
    plot_normalised_sensor_activations(plot_data, hh102_reversed_mapping)
    plot_cleaned_data(hh102, hh102_reversed_mapping)
    find_correlations(plot_data, hh102_reversed_mapping)
    print("Graphs plotted")
# Create various data plots


# Find best hyper-params and train final prediction model, or load best model from file
hh102_final_model, hh102_history = get_final_model(train_x, train_y, test_x, test_y)
# Find best hyper-params and train final prediction model, xor load best model from file
hh102_final_pred_error, hh101_r2_s, predictions, df = predict(anomaly_x, hh102_final_model, anomaly_y)

# Inject anomalies into hold-out data
# random_anomaly, zero_anomaly, activate_anomaly, deactivate_anomaly = generate_anomalous_data(get_stats(np.row_stack((test_y, train_y, anomaly_y))), anomaly_x, anomaly_y)
# Inject anomalies into hold-out data

# %% [markdown]
# Running this cell will create various plots of the data set

# %%

hh102_final_pred_error = hh102_final_pred_error[:len(hh101_final_pred_error)]
x = list(range(0,len(hh102_final_pred_error)))
# pyplot.plot(x,hh101_final_pred_error, label='HH101 prediction error', linewidth=2)
pyplot.figure(figsize=(20,6))
pyplot.plot(x,hh102_final_pred_error ,label='HH102 prediction error', linewidth=0.5)
pyplot.xlabel("SEW")
pyplot.ylabel("MSE")
pyplot.title("Prediction error of the HH101 and HH102 final models.")
pyplot.tight_layout()
pyplot.legend()
pyplot.savefig("best_models/final_errors")
# random_anomaly, activate_anomaly = generate_anomalous_data(get_stats(np.row_stack((test_y, train_y, anomaly_y))), anomaly_x, anomaly_y, hh102_reversed_mapping)
# # random_anomaly = anomaly_x[:100], anomaly_y[:100], [0], anomaly_y[:100]
# detect_anomalies(random_anomaly, activate_anomaly,  final_model)

# %%
# df = pandas.DataFrame({"HH101": hh101_error, "HH102": hh102_error[:len(hh101_error)]})
# # pyplot.boxplot([hh101_error, hh102_error], whis=(5,90), labels=["HH101", "HH102"], meanline=True, showfliers=False)
# b = sns.boxplot(data = df, showfliers=False, whis=1.5)
# b.set_ylabel("MSE", fontsize=12)
# b.set_xlabel("Data set", fontsize=12)
# b.set_title("MSE of final prediction models", fontsize=14)
# # # b.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16])
# sns.set(rc = {'figure.figsize':(15,12)})
# f = b.get_figure()
# f.savefig("data_plots/final_mse")
# sns.despine(offset = 5, trim = True)

# pyplot.xticks([])
# b = sns.boxplot(data = sensor_errors, showfliers=False, whis=1.5)
# b.set_ylabel("MSE", fontsize=12)
# b.set_xlabel("Sensor", fontsize=12)
# b.set_title("MSE per sensor in " + DATASET +" model", fontsize=14)
# # # b.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.14, 0.16])
# sns.set(rc = {'figure.figsize':(15,12)})
# f = b.get_figure()
# f.savefig("data_plots/" + DATASET+"/sensor_mse")
# sns.despine(offset = 5, trim = True)
# print(len(hh102_error[0]))
# dataset = pandas.DataFrame({'y': anomaly_y, 'y_pred': hh102_predictions}, columns=['y', 'y_pred]'])

# g = sns.lmplot(x = 'y', y='y_pred',data= dataset, hue='tag')
# g.fig.suptitle('True Vs Pred', y=1.02)
# g.set_axis_labels('y_true', 'y_pred')


