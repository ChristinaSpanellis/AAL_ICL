from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
import pandas
from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler


def parse(x):
        return datetime.strptime(x, '%Y %m %d %H')
class preprocessor:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.test_x = None
        self.test_y = None
        self.train_x = None
        self.train_x = None
        self.scaler = None
    # load data
    
    def load_data(self):
        dataset = read_csv('raw.csv',  parse_dates = [['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
        dataset.drop('No', axis=1, inplace=True)
        # manually specify column names
        dataset.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
        dataset.index.name = 'date'
        # mark all NA values with 0
        dataset['pollution'].fillna(0, inplace=True)
        # drop the first 24 hours
        dataset = dataset[24:]
        dataset.to_csv('pollution.csv')

       # load dataset
        dataset = read_csv('pollution.csv', header=0, index_col=0)
        values = dataset.values
        # integer encode direction
        encoder = LabelEncoder()
        values[:,4] = encoder.fit_transform(values[:,4])
        # ensure all data is float
        values = values.astype('float32')
        # normalize features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = self.scaler.fit_transform(values)
        # frame as supervised learning
        reframed = self.series_to_supervised(scaled, 1, 1)
        # drop columns we don't want to predict
        reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
        values = reframed.values
        n_train_hours = 365 * 24
        train = values[:n_train_hours, :]
        test = values[n_train_hours:, :]
        # split into input and outputs
        self.train_x, self.train_y = train[:, :-1], train[:, -1]
        self.test_x, self.test_y = test[:, :-1], test[:, -1]
        # reshape input to be 3D [samples, timesteps, features]
        self.train_x = self.train_x.reshape((self.train_x.shape[0], 1, self.train_x.shape[1]))
        self.test_x = self.test_x.reshape((self.test_x.shape[0], 1, self.test_x.shape[1]))
        print(self.train_x.shape, self.train_y.shape, self.test_x.shape, self.test_y.shape)


    def series_to_supervised(self, data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    def load_adlnormal(self):
        replacement_mapping_dict = {
            "M01" : 0,
            "M02" : 1,
            "M03" : 2,
            "M04" : 3,
            "M05" : 4,
            "M06" : 5,
            "M07" : 6,
            "M08" : 7,
            "M09" : 8,
            "M10" : 9,
            "M11" : 10,
            "M12" : 11,
            "M13" : 12,
            "M14" : 13,
            "M15" : 14,
            "M16" : 15,
            "M17" : 16,
            "M18" : 17,
            "M19" : 18,
            "M20" : 19,
            "M21" : 20,
            "M22" : 21,
            "M23" : 22,
            "M24" : 23,
            "M25" : 24,
            "M26" : 25,
            "I01" : 26,
            "I02" : 27,
            "I03" : 28,
            "I04" : 29,
            "I05" : 30,
            "I06" : 31,
            "I07" : 32,
            "I08" : 33,
            "D01" : 34,
            "AD1-A" : 35,
            "AD1-B" : 36,
            "AD1-C" : 37,
        }
        dataset = read_csv('datasets/adlnormal/data.csv')
        dataset = dataset[dataset["SensorId"] != "E01"]
        dataset = dataset[dataset["SensorId"] != "asterisk"]

        dataset["SensorValue"].replace("ON", 1.0, inplace=True)
        dataset["SensorValue"].replace("OFF", 0.0, inplace=True)
        dataset["SensorValue"] = dataset["SensorValue"].replace({"ABSENT" : 1.0, "PRESENT" : 0.0})
        dataset["SensorValue"] = dataset["SensorValue"].replace({"OPEN" : 1.0, "CLOSE" : 0.0})

        dataset["SensorId"].replace(replacement_mapping_dict, inplace=True)
        dataset.drop('Event', axis=1, inplace=True)
        data = []
        current_date = dataset.iloc[0]["Date"]
        first_row_time = dataset.iloc[0]["Time"].split(".")
        first_row_time = first_row_time[:1]
        current_time = ((datetime.strptime(first_row_time[0], "%H:%M:%S").second // 10 ) + 1 ) * 10 
        if (current_time > 50):
            current_time = 60
        sensor_vals = [0.0] * 39
        first_time = datetime.strptime(first_row_time[0], "%H:%M:%S")
        date_and_time = []
        date_and_time.append(current_date)
        date_and_time.append(str(first_time.hour) + ":" + str(first_time.minute) + "." + str(current_time))
        for i, row in dataset.iterrows():
            date = row["Date"]
            time = row["Time"].split(".")
            time = time[:1]
            time = datetime.strptime(time[0], "%H:%M:%S")
            if ((current_time == 60 and time.second < current_time ) or time.second >= current_time):
                date_and_time.extend(sensor_vals)
                data.append(date_and_time)
                date_and_time = []
                current_date = date
                if (current_time == 60):
                    current_time = 10
                else:
                    current_time += 10
                sensor_vals = [0.0] * 39
                if (sensor_vals[int(row["SensorId"])] != 1.0):
                    sensor_vals[int(row["SensorId"])] = row["SensorValue"]
                date_and_time.append(row["Date"])
                new_time = str(time.hour) + ":" + str(time.minute) + "." + str(current_time)
                date_and_time.append(new_time)
            if (sensor_vals[int(row["SensorId"])] != 1.0):
                sensor_vals[int(row["SensorId"])] = row["SensorValue"]
        new_data = pandas.DataFrame.from_records(data)      
        # print(new_data)
        # sum = (new_data[9] == 1.0).sum()
        # print(sum)
        values = new_data.values
        # specify columns to plot
        groups = []
        for i in range (2,41):
            groups.append(i)
        i = 1
        # plot each column
        pyplot.figure()
        for group in groups:
            print(group)
            pyplot.subplot(len(groups), 1, i)
            pyplot.plot(values[:, group])
            pyplot.title(new_data.columns[group], y=0.5, loc='right')
            i += 1
        pyplot.show()
            



        # now need to create data
        # date,time,0,1,2,3....,37, (time of day?)
        # need to deal with absent etc
        # finding that the data is very sparse because its just binary, either on or off and the off dominates the on
        # TO DO
        # Test with how it is, potentially change to 30 second intervals
        # Get some training / test data splits going
        # Analyse data and see whats cutting



def main():
    # preprocess = preprocessor('raw.csv')
    # preprocess.load_data()
    # LSTM = lstm(8, 1, preprocess)
    preprocess = preprocessor('dsff')
    preprocess.load_adlnormal()

if __name__=="__main__":
    main()
 
        