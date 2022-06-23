from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from pandas import DataFrame
from pandas import concat
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

def main():
    preprocess = preprocessor('raw.csv')
    preprocess.load_data()
    LSTM = lstm(8, 1, preprocess)


if __name__=="__main__":
    main()
 
        