import sklearn
from math import sqrt
from numpy import concatenate
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
from preprocessing import preprocessor
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import pickle

# NOTES
# 1. Need to decide if I'm going to use lags, if so how many?
# 2. How am I going to aggregate the data?
# 3. Need to just get something really basic working

class lstm:
    def __init__ (self, number_inputs, number_outputs, data, epochs=50, batch_size=72):
        self.number_inputs = number_inputs
        self.number_outputs = number_outputs
        self.epochs = epochs
        self.batch_size = batch_size
        self.data = data
        self.model = None
        self.history = None

    def create_model(self):
        self.model = Sequential()
        self.model.add(LSTM(50, input_shape=(1,self.number_inputs)))
        self.model.add(Dense(self.number_outputs))
        self.model.compile(loss='mae', optimizer='adam')

    def train_model(self):
        self.history = self.model.fit(self.data.train_x, self.data.train_y, epochs=self.epochs, batch_size=self.batch_size, validation_data=(self.data.test_x, self.data.test_y), verbose=2, shuffle=False)

    def evaluate_model(self):
        yhat = self.model.predict(self.data.test_x)
        test_x = self.data.test_x.reshape((self.data.test_x.shape[0], self.data.test_x.shape[2]))
        # invert scaling for forecast
        inv_yhat = concatenate((yhat, test_x[:, 1:]), axis=1)
        inv_yhat = self.data.scaler.inverse_transform(inv_yhat)
        inv_yhat = inv_yhat[:,0]
        # invert scaling for actual
        test_y = self.data.test_y.reshape((len(self.data.test_y), 1))
        inv_y = concatenate((test_y, test_x[:, 1:]), axis=1)
        inv_y = self.data.scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        # calculate RMSE
        rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
        print('Test RMSE: %.3f' % rmse)

    def save_model(self, name):
        pickle.dump(self.model, open(name, 'wb'))


def main():
    preprocess = preprocessor('raw.csv')
    preprocess.load_data()
    LSTM = lstm(8, 1, preprocess)
    LSTM.create_model()
    LSTM.train_model()
    LSTM.evaluate_model()
if __name__=="__main__":
    main()
 