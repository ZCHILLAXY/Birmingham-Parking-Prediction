#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Birmingham_Parking_Prediction -> split_train_test
@Author ：Zhuang Yi
@Date   ：2019/10/9 20:26
=================================================='''

from data_processing.data_cleaning import Clean_Data
from sklearn.model_selection import TimeSeriesSplit
import pandas as pd
import numpy as np


pd.set_option('display.max_columns', 500)
pd.options.display.expand_frame_repr = False

class Split_Dataset(object):
	def __init__(self):
		cd = Clean_Data()
		self.dataset = cd.fill_na()
		self.columns = self.dataset.columns
		self.values = self.dataset.values

	def split_train_test_normal(self):
		data_use = self.dataset.reset_index(drop=False)['PercentOccupied']
		print(data_use)
		return data_use



	def split_train_test_rnn(self):
		reframed = self.series_to_supervised(self.values, self.columns)
		n_values = reframed.values
		n_train_hours = 1385
		train =n_values[:int(n_train_hours * 0.8), :]
		test = n_values[int(n_train_hours * 0.8):, :]
		train_x, train_y = train[:, :-1], train[:, -1]
		test_x, test_y = test[:, :-1], test[:, -1]
		train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
		test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))
		return train_x, train_y, test_x, test_y

	def split_train_test_gru(self):
		data = self.values
		sequence_length = 19
		result = []
		for index in range(len(data) - sequence_length + 1):
			result.append(data[index: index + sequence_length])

		result = np.array(result)

		row = 1242  # 1260-19
		train = result[:row, :]
		np.random.shuffle(train)
		x_train = train[:, :-1]
		y_train = train[:, -1]
		x_test = result[row:, :-1]
		y_test = result[row:, -1]

		x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
		x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

		print('X_train shape:', x_train.shape)
		print('y_train shape:', y_train.shape)
		print('X_test shape:', x_test.shape)
		print('y_test shape:', y_test.shape)

		return x_train, y_train, x_test, y_test


	def series_to_supervised(self, data, columns, n_in=1, n_out=1, dropnan=True):
		"""
		Frame a time series as a supervised learning dataset.
		Arguments:
			data: Sequence of observations as a list or NumPy array.
			n_in: Number of lag observations as input (X).
			n_out: Number of observations as output (y).
			dropnan: Boolean whether or not to drop rows with NaN values.
		Returns:
			Pandas DataFrame of series framed for supervised learning.
		"""
		n_vars = 1 if type(data) is list else data.shape[1]
		df = pd.DataFrame(data)
		cols, names = list(), list()
		# input sequence (t-n, ... t-1)
		for i in range(n_in, 0, -1):
			cols.append(df.shift(i))
			names += [('%s%d(t-%d)' % (columns[j], j + 1, i)) for j in range(n_vars)]
		# forecast sequence (t, t+1, ... t+n)
		for i in range(0, n_out):
			cols.append(df.shift(-i))
			if i == 0:
				names += [('%s%d(t)' % (columns[j], j + 1)) for j in range(n_vars)]
			else:
				names += [('%s%d(t+%d)' % (columns[j], j + 1, i)) for j in range(n_vars)]
		# put it all together
		agg = pd.concat(cols, axis=1)
		agg.columns = names
		# drop rows with NaN values
		if dropnan:
			clean_agg = agg.dropna()
		return clean_agg




if __name__ == '__main__':
	sd = Split_Dataset()
	sd.split_train_test_gru()
