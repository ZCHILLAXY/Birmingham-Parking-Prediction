#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Birmingham_Parking_Prediction -> gru
@Author ：Zhuang Yi
@Date   ：2019/10/9 20:55
=================================================='''
from keras import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras.engine.saving import load_model
from keras.layers import GRU, Dense, K, Dropout
from data_processing.split_train_test import Split_Dataset
from util import GRU_MODEL_PATH
import matplotlib.pyplot as plt
from sklearn import metrics
import pandas as pd


class MY_GRU(object):
	def __init__(self):
		self.sd = Split_Dataset()
		self.training_x, self.training_y, self.testing_x, self.testing_y = self.sd.split_train_test_rnn()
		dt = self.sd.dataset.reset_index()['PercentOccupied']
		self.test_index = dt.index[len(self.training_y)+1:]


	def huber_loss(self, y_true, y_pred, delta=1):
		return K.mean((K.sqrt(1 + K.square((y_pred - y_true) / delta)) - 1) * delta ** 2, axis=-1)
		# if (abs(y_true-y_pred) <= delta) is True:
		# 	return K.mean(0.5 * (K.square(y_pred - y_true)), axis=-1)
		# else:
		# 	return K.mean(delta * (abs(y_pred - y_true) - 0.5 * delta), axis=-1)

	def gru_train(self):
		model = Sequential()
		model.add(GRU(50, input_shape=(self.training_x.shape[1], self.training_x.shape[2])))
		model.add(Dense(1))
		model.compile(loss='mean_squared_error', optimizer='adam')

		tensorboard = TensorBoard(log_dir='gru_log')
		checkpoint = ModelCheckpoint(filepath=GRU_MODEL_PATH, monitor='val_loss', mode='auto')
		callback_lists = [tensorboard, checkpoint]

		history = model.fit(self.training_x, self.training_y,  verbose=2, epochs=300, batch_size=18,
		                    validation_split=0.2,
		                    callbacks=callback_lists)

		plt.plot(history.history['loss'], label='train')
		plt.plot(history.history['val_loss'], label='validation')
		plt.legend()
		plt.show()

		print('Train Successfully')

	def test_predict(self):
		plt.figure(figsize=(16, 6))
		plt.title('GRU Model on Aggregate Data')
		plt.plot(self.training_y, label='Training Actual Occupancy Rate')
		plt.xlabel('Date')
		plt.ylabel('Percent Occupied')
		model = load_model(GRU_MODEL_PATH)
		y_pred = model.predict(self.testing_x)
		testing_y = pd.Series(self.testing_y, index=self.test_index)
		y_pred = pd.Series(y_pred[:, -1], index=self.test_index)
		print(testing_y)
		plt.plot(testing_y, label='Testing Actual Occupancy Rate')
		plt.plot(y_pred, color='purple', label='GRU Predicted Occupancy Rate')
		plt.legend()
		plt.show()
		self.report_metrics(self.testing_y, y_pred)

	def report_metrics(self, y_true, y_pred):
		print("Explained Variance:\n\t", metrics.explained_variance_score(y_true, y_pred))
		print("MAE:\n\t", metrics.mean_absolute_error(y_true, y_pred))

if __name__ == '__main__':
	gru = MY_GRU()
	gru.gru_train()
	gru.test_predict()