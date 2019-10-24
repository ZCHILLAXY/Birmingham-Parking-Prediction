#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Birmingham_Parking_Prediction -> fourier_fitting
@Author ：Zhuang Yi
@Date   ：2019/10/11 22:13
=================================================='''


import numpy as np
from scipy.optimize import curve_fit
from sklearn import metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from symfit import parameters, variables, sin, cos, Fit
import pandas as pd
import matplotlib.pyplot as plt

from data_processing.split_train_test import Split_Dataset


class Fourier_Fit(object):
	def __init__(self):
		self.sd = Split_Dataset()
		self.data = self.sd.split_train_test_normal()
		self.components = np.arange(1, 20)
		self.mses = []
		self.best = []
		self.min_mse, self.min_com = 1e10, 0


	def k_fold_validation(self):
		train_index = self.data[0:1260].index.values
		test_index = self.data[1260:1386].index.values
		train_x, train_y = train_index % 126, self.data[train_index].values
		test_x, test_y = test_index % 126, self.data[test_index].values
		self.fourier_fit(train_x, train_y, test_x, test_y)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(self.components, self.mses)
		# ax.set_yscale('log')
		ax.set_xlabel('Components')
		ax.set_ylabel('MSE')
		ax.set_title('Best degree = %s, MSE = %.2f' % (self.min_com, self.min_mse))
		plt.show()
		self.test_predict()

	def fourier_series(self, x, f, n=0):
		"""
        Returns a symbolic fourier series of order `n`.

        :param n: Order of the fourier series.
        :param x: Independent variable
        :param f: Frequency of the fourier series
        """
		# Make the parameter objects for all the terms
		a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
		sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))
		# Construct the series
		series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
						  for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
		return series

	def fourier_model(self, com):
		x, y = variables('x, y')
		w, = parameters('w')
		model_dict = {y: self.fourier_series(x, f=w, n=com)}
		print(model_dict)
		return model_dict

	def fourier_fit(self, x_train, y_train, x_test, y_test):
		for com in self.components:
			fit = Fit(self.fourier_model(com), x=x_train, y=y_train)
			fit_result = fit.execute()
			x_test = x_test.reshape(-1, 1)
			y_test_pred = fit.model(x=x_test, **fit_result.params).y
			fourier_mse = mean_squared_error(y_test, y_test_pred)
			self.mses.append(fourier_mse)
			# degree交叉验证
			if self.min_mse > fourier_mse:
				self.min_mse = fourier_mse
				self.min_com = com
			print('degree = %s, MSE = %.2f' % (com, fourier_mse))
			if com == 18:
				self.best.append(y_test_pred)

	def test_predict(self):
		plt.figure(figsize=(16, 6))
		plt.title('Fourier_Fitting Model on Aggregate Data')
		plt.plot(self.data[0:1260], label='Training Actual Occupancy Rate')
		plt.xlabel('Date')
		plt.ylabel('Percent Occupied')
		y_pred = self.best[0].reshape(1, -1)[0]
		y_pred = pd.Series(y_pred, index=list(range(1259, 1385)))
		plt.plot(self.data[1260:1386], label='Testing Actual Occupancy Rate')
		plt.plot(y_pred, color='purple', label='Fourier_Fitting Predicted Occupancy Rate')
		plt.legend()
		plt.show()
		self.report_metrics(self.data[1260:1386], y_pred)

	def report_metrics(self, y_true, y_pred):
		print("Explained Variance:\n\t", metrics.explained_variance_score(y_true, y_pred))
		print("MAE:\n\t", metrics.mean_absolute_error(y_true, y_pred))
		print("MSE:\n\t", metrics.mean_squared_error(y_true, y_pred))


if __name__ == '__main__':
	ff = Fourier_Fit()
	ff.k_fold_validation()
