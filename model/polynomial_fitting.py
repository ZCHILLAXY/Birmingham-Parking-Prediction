#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Birmingham_Parking_Prediction -> polynomial_fitting
@Author ：Zhuang Yi
@Date   ：2019/10/10 11:52
=================================================='''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Perceptron
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from data_processing.split_train_test import Split_Dataset
from sklearn.model_selection import KFold
import pandas as pd

class Poly_Fit(object):
	def __init__(self):
		self.sd = Split_Dataset()
		self.degrees = np.arange(1, 10)
		self.data = self.sd.split_train_test_normal()
		self.mses = []
		self.best = []
		self.min_mse, self.min_deg = 1e10, 0
		dt = self.sd.dataset.reset_index()['PercentOccupied']
		self.test_index = dt.index[1260:]

	def poly_fitting_degrees(self, x_train, y_train, x_test, y_test):
		for deg in self.degrees:
			poly = PolynomialFeatures(degree=deg, include_bias=True)
			x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))

			poly_reg = LinearRegression(fit_intercept=True)
			poly_reg.fit(x_train_poly, y_train)
			# print(poly_reg.coef_,poly_reg.intercept_)
			x_test_poly = poly.fit_transform(x_test.reshape(-1, 1))
			y_test_pred = poly_reg.predict(x_test_poly)
			poly_mse = mean_squared_error(y_test, y_test_pred)
			self.mses.append(poly_mse)
			# degree交叉验证
			if self.min_mse > poly_mse:
				self.min_mse = poly_mse
				self.min_deg = deg
			print('degree = %s, MSE = %.2f' % (deg, poly_mse))
			if deg == 7:
				self.best.append(y_test_pred)

	def poly_fitting(self):
		train_index = self.data[0:1260].index.values
		test_index = self.data[1260:1386].index.values
		train_x, train_y = train_index % 126, self.data[train_index].values
		test_x, test_y = test_index % 126, self.data[test_index].values
		self.poly_fitting_degrees(train_x, train_y, test_x, test_y)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		ax.plot(self.degrees, self.mses)
		# ax.set_yscale('log')
		ax.set_xlabel('Components')
		ax.set_ylabel('MSE')
		ax.set_title('Best degree = %s, MSE = %.2f' % (self.min_deg, self.min_mse))
		plt.show()
		self.test_predict()

	def test_predict(self):
		plt.figure(figsize=(16, 6))
		plt.title('Polynomial_Fitting Model on Aggregate Data')
		plt.plot(self.data[0:1260], label='Training Actual Occupancy Rate')
		plt.xlabel('Date')
		plt.ylabel('Percent Occupied')
		y_pred = self.best[0]
		y_pred = pd.Series(y_pred, index=self.test_index)
		plt.plot(self.data[1260:1386], label='Testing Actual Occupancy Rate')
		print(self.data[1260:1386])
		plt.plot(y_pred, color='purple', label='Polynomial_Fitting Predicted Occupancy Rate')
		plt.legend()
		plt.show()
		self.report_metrics(self.data[1260:1386], y_pred)

	def report_metrics(self, y_true, y_pred):
		print("Explained Variance:\n\t", metrics.explained_variance_score(y_true, y_pred))
		print("MAE:\n\t", metrics.mean_absolute_error(y_true, y_pred))
		print("MSE:\n\t", metrics.mean_squared_error(y_true, y_pred))




if __name__ == '__main__':
	pf = Poly_Fit()
	pf.poly_fitting()

