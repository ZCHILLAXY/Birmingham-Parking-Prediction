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
from sklearn.metrics import mean_absolute_error, r2_score
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
		self.min_mae, self.min_deg = 1e10, 0

	def poly_fitting_degrees(self, x_train, y_train, x_test, y_test):
		for deg in self.degrees:
			poly = PolynomialFeatures(degree=deg, include_bias=False)
			x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))

			poly_reg = LinearRegression(fit_intercept=True)
			poly_reg.fit(x_train_poly, y_train)
			# print(poly_reg.coef_,poly_reg.intercept_)
			x_test_poly = poly.fit_transform(x_test.reshape(-1, 1))
			y_test_pred = poly_reg.predict(x_test_poly)
			poly_mae = mean_absolute_error(y_test, y_test_pred)
			self.mses.append(poly_mae)
			# degree交叉验证
			if self.min_mae > poly_mae:
				self.min_mae = poly_mae
				self.min_deg = deg
			print('degree = %s, MAE = %.2f' % (deg, poly_mae))
			if deg == 6:
				self.best.append(y_test_pred)

	def k_fold_validation(self):
		kf = KFold(n_splits=10)
		for train_index, test_index in kf.split(self.data):
			train_x, train_y = train_index, self.data[train_index]
			test_x, test_y = test_index, self.data[test_index]
			self.poly_fitting_degrees(train_x, train_y, test_x, test_y)
		fig = plt.figure()
		ax = fig.add_subplot(111)
		for i in range(0, 10):
			ax.plot(self.degrees, self.mses[9 * i:9 * i+9], label="set%d" % (i+1))
			plt.legend(loc='best')
		# ax.set_yscale('log')
		ax.set_xlabel('Degree')
		ax.set_ylabel('MSE')
		ax.set_title('Best degree = %s, MAE = %.2f' % (self.min_deg, self.min_mae))
		plt.show()
		self.test_predict()

	def test_predict(self):
		plt.figure(figsize=(16, 6))
		plt.title('Polynomial_Fitting Model on Aggregate Data')
		plt.plot(self.data[0:1247], label='Training Actual Occupancy Rate')
		plt.xlabel('Date')
		plt.ylabel('Percent Occupied')
		y_pred = self.best[1]
		print(y_pred)
		y_pred = pd.Series(y_pred, index=list(range(1247, 1386)))
		plt.plot(self.data[1247:1386], label='Testing Actual Occupancy Rate')
		plt.plot(y_pred, color='purple', label='Polynomial_Fitting Predicted Occupancy Rate')
		plt.legend()
		plt.show()
		self.report_metrics(self.data[1247:1386], y_pred)

	def report_metrics(self, y_true, y_pred):
		print("Explained Variance:\n\t", metrics.explained_variance_score(y_true, y_pred))
		print("MAE:\n\t", metrics.mean_absolute_error(y_true, y_pred))




if __name__ == '__main__':
	pf = Poly_Fit()
	pf.k_fold_validation()

