#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Birmingham_Parking_Prediction -> fourier_fitting
@Author ：Zhuang Yi
@Date   ：2019/10/11 22:13
=================================================='''


import numpy as np
from scipy.optimize import curve_fit

from data_processing.split_train_test import Split_Dataset


class Fourier_Fit(object):
	def __init__(self):
		self.sd = Split_Dataset()
		self.data = self.sd.split_train_test_normal()

	def fourier(self, x, *a):
		tau = 0.045
		ret = a[0] * np.cos(np.pi / tau * x)
		for deg in range(1, len(a)):
			ret += a[deg] * np.cos((deg + 1) * np.pi / tau * x)
		return ret

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

	def fourier_fit(self):
		popt, pcov = curve_fit(self.fourier, )
