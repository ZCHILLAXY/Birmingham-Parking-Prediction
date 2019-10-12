#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Birmingham_Parking_Prediction -> data_plot
@Author ：Zhuang Yi
@Date   ：2019/10/9 15:54
=================================================='''


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import pandas as pd

from data_processing.data_cleaning import Clean_Data


class Plot_Data(object):
	def __init__(self):
		self.cd = Clean_Data()
		self.dataset_1, self.dataset_2 = self.cd.feature_add()
		self.clear_dataset = self.cd.fill_na()


	def sample_plots_by_scn(self, df, num_graphs, num_per_row, fig_width=16, hspace=0.6):
		"""Print a sample of the data by Parking location, identified with the field SystemCodeNumber
		Parameters:
		num_graphs: Number of locations to make graphs for, ordered by appearance in the dataset.

		num_per_row: Number of columns in subplot.

		fig_width: Used to adjust the width of the subplots figure.  (default=16)

		hspace: Used to adjust whitespace between each row of subplots. (default=0.6)"""
		num_rows = int(np.ceil(num_graphs / num_per_row))
		fig, axes = plt.subplots(nrows=num_rows, ncols=num_per_row, figsize=(fig_width, num_rows * fig_width / 4))
		fig.subplots_adjust(hspace=hspace)
		plt.xticks(rotation=45)
		for i, scn in enumerate(df.SystemCodeNumber.unique()[:num_graphs]):
			temp_df = df[df.SystemCodeNumber == scn]
			ax = axes[i // num_per_row, i % num_per_row]
			ax.plot(temp_df.LastUpdated, temp_df.PercentOccupied)
			ax.set_title('Parking Area: {}'.format(scn))
			ax.set_xlabel('Date')
			ax.set_ylabel('Percent Occupied')
			ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

		for ax in fig.axes:
			plt.sca(ax)
			plt.xticks(rotation=45)
		plt.show()

	def plot_clear_dataset(self, ds):
		plt.figure(figsize=(18, 6))
		plt.plot(ds)
		plt.plot(ds.shift(18))
		plt.xlabel('Date', fontsize=14)
		plt.show()

	def subplots_acf_pacf(self, series):
		fig = plt.figure(figsize=(12, 8))
		ax1 = fig.add_subplot(211)
		fig = plot_acf(series, lags=40, ax=ax1)
		ax2 = fig.add_subplot(212)
		fig = plot_pacf(series, lags=40, ax=ax2)
		plt.show()

	def test_stationarity(self, timeseries, window):

		# Determing rolling statistics
		rolmean = timeseries.rolling(window=window).mean()
		rolstd = timeseries.rolling(window=window).std()

		# Plot rolling statistics:
		fig = plt.figure(figsize=(12, 8))
		orig = plt.plot(timeseries.iloc[window:], color='blue', label='Original')
		mean = plt.plot(rolmean, color='red', label='Rolling Mean')
		std = plt.plot(rolstd, color='black', label='Rolling Std')
		plt.legend(loc='best')
		plt.title('Rolling Mean & Standard Deviation')
		plt.show()

		# Perform Dickey-Fuller test:
		print('Results of Dickey-Fuller Test:')
		dftest = adfuller(timeseries, autolag='AIC')
		dfoutput = pd.Series(dftest[0:4],
		                     index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
		for key, value in dftest[4].items():
			dfoutput['Critical Value (%s)' % key] = value
		print(dfoutput)


	def run(self):
		# self.sample_plots_by_scn(df=self.dataset_1, num_graphs=6, num_per_row=2)
		# self.plot_clear_dataset(self.clear_dataset)
		# self.subplots_acf_pacf(self.clear_dataset)
		self.test_stationarity(self.clear_dataset.squeeze(), 18)



if __name__ == '__main__':
	pdd = Plot_Data()
	pdd.run()
