#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   ：Birmingham_Parking_Prediction -> data_cleaning
@Author ：Zhuang Yi
@Date   ：2019/10/9 15:35
=================================================='''
import datetime

import pandas as pd
from util import DATA_SET_PATH
from datetime import timedelta

pd.set_option('display.max_columns', 500)


class Clean_Data(object):
	def __init__(self):
		self.data_set = pd.read_csv(DATA_SET_PATH)

	def show_raw_data(self):
		print(self.data_set.head(10))
		print(len(self.data_set))
		print(self.data_set.dtypes)

	def feature_add(self):
		ds_clean = self.data_set.copy()
		ds_clean.LastUpdated = ds_clean.LastUpdated.astype('datetime64')
		ds_clean['PercentOccupied'] = ds_clean.Occupancy / ds_clean.Capacity
		ds_clean['date'] = ds_clean.LastUpdated.dt.date
		ds_clean['day_of_week'] = ds_clean.LastUpdated.dt.dayofweek
		ds_clean['date_time_halfhour'] = ds_clean.LastUpdated.dt.round('30min')
		ds_clean['time'] = ds_clean.date_time_halfhour.dt.time
		# print(ds_clean.head())   # add some tags
		ds_clean = ds_clean[ds_clean.time > datetime.time(7, 30)]  # cut time before 7:30
		ds_clean = ds_clean.drop_duplicates()  # drop  duplicates
		ds_clean.Occupancy = ds_clean.apply(lambda x: max(0, min(x['Capacity'], x['Occupancy'])), axis=1)
		ds_clean['PercentOccupied'] = ds_clean.Occupancy / ds_clean.Capacity  # limit the percent from 0 to 100
		ds_agg_dthh = ds_clean.groupby('date_time_halfhour').agg(
			{'Occupancy': ['sum', 'count'], 'Capacity': ['sum', 'count']})
		ds_agg_dthh['PercentOccupied'] = ds_agg_dthh.Occupancy['sum'] / ds_agg_dthh.Capacity['sum']
		ds_agg_dthh.drop(columns=['Occupancy', 'Capacity'], inplace=True)
		ds_agg_dthh.drop([pd.Timestamp('2016-10-28 08:00:00'), pd.Timestamp('2016-12-13 13:30:00')], inplace=True)
		return ds_clean, ds_agg_dthh

	def fill_with_week_prior(self, df, column, year, month, day, hour, minutes):
		df.loc[pd.to_datetime(datetime.datetime(year, month, day, hour, minutes)), column] = \
			df.loc[
				pd.to_datetime(datetime.datetime(year, month, day, hour, minutes) + timedelta(days=-7)), column].values[
				0]

	def fill_na(self):
		"""
		# All of 10/20 and 10/21 are missing
		# 10/30 missing 16:00 and 16:30
		# 11/18 missing 9:00
		# 11/25 missing 8:30
		# 12/14 missing 11:00
		# 10/28 and 12/13 dropped times as noted above
		:return:
		"""
		hc_ds, c_ds = self.feature_add()
		clear_ds = c_ds.copy()
		for hour in range(8, 17):
			for half_hour in [0, 30]:
				self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 10, 20, hour, half_hour)
				self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 10, 21, hour, half_hour)
				self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 12, 3, hour, half_hour)
				self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 12, 4, hour, half_hour)
		self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 10, 30, 16, 0)
		self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 10, 30, 16, 30)
		self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 11, 18, 9, 0)
		self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 11, 25, 8, 30)
		self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 12, 14, 11, 0)
		self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 10, 28, 8, 0)
		self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 12, 13, 13, 30)
		clear_ds.sort_index(inplace=True)
		return clear_ds


	def fill_parking_na(self):
		hc_ds, c_ds = self.feature_add()
		hclear_ds = hc_ds.copy()
		hclear_ds.drop(columns=['LastUpdated', 'date_time_halfhour', 'date'], inplace=True)
		# for hour in range(8, 17):
		# 	for half_hour in [0, 30]:
		# 		self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 10, 20, hour, half_hour)
		# 		self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 10, 21, hour, half_hour)
		# 		self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 12, 3, hour, half_hour)
		# 		self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 12, 4, hour, half_hour)
		# self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 10, 30, 16, 0)
		# self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 10, 30, 16, 30)
		# self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 11, 18, 9, 0)
		# self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 11, 25, 8, 30)
		# self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 12, 14, 11, 0)
		# self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 10, 28, 8, 0)
		# self.fill_with_week_prior(clear_ds, 'PercentOccupied', 2016, 12, 13, 13, 30)
		hclear_ds.sort_index(inplace=True)
		print(hclear_ds)




if __name__ == '__main__':
	cd = Clean_Data()
	cd.fill_parking_na()

