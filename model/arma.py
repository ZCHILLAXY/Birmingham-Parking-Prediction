'''=================================================
@Project -> File   ：Birmingham_Parking_Prediction -> arma
@Author ：Zhuang Yi
@Date   ：2019/10/21 12:42
=================================================='''
from sklearn import metrics
from statsmodels.tsa.arima_model import ARMA, ARIMA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from data_processing.split_train_test import Split_Dataset


class MY_ARMA(object):
    def __init__(self):
        self.sd = Split_Dataset()
        self.dt = self.sd.dataset.reset_index()['PercentOccupied']
        self.training_act = self.dt[:1260]

        self.test_index = self.dt.index[1260:]

    def train_arma(self):

        tscv = TimeSeriesSplit(n_splits=10)

        for train_index, test_index in tscv.split(self.dt):
            train = self.dt[train_index]
            test = self.dt[test_index]
        arma_model = ARMA(train, order=(18, 0))
        results_AR = arma_model.fit(disp=-1)
        plt.figure(figsize=(16, 6))
        plt.title('ARMA Model on Aggregate Data')
        plt.plot(train, label='Training Actual Occupancy Rate')
        plt.xlabel('Date')
        plt.ylabel('Percent Occupied')
        y_pred_AR = pd.Series(results_AR.forecast(steps=len(test))[0], index=test.index)
        plt.plot(test, label='Testing Actual Occupancy Rate')
        plt.plot(y_pred_AR, color='purple', label='ARMA Predicted Occupancy Rate')
        plt.legend()

        plt.show()
        print('ARMA Model Metrics on Test Data')
        self.report_metrics(test.squeeze(), y_pred_AR.squeeze())

    def report_metrics(self, y_true, y_pred):
        print("Explained Variance:\n\t", metrics.explained_variance_score(y_true, y_pred))
        print("MAE:\n\t", metrics.mean_absolute_error(y_true, y_pred))
        print("MSE:\n\t", metrics.mean_squared_error(y_true, y_pred))

if __name__ == '__main__':
    arma = MY_ARMA()
    arma.train_arma()

