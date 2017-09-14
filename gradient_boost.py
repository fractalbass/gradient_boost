# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
from sklearn import ensemble
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


class GradientBoost:

    #clf = ensemble.GradientBoostingRegressor(n_estimators=100, max_depth=4, min_samples_split=5, learning_rate=0.2)
    clf = XGBRegressor(n_estimators=100, max_depth=4, min_samples_split=5, learning_rate=0.2)

    def train(self, train_x, train_y, feature_names):
        training_fields = train_x
        train_y = train_y.squeeze()
        self.clf.fit(training_fields, train_y.values.tolist())
        feature_importance = self.clf.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        sorted_features = [feature_names[int(x)] for x in sorted_idx]

        print("---------")
        print("Gradient Boost Relative Feature Importance:")
        for x in range(len(sorted_features)):
            print("{0}: {1}".format(feature_importance[x], sorted_features[x]))
        print("---------")

    def predict(self, test_x, test_y):
        predictions = self.clf.predict(test_x)
        mse = mean_squared_error(predictions, test_y)

        return mse, predictions


    # def execute(self, train_x, train_y, test_x, test_y, feature_names):
    #
    #
    #     # Pass into the trained model test_x and see how close we get to test_y (MSE)
    #     mse = mean_squared_error(test_y, clf.predict(test_x))
    #
    #     # #############################################################################
    #     # Plot training deviance
    #
    #     # compute test set deviance
    #     test_score = np.zeros((params['n_estimators'],), dtype=np.float64)
    #
    #     for i, y_pred in enumerate(clf.staged_predict(test_x)):
    #         test_score[i] = clf.loss_(test_y, np.average(y_pred))
    #
    #     plt.figure(figsize=(12, 6))
    #     plt.subplot(1, 2, 1)
    #     plt.title('Deviance')
    #     plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
    #              label='Training Set Deviance')
    #     plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
    #              label='Test Set Deviance')
    #     plt.legend(loc='upper right')
    #     plt.xlabel('Boosting Iterations')
    #     plt.ylabel('Deviance')
    #
    #     # #############################################################################
    #     # Plot feature importance
    #     feature_importance = clf.feature_importances_
    #     # make importances relative to max importance
    #     feature_importance = 100.0 * (feature_importance / feature_importance.max())
    #     sorted_idx = np.argsort(feature_importance)
    #     pos = np.arange(sorted_idx.shape[0]) + .5
    #     plt.subplot(1, 2, 2)
    #     sorted_features = [feature_names[int(x)] for x in sorted_idx]
    #     plt.barh(pos, feature_importance[sorted_idx], align='center')
    #     plt.yticks(pos, sorted_features)
    #     plt.xlabel('Relative Importance')
    #     plt.title('Variable Importance')
    #     plt.show()
    #     return mse