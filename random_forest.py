from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

class RandomForest:

    clf = RandomForestClassifier(random_state=0, n_estimators=1500, max_depth=16)

    def train(self, train_x, train_y, feature_names):

        train_y = train_y.squeeze()

        self.clf.fit(train_x, train_y)
        feature_importance = self.clf.feature_importances_
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = np.argsort(feature_importance)
        sorted_features = [feature_names[int(x)] for x in sorted_idx]

        print("---------")
        print("Random Forest Relative Feature Importance:")
        for x in range(len(sorted_features)):
            print("{0}: {1}".format(feature_importance[x],sorted_features[x]))
        print("---------")

    def predict(self, test_x, test_y):
        predictions = self.clf.predict(test_x)
        mse = mean_squared_error(predictions, test_y)

        return mse, predictions
