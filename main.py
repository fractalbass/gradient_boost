from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from gradient_boost import GradientBoost
from random_forest import RandomForest
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

class main:

    def run(self):
        forest = RandomForest()
        boost = GradientBoost()
        train_x, train_y, test_x, test_y, feature_names = self.load("./data/parker_sleeps.csv")
        forest.train(train_x, train_y, feature_names)
        f_accuracy, forest_predictions = forest.predict(test_x, test_y)

        print("                 Results")
        print("--------- Random Forest Technique --------- \n\n")
        print("Feature number\tFeatures\tPrediction\tActual")

        for i in range(len(test_x)):
            print("{0}\t{1}\t{2}\t{3}".format(i, train_x.iloc[i].values, forest_predictions[i], test_y.iloc[i].values[0]))

        print("\nRandom Forest Accuracy (MSE) {:.4f}".format(f_accuracy))

        boost.train(train_x, train_y, feature_names)
        b_accuracy, gb_predictions = boost.predict(test_x, test_y)

        print("\n\n--------- Gradient Boost Technique --------- \n\n")
        print("Feature number\tFeatures\tPrediction\tActual")

        for i in range(len(test_x)):
            print("{0}\t{1}\t{2:.2f}\t{3}".format(i, train_x.iloc[i].values, gb_predictions[i], test_y.iloc[i].values[0]))

        print("\nGradient Boost Accuracy (MSE) {:.4f}".format(b_accuracy))


    def load(self, filename):

        df = pd.read_csv(filename, header=0, delim_whitespace=True)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X = df[['Sunshine']].values.squeeze()
        Y = df[["Nite_Activity"]].values.squeeze()
        Z = df[['Nappytime']].values.squeeze()
        # ax.plot_wireframe(df[['Sunshine']].values, df[["Nite_Activity"]].values, df[['Nappytime']].values, ccount=1)
        ax.plot_trisurf(X, Y, Z, cmap=cm.jet, linewidth=0)
        ax.set_title("Parker Daytime Sleeping Habits")
        ax.legend()
        plt.show()

        feature_cols = ['Sunshine','Nite_Activity']
        target_cols = ['Nappytime']

        features = self.get_columns(df, feature_cols)
        targets = self.get_columns(df, target_cols)

        feature_names = list(features.columns.values)

        X, Y = shuffle(features, targets, random_state=13)
        X = X.astype(np.float32)
        offset = int(X.shape[0] * 0.5)
        train_x, train_y = X[:offset], Y[:offset]
        test_x, test_y = X[offset:], Y[offset:]

        return train_x, train_y, test_x, test_y, feature_names

    def get_columns(self, df, collist):
        new_df = df[collist]
        return new_df


if __name__ == "__main__":
    m = main()
    m.run()




