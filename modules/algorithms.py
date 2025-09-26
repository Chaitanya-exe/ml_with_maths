import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

class Algorithms:
    def __init__(self):
        pass

    def __cost_fxn(self, X, y, theta):
        m = len(y)
        predictions = X.dot(theta)
        error = predictions - y
        cost = (1/(2*m)) * np.sum(error**2)
        return cost


    def batch_gradient_descent(self):

        # loading clean dataset
        data = fetch_california_housing(as_frame=True)
        df = data.frame

        # initiating features and inputs
        X = df[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 
        'Population', 'AveOccup', 'Latitude', 'Longitude']].values

        y = df['MedHouseVal'].values

        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)

        X = (X - X_mean)/X_std

        # Adding bias terms
        X_b = np.c_[np.ones((X.shape[0],1)), X]

        m, n = X_b.shape

        y = y.reshape(-1, 1)

        # initialie parameters
        theta = np.zeros((n, 1))

        # Gradient Descent parameters
        alpha = 0.1
        iterations = 200
        costs = []

        for i in range(iterations):
            predictions = X_b.dot(theta)
            gradients = (1 / m) * X_b.T.dot(predictions - y)
            theta = theta - alpha * gradients
            cost = self.__cost_fxn(X_b, y, theta)
            costs.append(cost)

        print("Final parameters (theta):\n", theta.ravel())
        print("Final costs: ", costs[-1])

        #predictions
        y_pred = X_b.dot(theta)

        mse = mean_squared_error(y, y_pred)
        r = r2_score(y, y_pred)
        print("Mean Squared error: ", mse)
        print("R2 score: ", r)

        # plotting of cost function decrease
        plt.plot(range(iterations), costs, 'b-')
        plt.xlabel("iteration")
        plt.ylabel("Cost function (MSE)")
        plt.title("Cost function convergence")
        plt.show()




