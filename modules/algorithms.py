import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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

    @classmethod
    def sigmoid(cls, z):
        return (1 / (1 + np.exp(-z)))
    
    @classmethod
    def compute_loss(cls, y, y_hat):
        m = y.shape[0]
        return -(1/m) * np.sum(y*np.log(y_hat + 1e-9) + (1-y)*np.log(1-y_hat + 1e-9))
    
    @classmethod
    def predict(cls, X, w, b):
        z = np.dot(X, w) + b
        y_hat = cls.sigmoid(z)
        return (y_hat > 0.5).astype(int)

    @classmethod
    def train_logistic_regression(cls, X, y, lr=0.1, epochs=1000):
        try:
            X = np.array(X)
            y = np.array(y).reshape(-1)
            m, n = X.shape
            w = np.zeros(n)
            b = 0

            for epoch in range(epochs):
                z = np.dot(X, w) + b
                y_hat = cls.sigmoid(z).reshape(-1)
                dw = (1/m) * np.dot(X.T, (y_hat - y))
                db = (1/m) * np.sum(y_hat - y)

                w -= lr * dw
                b -= lr * db

                if epoch % 200 == 0:
                    loss = cls.compute_loss(y, y_hat)
                    print(f"Epoch {epoch}, Loss: {loss:.4f}")
            
            return w, b
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Some error occured: {e}")
            return None

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


    def logistic_regression_toy(self):
        try:
            X = np.array([[0,0],[0,1],[1,0],[1,1]])
            y = np.array([[0],[0],[0],[1]])

            w, b = Algorithms.train_logistic_regression(X, y, lr=0.5,epochs=2000)
            preds = Algorithms.predict(X, w, b)
            print(f"Final Predictions:\n{preds.ravel()}")
            print(f"Weights: {w.ravel()}\nBias: {b}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Some error occured: {e}")

    def logistic_regression_dataset(self):
        data = load_breast_cancer(as_frame=True)
        X, y = data.data, data.target
        print(f"Dataset shape: {X.shape}")
        print(f"Labels: {np.unique(y)}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test  = scaler.transform(X_test)

        weights, bias = Algorithms.train_logistic_regression(X_train, y_train, lr=0.5, epochs=2000)

        z_test = np.dot(X_test, weights) + bias
        y_prob = Algorithms.sigmoid(z_test)
        y_pred = (y_prob > 0.5).astype(int)

        accuracy = np.mean(y_pred == y_test)
        print(f"Final accuracy on the test set: {accuracy}")

        # plotting the 30-D data onto 2-D plane
        X_vis = X_train[:, :2]
        weights_vis, bias_vis = Algorithms.train_logistic_regression(X_vis, y_train, lr=0.1, epochs=2000)

        plt.figure(figsize=(8,6))
        plt.scatter(X_vis[:,0], X_vis[:, 1], c=y_train, cmap='bwr', alpha=0.7)

        x1_min, x1_max = X_vis[:,0].min(), X_vis[:,0].max()
        x2_min, x2_max = X_vis[:,1].min(), X_vis[:,1].max()
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                               np.linspace(x2_min, x2_max, 100))
        
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        probs = Algorithms.sigmoid(np.dot(grid, weights_vis) + bias_vis).reshape(xx1.shape)

        plt.contourf(xx1, xx2, probs, alpha=0.3, cmap='bwr')
        plt.ylabel("Feature 2 (main texture)")
        plt.xlabel("Feature 1 (main radius)")
        plt.show()       


