import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

class LinearAlgebra:
    def __init__(self):
        pass

    def vector_basics_and_plots(self):
        v1 = np.array([2,3])
        v2 = np.array([4,0])

        dot_product = np.dot(v1,v2)
        magnitude_v1 = np.linalg.norm(v1)

        df = pd.DataFrame({
            "vector 1": v1,
            "vector 2": v2
        })

        print(df)
        print(f"Dot product: {dot_product}")
        print(f"Magnitude of v1: {magnitude_v1:.2f}")

        print("calculating v1 + v2")
        v3 = np.add(v1, v2)
        print(f"{v3[0]}i {v3[1]}j")

        df.insert(len(df.columns),"vector 3", v3)

        print(df)


        plt.figure(figsize=(6,6))
        plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='v1')
        plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='v2')
        plt.quiver(0, 0, v3[0], v3[1], angles='xy', scale_units='xy', scale=1, color='g', label='v3 = v1 + v2')
        plt.quiver(v1[0], v1[1], v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='cyan', label='v4')
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        plt.grid()
        plt.legend()
        plt.show()
        return
    
    def matrix_and_dot_product(self):
        v1 = np.array([4, 3])
        v2 = np.array([4, 1])

        print(f"Vector 1: {v1[0]}i {v1[1]}j\nVector 2: {v2[0]}i {v2[1]}j")
        dot_product = np.dot(v1, v2)
        print(f"Dot Product: {dot_product}")

        mag_v1 = np.linalg.norm(v1)
        print(f"Magnitude v1: {mag_v1}")

        A = np.array([[2, 0],
                      [0, 3]])
        print(f"Matrix A:\n{A}")
        result = A @ v1
        print(f"Matrix multiplication result = {result}")

    def dot_product_and_projection(self):
        v1 = np.array([3, 1])
        v2 = np.array([2, 4])

        dot_product = np.dot(v1, v2)
        projection = (dot_product / np.dot(v2, v2)) * v2
        print(f"Vector 1: {v1[0]}i {v1[1]}j\nVector 2: {v2[0]}i {v2[1]}j")
        print(f"Dot product: {dot_product}")
        print(f"Projection of v1 on v2: {projection}")

        plt.figure(figsize=(6,6))
        plt.title("Projection visualisation")
        plt.quiver(0, 0, v1[0], v1[1], scale_units='xy', angles='xy', scale=1, color='r',label='v1')
        plt.quiver(0, 0, v2[0], v2[1], scale_units='xy', angles='xy', scale=1, color='b',label='v2')
        plt.quiver(0, 0, projection[0], projection[1], scale_units='xy', angles='xy', scale=1, color='g',label='projection')
        plt.plot([projection[0], v1[0]], [projection[1], v1[1]], 'k--', alpha=0.6)
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0, color='black', linewidth=0.5)
        plt.grid()
        plt.legend()
        plt.show()

    def cosine_similarity(self):
        v1, v2 = np.array([3, 1]), np.array([2, 4])

        dot = np.dot(v1, v2)
        mag_v1, mag_v2 = np.linalg.norm(v1), np.linalg.norm(v2)

        cos_angle = (dot/(mag_v1 * mag_v2))
        print(f"Dot product of v1: {v1} and v2: {v2}: {dot}")
        print(f"Angle Between v1 and v2: {cos_angle}")

        plt.figure(figsize=(6, 6))
        plt.title("Cosine Similarity")
        plt.quiver(0, 0, v1[0], v1[1], scale_units='xy', color='r', scale=1, angles='xy', label='v1')
        plt.quiver(0, 0, v2[0], v2[1], scale_units='xy', color='r', scale=1, angles='xy', label='v2')
        circle = plt.Circle((0,0), 1, color='gray', alpha=0.6, fill=False, linestyle='--')
        plt.gca().add_artist(circle)

        plt.xlim(-1, 5)
        plt.ylim(-1, 5)
        plt.axhline(0, linewidth=0.5, color='black')
        plt.axvline(0, linewidth=0.5, color='black')
        plt.grid()
        plt.legend()
        plt.show()

    def linear_regression(self):
        np.random.seed(42)                      # Data generation
        X = 2 * np.random.rand(100, 1)          # 100 samples, 1 feature
        y = 4 + 3 * X + np.random.randn(100, 1) # y = 4 + 3x + noise

        x_b = np.c_[np.ones((100, 1)), X]                       # Add Bias terms
        theta_best = np.linalg.inv(x_b.T @ x_b) @ x_b.T @ y     # Apply the normal equation
        print(f"Best parameters (w, b):", theta_best.ravel())   

        # Calculating predictions.
        X_new = np.array([[0], [2]])            
        X_new_b = np.c_[np.ones((2, 1)), X_new]
        y_predict = X_new_b @ theta_best

        # test the prediction with r2_score
        mse = mean_squared_error(y, y_predict)
        r2 = r2_score(y, y_predict)
        print(f"Mean squared error: {mse}\nR2_score: {r2}")

        # Plot the graph

        plt.scatter(X, y, alpha=0.6, label='data')
        plt.plot(X_new, y_predict, 'r-', linewidth=2, label='Best fit line')
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid()
        plt.show()

    def scikit_learn(self):
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame.head(500)

        X = df[['MedInc']]
        y = df['MedHouseVal']

        lin_reg = LinearRegression()
        lin_reg.fit(X, y)

        print(f"Intercept (b): {lin_reg.intercept_}")
        print(f"Slope: {lin_reg.coef_}")

        y_pred = lin_reg.predict(X)

        plt.figure(figsize=(6,6))
        plt.scatter(X, y, alpha=0.6, label='Actual Data')
        plt.plot(X, y_pred, color='red', label='regression line', linewidth=2)
        plt.xlabel("Median Income")
        plt.ylabel("House price")
        plt.legend()
        plt.show()

    
    # implementaion of linear regression single dimension with housing dataset.
    def excercise_1(self):
        housing = fetch_california_housing(as_frame=True)
        df = housing.frame.head(500)

        X = df[['MedInc']].values
        y = df['MedHouseVal'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        x_b = np.c_[np.ones((X_train.shape[0],1)), X_train]

        theta_best = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_train)

        b = theta_best[0]
        w = theta_best[1]

        print(f"Intercept: {b}")
        print(f"Coefficient: {w}")

        x_test_b = np.c_[np.ones((X_test.shape[0], 1)), X_test]
        y_pred = x_test_b.dot(theta_best)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE: {mse}\nR2 score: {r2}")

        plt.title("Linear Regression with housing dataset")
        plt.figure(figsize=(6,6))
        plt.scatter(X_test, y_test, alpha=0.3, label='actual data')
        plt.plot(X_test, y_pred, color='r', linewidth=2, label='Regression line')
        plt.xlabel("median income")
        plt.ylabel("House Value")
        plt.legend()
        plt.grid()
        plt.show()

    
    # implementation of multivariate linear regression with housing dataset using NumPy.
    def excercise_2(self):
        housing = fetch_california_housing(as_frame=True)
        X, y = housing.data, housing.target
        feature_names = "This is a string"
        feature_names = housing.feature_names

        print(f"Features: {feature_names}")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")

        X = np.c_[np.ones((X.shape[0])), X]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        theta_best = np.linalg.inv(X_train.T.dot(X_train)).dot(X_train.T).dot(y_train)
        print(f"Intercept: {theta_best[0]}")
        print(f"Coefficient: {theta_best[1]}")

        y_pred = X_test.dot(theta_best)
        print(f"first 5 predictions: {y_pred[:5]}")
        print(f"First 5 actual : {y_test[:5]}")

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"Mean squared error: {mse}")
        print(f"R2 score: {r2}")
    