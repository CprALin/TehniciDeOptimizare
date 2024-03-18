import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

df = pd.read_csv('Orange_Telecom.csv')

def solve(predictors, target):
    A = np.array(df[predictors])
    A = np.sort(A)
    A = A.reshape(-1, 1)
    A = np.c_[np.ones(A.shape[0]), A]

    maxes = []

    for i in range(A.shape[1]):
        maxes.append(np.max(A[:, i]))
        A[:, i] = A[:, i] / maxes[-1]

    b = np.array(df[target]).reshape(-1, 1)

    for i in range(1, 9):
        start_time = time.time()
        if i > 1:
            A = np.c_[A, np.power(A, i)]

        lr = LinearRegression()
        lr.fit(A, b)
        pred = lr.predict(A)
        
        mse = mean_squared_error(b, pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(b, pred)
        sse = np.sum(np.square(pred - b))
        r2 = r2_score(b, pred)

        print(f"Gradul {i}:")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"SSE: {sse}")
        print(f"R2: {r2}")
        print(f"Execution time: {time.time() - start_time}")
        
        plt.scatter(A[:, 1], b, alpha=0.2)
        plt.plot(A[:, 1], np.dot(A, np.insert(lr.coef_[0, 1:], 0, lr.intercept_)), color='red')
        plt.show()
        
#solve('total_day_calls', 'total_intl_minutes')
solve('total_day_calls', 'total_intl_calls')