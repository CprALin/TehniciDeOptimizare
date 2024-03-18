import numpy as np
from numpy import linalg as la
import pandas as pd
import time
import matplotlib.pyplot as plt

df = pd.read_csv('Orange_Telecom.csv')

df.drop(['phone_number', 'area_code', 'state', 'intl_plan', 'voice_mail_plan', 'churned'], axis='columns', inplace=True)

# Matrice de covarianta
corr = df.corr() # Target: total_intl_minutes
corr['total_intl_minutes'].sort_values() 

corr = df.corr() # Target: total_intl_calls
corr['total_intl_calls'].sort_values()

# MSE = Mean Squared Error
# RMSE = Root Mean Squared Error
# MAE = Mean Absolute Error
# SSE = Sum of Squared Erros
# R2
# Error = diferenta dintre valoarea prezisa de model si valoarea reala

def solve(predictors, target):
    A = np.array(df[predictors])
    A = np.sort(A)
    A = A.reshape(-1, 1)
    A = np.c_[np.ones(A.shape[0]), A]

    maxes = [] # Scalare

    for i in range(A.shape[1]):
        maxes.append(np.max(A[:, i]))
        A[:, i] = A[:, i] / maxes[-1]

    b = np.array(df[target])
    b = b.reshape(-1, 1)

    for i in range(1, 9):
        start_time = time.time()
        if i > 1:
            A = np.c_[A, np.power(A, i)]

        inv = la.pinv(np.dot(A.T, A)) # Ecuatia normala
        x = np.dot(np.dot(inv, A.T), b) # Ecuatia normala

        pred = np.dot(A, x)

        mse = np.mean(np.square(pred - b))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(pred - b))
        sse = np.sum(np.square(pred - b))
        r2 = 1 - (sse / np.sum(np.square(b - np.mean(b))))

        print(f"Gradul {i}:")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"SSE: {sse}")
        print(f"R2: {r2}")
        print(f"Execution time: {time.time() - start_time}")

        plt.scatter(A[:, 1], b, alpha=0.2)
        plt.plot(A[:, 1], np.dot(A, x), color='red')
        plt.show()
        
#solve('total_day_calls', 'total_intl_minutes')
solve('total_day_calls', 'total_intl_calls')