import numpy as np
from numpy import linalg as la
import pandas as pd
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('Orange_Telecom.csv')

def solve(predictors, target):
    A = np.array(df[predictors])
    A = np.sort(A, axis=0)
    
    b = np.array(df[target])
    b = b.reshape(-1, 1)

    for i in range(1, 4):
        start_time = time.time()
        
        poly_feat = PolynomialFeatures(i)
        A_new = poly_feat.fit_transform(A)

        inv = la.pinv(np.dot(A_new.T, A_new))
        x = np.dot(np.dot(inv, A_new.T), b)
        
        pred = np.dot(A_new, x)

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
        print()
        
        t1_grafic = np.linspace(np.min(A[:, 0]), np.max(A[:, 0]), A.shape[0])
        t2_grafic = np.linspace(np.min(A[:, 1]), np.max(A[:, 1]), A.shape[0])
        
        T1, T2 = np.meshgrid(t1_grafic, t2_grafic)
        
        t1_test = T1.reshape(-1, 1)
        t2_test = T2.reshape(-1, 1)
        
        A_test = poly_feat.fit_transform(np.hstack((t1_test.reshape(-1, 1), t2_test.reshape(-1, 1))))
        b_pred_test = np.dot(A_test, x)
        B_pred_test = b_pred_test.reshape(A.shape[0], A.shape[0])
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title(f'Gradul {i}')
        ax.scatter(A[:, 0], A[:, 1], b, color='red')
        ax.plot_surface(T1, T2, B_pred_test, alpha=0.5)
    
        plt.show()
        
#solve(['total_day_calls', 'total_night_calls'], 'total_intl_minutes')
solve(['total_eve_charge', 'total_day_calls'], 'total_intl_calls')