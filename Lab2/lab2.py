import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

#Polinom grad 1 - regresie
m = 50
t = np.random.randint(1,10 , (m , 1))
b = np.random.rand(m , 1) * 390 + 9
#A = np.ones(m , int)
#stack = np.hstack((A , t))

A = np.hstack((np.ones_like(t) , t))
#coeficienti de regresie
X = np.linalg.solve(A.T@A , A.T@b)

b_pred = np.polyval(X[::-1] , t)

SSE = np.linalg.norm(b - b_pred)**2

plt.scatter(t , b , color = "red")
#t_idx = np.argsort(t.reshape(-1 , ))
#t = t[t_idx]
#b_pred = b_pred[t_idx]
t_grafic = np.linspace(np.min(t) , np.max(t) , 100)
b_pred_grafic = np.polyval(X[::-1] , t_grafic)

#plt.plot(t, b_pred)
plt.plot(t_grafic , b_pred_grafic)
plt.show()

for n in range(2, m):
    A = np.hstack((A , t**n))
    x = np.linalg.solve(A.T@A , A.T@b)
    b_pred = np.polyval(X[::-1] , t)
    SSE = np.linalg.norm(b - b_pred)**2
    MAE = np.linalg.norm( b - b_pred , 1)/m
    MAE2 = metrics.mean_absolute_error(b , b_pred)
    MSE = SSE / m
    MSE2 = metrics.mean_squared_error(b , b_pred)
    RMSE = np.sqrt(MSE)
    RMSE2 = metrics.root_mean_squared_error(b , b_pred)
    R2 = 1 - SSE / (np.linalg.norm(b-np.mean(b))**2)
    R2_2 = metrics.R2_score(b , b_pred)
    plt.plot(t , b , color="red")
    t_grafic = np.linspace(np.min(t) , np.max(t) , 100)
    b_pred_grafic = np.polyval(X[::-1] , t_grafic)

    #plt.plot(t, b_pred)
    plt.plot(t_grafic , b_pred_grafic)
    plt.show()