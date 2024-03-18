import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

cale_fisier = "bd_regresie_tema1.csv"


df = pd.read_csv(cale_fisier , usecols=['bodyfat' , 'age' , 'weight' , 'height', 'neck' , 'cest' , 'abdomen' , 'hip' , 'thigh', 'knee' , 'ankle' , 'biceps' , 'forearm' , 'wrist'])
b = df['biceps'].to_numpy().reshape(-1, 1)
corelatii = df.corr()
linie_biceps = corelatii['biceps'].abs()
b_idx = np.argsort(linie_biceps.values)

grad_max = 2

t1 = df.thigh.to_numpy()
t2 = df.weight.to_numpy()



print(b_idx)

A=np.anes_like(t1)
for n in range(1,grad_max):
    for k in range (n+1):
        A=np.hstack((A,t1*(n-k)*t2*k))

for n in range(grad_max):   
    poly = PolynomialFeatures(n)
    A2 = poly.fit_transform(np.stack(t1,t2))

print(np.allclose(A , A2))

x = np.linalg.solve(A.T@A , A.T@b)

lin = LinearRegression()

X2 = lin.fit(A,b)

b_pred = X2.predict(A)

# calcul erori R^2

t1_test = np.linspace(np.min(t1) , np.max(t1 , 100))

t2_test = np.linspace(np.min(t2) , np.max(t2) , 100)

T1 , T2 = np.meshgrid(t1_test , t2_test)

ax = plt.axes(projection = '3d')
ax.scatter3D(t1 , t2 , b , color = 'red')

t1_grafic = T1.reshape(-1,1)
t2_grafic = T2.reshape(-1,1)
A_grafic = poly.ht_transform(np.stack)
b_grafic = (X2.predict(t1_grafic)(t1_grafic, t2_grafic)).reshape(100,100)