# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:24:53 2019

@author: Tristan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix

Rsun=6.9599e10
Lsun=3.826e33

data1 = pd.read_csv('starmodl1.dat',delim_whitespace=True,index_col=False,header=15)
data2 = pd.read_csv('starmodl2.dat',delim_whitespace=True,index_col=False,header=15)
data3 = pd.read_csv('starmodl3.dat',delim_whitespace=True,index_col=False,header=15)


r1 = np.array(data1['r'])/Rsun
Qm1 = np.array(data1['Qm'])
L1 = np.array(data1['L_r'])/Lsun
T1 = np.array(data1['T'])
P1 = np.array(data1['P'])
rho1 = np.array(data1['rho'])
kappa1 = np.array(data1['kap'])
epsilon1 = np.array(data1['eps']).T
zone1 = np.array(data1['zone'])
dlPdlT1 = np.array(data1['dlPdlT'])

r2 = np.array(data2['r'])/Rsun
Qm2 = np.array(data2['Qm'])
L2 = np.array(data2['L_r'])/Lsun
T2 = np.array(data2['T'])
P2 = np.array(data2['P'])
rho2 = np.array(data2['rho'])
kappa2 = np.array(data2['kap'])
epsilon2 = np.array(data2['eps']).T
zone2 = np.array(data2['zone'])
dlPdlT2 = np.array(data2['dlPdlT'])

r3 = np.array(data3['r'])/Rsun
Qm3 = np.array(data3['Qm'])
L3 = np.array(data3['L_r'])/Lsun
T3 = np.array(data3['T'])
P3 = np.array(data3['P'])
rho3 = np.array(data3['rho'])
kappa3 = np.array(data3['kap'])
epsilon3 = np.array(data3['eps']).T
zone3 = np.array(data3['zone'])
dlPdlT3 = np.array(data3['dlPdlT'])

'''y1 = np.where(zone1 == 'c',0,1)
y2 = np.where(zone2 == 'c',0,1)
y = np.concatenate((y1,y2))'''

'''x1 = np.array([])
x1 = np.transpose(x1)
x2 = np.array([kappa2])
x2 = np.transpose(x2)
x = np.concatenate((x1,x2))'''

y = np.where(zone1 == 'c',0,1)
#x = np.array([dlPdlT1]) #yes to dlPdlT
#x = np.array([r1,dlPdlT1])# no to radius
x = np.array([r1,kappa1])#
#x = np.array([T1,P1])# no to both
#x = np.array([kappa1])# yes to kappa
#x = np.array([kappa1,dlPdlT1]) #extra YES
x = np.transpose(x)

xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.3,random_state=0)

#sts = StandardScaler()
#sts.fit_transform(xtrain)

classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(xtrain,ytrain)
ypred = classifier.predict(xtest)

cm = confusion_matrix(ytest,ypred)

X_set, y_set = xtest, ytest
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

#plt.scatter(kappa1,dlPdlT1,c='C0')
#plt.scatter(xtest,ypred,c='C1')
plt.ylabel(r'Temperature Gradient $\frac{d\left(lnP\right)}{d\left(lnT\right)}$')
plt.xlabel(r'Opacity $\kappa$')
plt.grid()
plt.show()

'''
plt.scatter(r1,Qm1,c='C0')
plt.scatter(r2,Qm2,c='C1')
plt.scatter(r3,Qm3,c='C2')
plt.xlabel(r'Radius $\frac{R}{R_{\odot}}$')
plt.ylabel(r'Mass $\frac{M}{M_{\odot}}$')
plt.grid()
plt.show()

plt.scatter(r1,T1,c='C0')
plt.scatter(r2,T2,c='C1')
plt.scatter(r3,T3,c='C2')
plt.xlabel(r'Radius $\frac{R}{R_{\odot}}$')
plt.ylabel(r'Temperature (K)')
plt.grid()
plt.show()

plt.scatter(r1,L1,c='C0')
plt.scatter(r2,L2,c='C1')
plt.scatter(r3,L3,c='C2')
plt.xlabel(r'Radius $\frac{R}{R_{\odot}}$')
plt.ylabel(r'Luminosity $\frac{L}{L_{\odot}}$')
plt.grid()
plt.show()

plt.scatter(r1,rho1,c='C0')
plt.scatter(r2,rho2,c='C1')
plt.scatter(r3,rho3,c='C2')
plt.xlabel(r'Radius $\frac{R}{R_{\odot}}$')
plt.ylabel(r'Density $\rho$')
plt.grid()
plt.show()

plt.scatter(r1,P1,c='C0')
plt.scatter(r2,P2,c='C1')
plt.scatter(r3,P3,c='C2')
plt.xlabel(r'Radius $\frac{R}{R_{\odot}}$')
plt.ylabel(r'Pressure')
plt.grid()
plt.show()

plt.scatter(r1,kappa1,c='C0')
plt.scatter(r2,kappa2,c='C1')
plt.scatter(r3,kappa3,c='C2')
plt.xlabel(r'Radius $\frac{R}{R_{\odot}}$')
plt.ylabel(r'Opacity $\kappa_f$')
plt.grid()
plt.show()

plt.scatter(r1,epsilon1,c='C0')
plt.scatter(r2,epsilon2,c='C1')
plt.scatter(r3,epsilon3,c='C2')
plt.xlabel(r'Radius $\frac{R}{R_{\odot}}$')
plt.ylabel(r'Total Energy $\epsilon$')
plt.grid()
plt.show()

plt.scatter(r1,dlPdlT1,c='C0')
plt.scatter(r2,dlPdlT2,c='C1')
plt.scatter(r3,dlPdlT3,c='C2')
plt.xlabel(r'Radius $\frac{R}{R_{\odot}}$')
plt.ylabel(r'Temperature Gradient $\frac{d\left(lnP\right)}{d\left(lnT\right)}$')
plt.grid()
plt.show()

plt.scatter(T1,P1,c='C0')
plt.scatter(T2,P2,c='C1')
plt.scatter(T3,P3,c='C2')
plt.xlabel(r'Temperature (K)')
plt.ylabel(r'Pressure')
plt.grid()
plt.show()
'''