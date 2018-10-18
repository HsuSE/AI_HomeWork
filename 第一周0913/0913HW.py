import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

for i in range(3):

    x1 = np.linspace(0, 2*np.pi, 9)
    y1 = np.sin(x1)+np.random.randn(len(x1))/5.0
    x1 = x1.reshape(-1, 1)

    poly_features_3 = PolynomialFeatures(degree=(i+1)*3, include_bias=False)
    X_poly_3 = poly_features_3.fit_transform(x1)
    lin_reg_3 = LinearRegression()
    lin_reg_3.fit(X_poly_3, y1)
    print(lin_reg_3.intercept_, lin_reg_3.coef_)
    X_plot = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
    X_plot_poly = poly_features_3.fit_transform(X_plot)

    y_plot = np.dot(X_plot_poly, lin_reg_3.coef_.T) + lin_reg_3.intercept_
    plt.subplot(3, 3, 1+(i*3))
    plt.plot(x1, y1, 'b.')
    plt.plot(X_plot, y_plot, 'r-')


    x1 = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x1)+np.random.randn(len(x1))/5.0
    x1 = x1.reshape(-1, 1)

    poly_features_d = PolynomialFeatures(degree=(i+1)*3, include_bias=False)
    X_poly_d = poly_features_d.fit_transform(x1)
    lin_reg_d = LinearRegression()
    lin_reg_d.fit(X_poly_d, y1)
    print(lin_reg_d.intercept_, lin_reg_d.coef_)
    X_plot = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
    X_plot_poly = poly_features_d.fit_transform(X_plot)
    y_plot = np.dot(X_plot_poly, lin_reg_d.coef_.T) + lin_reg_d.intercept_
    plt.subplot(3, 3, 2+(i*3))
    plt.plot(x1, y1, 'b.')
    plt.plot(X_plot, y_plot, 'r-')


    x1 = np.linspace(0, 2*np.pi, 1000)
    y1 = np.sin(x1)+np.random.randn(len(x1))/5.0
    x1 = x1.reshape(-1, 1)

    poly_features_d = PolynomialFeatures(degree=(i+1)*3, include_bias=False)
    X_poly_d = poly_features_d.fit_transform(x1)
    lin_reg_d = LinearRegression()
    lin_reg_d.fit(X_poly_d, y1)
    print(lin_reg_d.intercept_, lin_reg_d.coef_)
    X_plot = np.linspace(0, 2*np.pi, 1000).reshape(-1, 1)
    X_plot_poly = poly_features_d.fit_transform(X_plot)
    y_plot = np.dot(X_plot_poly, lin_reg_d.coef_.T) + lin_reg_d.intercept_
    plt.subplot(3, 3, 3+(i*3))
    plt.plot(x1, y1, 'b.')
    plt.plot(X_plot, y_plot, 'r-')
plt.tight_layout()
plt.savefig("Photo.png")
plt.show()
