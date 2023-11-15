import numpy as np
import sympy as sp

def f(X):
    alpha = 0
    x0 = 1
    y0 = 3
    x_new = (X[0]-x0)*np.cos(alpha) + (X[1]-y0)*np.sin(alpha)
    y_new = (X[1]-y0)*np.cos(alpha) - (X[0]-x0)*np.sin(alpha)
    return np.exp(x_new**2)+np.exp(y_new**2)+np.exp(X[2]**2)

def function_f():
    alpha = 0
    x0 = 1
    y0 = 3
    x, y, z = sp.symbols('x y z')

    f = sp.exp(((x - x0) * sp.cos(alpha) + (y - y0) * np.sin(alpha)) ** 2) + \
        sp.exp(((y - y0) * sp.cos(alpha) - (x - x0) * np.sin(alpha)) ** 2) + \
        sp.exp(z ** 2)
    return f

def gradient(X):
    x, y, z = sp.symbols('x y z')
    f = function_f()

    df_dx = sp.diff(f, x)
    df_dy = sp.diff(f, y)
    df_dz = sp.diff(f, z)

    df_dx_value = np.float64(df_dx.subs({x: X[0], y: X[1], z: X[2]}))
    df_dy_value = np.float64(df_dy.subs({x: X[0], y: X[1], z: X[2]}))
    df_dz_value = np.float64(df_dz.subs({x: X[0], y: X[1], z: X[2]}))

    g = [df_dx_value, df_dy_value, df_dz_value]

    return np.array(g)

def hesse(X):
    x, y, z = sp.symbols('x y z')
    f = function_f()

    df_dx_dx =  sp.diff(sp.diff(f, x), x)
    df_dx_dx_value = np.float64(df_dx_dx.subs({x: X[0], y: X[1], z: X[2]}))

    df_dx_dy = sp.diff(sp.diff(f, x), y)
    df_dx_dy_value = np.float64(df_dx_dy.subs({x: X[0], y: X[1], z: X[2]}))

    df_dx_dz = sp.diff(sp.diff(f, x), z)
    df_dx_dz_value = np.float64(df_dx_dz.subs({x: X[0], y: X[1], z: X[2]}))

    df_dy_dy = sp.diff(sp.diff(f, y), y)
    df_dy_dy_value = np.float64(df_dy_dy.subs({x: X[0], y: X[1], z: X[2]}))

    df_dy_dz = sp.diff(sp.diff(f, y), z)
    df_dy_dz_value = np.float64(df_dy_dz.subs({x: X[0], y: X[1], z: X[2]}))

    df_dz_dz = sp.diff(sp.diff(f, z), z)
    df_dz_dz_value = np.float64(df_dz_dz.subs({x: X[0], y: X[1], z: X[2]}))

    H = [[df_dx_dx_value, df_dx_dy_value, df_dx_dz_value],
         [df_dx_dy_value, df_dy_dy_value, df_dy_dz_value],
         [df_dx_dz_value, df_dy_dz_value, df_dz_dz_value]]
    return np.array(H)

X = [0, 0, 4]
gradient_X = gradient(X)
g = gradient_X.reshape(-1, 1)
H = hesse(X)
H = np.linalg.inv(H)

i=0
while True:
    s = -np.dot(H, g).flatten()

    X_new = X + s
    print(X_new)
    i = i+1
    print(i)
    s_k = X_new - X
    y_k = gradient(X_new)-gradient(X)

    k = 1/np.dot(y_k, s_k)
    I = np.identity(3)
    H = np.dot(np.dot((I - k*np.dot(s_k.reshape(3,1),y_k.reshape(1,3))),H), (I - k*np.dot(y_k.reshape(3,1),s_k.reshape(1,3)))) \
        + k*np.dot(s_k.reshape(3,1),s_k.reshape(1,3))

    X = X_new
    g = gradient(X)

    if np.linalg.norm(g)< 0.01:
        break
print(X)
