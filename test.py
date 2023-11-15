import numpy as np
import sympy as sp
def f(X):
    x0 = 3
    y0 = 3
    alpha = 0
    #np.pi/3
    x_new = (X[0]-x0)*np.cos(alpha) + (X[1]-y0)*np.sin(alpha)
    y_new = (X[1]-y0)*np.cos(alpha) - (X[0]-x0)*np.sin(alpha)

    return np.exp(x_new**2)+np.exp(y_new**2)+np.exp(X[2]**2)


def minimum_golden_ratio_loop(X, a, b):
    epsilon = 0.0000001
    phi = (1+pow(5,1/2))/2
    loop = 2
    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi
    y1 = f(X + x1)
    y2 = f(X + x2)
    while np.linalg.norm(b - a) > 2*epsilon and np.abs(y1-y2) > epsilon:
        loop += 1
        if y1 >= y2:
            a = x1
            x1 = x2
            y1 = y2
            x2 = a + b - x1
            y2 = f(X + x2)
        else:
            b = x2
            x2 = x1
            y2 = y1
            x1 = a + b - x2
            y1 = f(X + x1)
    return (a + b) / 2, loop

# Метод координатного спуска
def min_by_direct(X, p):
    d_alpha = 0.001
    delta = d_alpha*2

    if f(X + delta*p) < f(X + d_alpha*p):
        while f(X + delta*p) < f(X + d_alpha*p):
            d_alpha = delta
            delta = delta*2
        return minimum_golden_ratio_loop(X, np.array([0, 0, 0]), delta * p)

    if f(X + delta * p) > f(X + d_alpha * p):
        while f(X - delta * p) < f(X - d_alpha * p):
            d_alpha = delta
            delta = delta * 2
        return minimum_golden_ratio_loop(X, -delta * p, np.array([0, 0, 0]))

#
#
# X = [0, 0, 4]
# X_start = X
# min_by_x = min_by_direct(X, np.array([1, 0, 0]))
# X = X + min_by_x
# min_by_y = min_by_direct(X, np.array([0, 1, 0]))
# X = X + min_by_y
# min_by_z = min_by_direct(X, np.array([0, 0, 1]))
# X = X + min_by_z
#
# X_end = X
#
# while True:
#     if np.linalg.norm(X_end - X_start) > 0.001:
#         X_start = X_end
#
#         min_by_x = min_by_direct(X_end, np.array([1, 0, 0]))
#         X_end = X_end + min_by_x
#
#         min_by_y = min_by_direct(X_end, np.array([0, 1, 0]))
#         X_end = X_end + min_by_y
#
#         min_by_z = min_by_direct(X_end, np.array([0, 0, 1]))
#         X_end = X_end + min_by_z
#     else:
#         break
# print(X_end)

# Метод наискорейского спуска
# #
# def f(X):
#     x0 = 1
#     y0 = 3
#     alpha = np.pi/3
#     x_new = (X[0]-x0)*np.cos(alpha) + (X[1]-y0)*np.sin(alpha)
#     y_new = (X[1]-y0)*np.cos(alpha) - (X[0]-x0)*np.sin(alpha)
#
#     return np.exp(x_new**2)+np.exp(y_new**2)+np.exp(X[2]**2)
#
# def gradient_f(X):
#     x0 = 1
#     y0 = 3
#     alpha = np.pi/3
#     x_new = (X[0] - x0) * np.cos(alpha) + (X[1] - y0) * np.sin(alpha)
#     y_new = (X[1] - y0) * np.cos(alpha) - (X[0] - x0) * np.sin(alpha)
#
#     grad = np.array([2*x_new*np.exp(x_new**2)*np.cos(alpha) - 2*y_new*np.exp(y_new**2)*np.sin(alpha),
#                      2*y_new*np.exp(y_new**2)*np.cos(alpha) + 2*x_new*np.exp(x_new**2)*np.sin(alpha),
#                      2*X[2]*np.exp(X[2]**2)])
#
#     return grad/np.linalg.norm(grad)
#
#
# def min_by_gradient(X):
#     d_alpha = 0.0000001
#     delta = d_alpha * 2
#     p = gradient_f(X)
#
#     if f(X + delta * p) < f(X + d_alpha * p):
#         while f(X + delta * p) < f(X + d_alpha * p):
#             d_alpha = delta
#             delta = delta * 2
#         min_point, loop = minimum_golden_ratio_loop(X, np.array([0, 0, 0]), delta * p)
#         print(X + min_point, loop)
#         return min_point, loop
#
#     if f(X + delta * p) > f(X + d_alpha * p):
#         while f(X - delta * p) < f(X - d_alpha * p):
#             d_alpha = delta
#             delta = delta * 2
#         min_point, loop = minimum_golden_ratio_loop(X, -delta * p, np.array([0, 0, 0]))
#         print(X + min_point, loop)
#         return min_point, loop
#
#
# X = [0, 0, 4]
# X_start = X
#
# min_by_grad, loop0 = min_by_gradient(X)
# X = X + min_by_grad
# print(loop0)
# number_loop = loop0
# X_end = X
# while True:
#     if np.linalg.norm(X_end - X_start) > 0.000001:
#         X_start = X_end
#         min_by_grad, loop = min_by_gradient(X_end)
#         number_loop += loop
#         print(loop)
#         X_end = X_end + min_by_grad
#     else:
#         break
#
# print(X_end)
# print(number_loop)

# Метод Ньютона
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

i = 0
while True:
    g = gradient(X).reshape(-1, 1)
    H = hesse(X)
    H_inv = np.linalg.inv(H)
    s = np.dot(H_inv, g).flatten()
    i=i+1
    X = X-s
    print(X)
    if np.linalg.norm(s) < 0.000001:
        break
print(X)
print(i)