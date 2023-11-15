# Вариант 15
import math

# def funtion(x):
#     return (x-3)**2 +4
def function(x):
    return -2/(math.cosh(4*x+3)+3)

number_loop = 0


delta = 0.0001
epsilon = 0.00001
def minimum_dikhotomi(a, b, epsilon, number_loop):
    if  abs(b-a) <= 2*delta:
        return number_loop, (a+b)/2, abs(b-a)/2
    else:
        x1 = (a + b - epsilon)/2
        x2 = (a + b + epsilon)/2
        y1 = function(x1)
        y2 = function(x2)
        if y1 >= y2:
            return minimum_dikhotomi(x1, b, epsilon, number_loop + 1)
        else:
            return minimum_dikhotomi(a, x2, epsilon, number_loop + 1)
a = -3
b = 2
#
print(minimum_dikhotomi(a, b, epsilon, 0))


epsilon = 0.00001

phi = (1+pow(5,1/2))/2
def minimun_golden_ratio(a, b, epsilon, numberloop):
    if abs(b-a)>epsilon:
        x1 = b - (b-a)/phi
        x2 = a + (b-a)/phi
        if function(x1) >= function(x2):
            return minimun_golden_ratio(x1, b, epsilon, numberloop+1)
        else:
            return minimun_golden_ratio(a, x2, epsilon, numberloop+1)
    else:
        return numberloop, (a+b)/2
#
#
epsilon = 0.0001

def minimun_golden_ratio_loop(a,b):
    loop = 2
    x1 = b - (b - a) / phi
    x2 = a + (b - a) / phi
    y1 = function(x1)
    y2 = function(x2)
    while abs(b - a) > epsilon:
        loop += 1
        if y1 >= y2:
            a = x1
            x1 = x2
            y1 = y2
            x2 = a + b - x1
            y2 = function(x2)
        else:
            b = x2
            x2 = x1
            y2 = y1
            x1 = a + b - x2
            y1 = function(x1)

    return loop, (a + b) / 2, (b-a)/2
a = -3
b = 2
res = minimun_golden_ratio_loop(a, b)
print(res)

n = 200
fibonacci = []
fibonacci.append(0)
fibonacci.append(1)
for i in range(2, n):
    fibonacci.append(fibonacci[i-1]+fibonacci[i-2])

def minimum_fibonacci(a, b, n):

    x1 = a + fibonacci[n-2]/fibonacci[n]*(b-a)
    x2 = a + fibonacci[n-1]/fibonacci[n]*(b-a)

    y1 = function(x1)
    y2 = function(x2)
    for i in range(1, n-1):
        if y1 > y2:
            a = x1
            x1 = x2
            y1 = y2
            x2 = b - (x1-a)
            y2 = function(x2)
        else:
            b = x2
            x2 = x1
            y2 = y1
            x1 = a + (b-x2)
            y1 = function(x1)
        print(b-a)
    return (a+b)/2, (b-a)/2
# print(fibonacci)
a = -3
b = 2
n = 25
res = minimum_fibonacci(a,b,n)
print(res)