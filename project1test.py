import numpy as np
import matplotlib.pyplot as plt
import csv
import math # Needed for checking for invalid values

def squarederrorcf(x,y,w,b):
    J = 0
    m = len(x)
    if m == 0:
        return 0
    for i in range(m):
        temp = w*x[i]+b
        J += (temp-y[i])**2
    return (J/(2*m))

def predictions(x,w,b):
    y = []
    for i in range(len(x)):
        y.append(w*x[i]+b)
    return y

def derivative(x,y,w,b):
    grad = 0
    grad1 = 0
    m = len(x)
    if m == 0:
        return 0, 0
    for i in range(m):
        temp = w*x[i]+b
        grad += (temp - y[i])*x[i]
        grad1 += (temp-y[i])
    return (grad/m), (grad1/m) 

def gradient_descent(x,y,w,b,alpha):
    for i in range(0,10000):
        djdw,djdb = derivative(x,y,w,b)
        if (abs(djdw) < 0.0001 and abs(djdb) < 0.0001):
            print("Convergence reached early.")
            break
            
        w = w - alpha*djdw
        b = b - alpha*djdb
        cost = squarederrorcf(x,y,w,b)
        if math.isinf(cost) or math.isnan(cost):
            print("Cost is exploding! Your alpha might be too large.")
            break
        if (i%1000 == 0):
            print(f"Iteration {i}: Cost = {cost:0.2e}, dJ/dw = {djdw:0.2e}, dJ/db = {djdb:0.2e}")
            
    return w,b
alpha = 0.001 
w = 0
b = 0
x = []
y = []

try:
    with open('hello.txt','r') as fobj:
        for line in fobj:
            parts = line.split()
            if len(parts) >= 2:
                x.append(int(parts[0]))
                y.append(int(parts[1]))
except FileNotFoundError:
    print("Error: 'hello.txt' could not be found.")
    exit()

if not x:
    print("Error: No data loaded from 'hello.txt'. The file might be empty.")
    exit()

w,b = gradient_descent(x,y,w,b,alpha)

print(f"\nTraining Complete!\nFinal w: {w:.4f}\nFinal b: {b:.4f}")

# Plot the original data points
plt.scatter(x, y, marker='x', c='r', label="Actual Data")
# Generate predictions with the *final* w and b
y_pred = predictions(x,w,b)
# Plot the line your model learned
plt.plot(x, y_pred, c='b', label="Model Prediction")

plt.title("Anxiety vs Depression")
plt.ylabel('Depression')
plt.xlabel('Anxiety')
plt.legend()
plt.show()