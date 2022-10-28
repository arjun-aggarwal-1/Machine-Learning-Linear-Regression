import numpy as np
import matplotlib.pyplot as plt


def main():

    [X, Y] = genData()
    [mean, stdDev, W, b] = learn(X, Y)
    print()
    print ('Mean of data =', mean)
    print ('stdDev of data =', stdDev)
    print ('Learnt weights: W =', W, 'b =', b)

    newX = np.arange(100)
    newX = newX * 2*np.pi / 100
   
    predict(newX, mean, stdDev, W, b)
    plt.show()

def predict(newX, mean, stdDev, W, b) :
    X = np.zeros((len(newX), 8))#
    X[:, 0] = newX
    X[:, 1] = np.power(newX, 2)
    X[:, 2] = np.power(newX, 3)
    X[:, 3] = np.power(newX, 4)
    X[:, 4] = np.power(newX, 5)
    X[:, 5] = np.power(newX, 6)
    X[:, 6] = np.power(newX, 7)
    X[:, 7] = np.power(newX, 8)#

    X = np.divide((X - mean), stdDev)
    Y = np.matmul(X, W) + b

    plt.figure()    
    plt.xlabel('Time - as a fraction of Pi')
    plt.ylabel('Predicted Temperature (Centigrade)')
    plt.title('Day Time vs Predicted Temperature')
    plt.plot(newX, Y, 'bo', markersize=2)
   
   
   
   

    ySin = np.sin(newX) * 12 + 26
    # print(Y.shape, ySin.shape)

   
    plt.plot(newX, ySin, 'r')


def genData() :
    x = np.zeros((1000, 8), dtype=float)#
    y = np.zeros((1000), dtype=float)
    mu = 0
    sigma = 2.0
    v = np.random.randn(1000) * sigma + mu
    for i in range (1000):
        x[i, 0] = 2*np.pi * i / 1000
        x[i, 1] = (x[i, 0])**2
        x[i, 2] = (x[i, 0])**3
        x[i, 3] = (x[i, 0])**4
        x[i, 4] = (x[i, 0])**5
        x[i, 5] = (x[i, 0])**6
        x[i, 6] = (x[i, 0])**7
        x[i, 7] = (x[i, 0])**8#
        y[i] = (np.sin(x[i, 0])*12) + 26 + v[i]

    plt.plot(x[:, 0], y, 'ro', markersize=0.5)
    plt.xlabel('Time - as a fraction of Pi')
    plt.ylabel('Temperature (Centigrade)')
    plt.title('Day Time vs Temperature')

    return [x, y]

def learn(X, Y) :
    mean = np.mean(X, axis=0)
    stdDev = np.std(X, axis=0)
    X = np.divide((X - mean), stdDev)
   
    W = np.zeros((X.shape[1]))
    dW = np.zeros((X.shape[1]))
    b = 0
    alpha = 0.01

   
    for i in range(500000) :
        yCap = np.matmul(X, W) + b
        diff = yCap - Y
        dW[0] = np.mean(np.multiply(X[:, 0], diff))
        dW[1] = np.mean(np.multiply(X[:, 1], diff))
        dW[2] = np.mean(np.multiply(X[:, 2], diff))
        dW[3] = np.mean(np.multiply(X[:, 3], diff))
        dW[4] = np.mean(np.multiply(X[:, 4], diff))
        dW[5] = np.mean(np.multiply(X[:, 5], diff))
        dW[6] = np.mean(np.multiply(X[:, 6], diff))
        dW[7] = np.mean(np.multiply(X[:, 7], diff))#
        db = np.mean(diff)
        W = W - alpha * dW
        b = b - alpha * db

        if ((i % 10000) == 0) :
            print(W, b)

    return [mean, stdDev, W, b]

main()
    
