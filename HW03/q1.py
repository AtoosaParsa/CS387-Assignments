import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score 
import time
from preprocessing import preprocessing

def predict(x, y, theta):
    y_predict = np.matmul(x, theta).flatten()
    loss = ((y_predict-y) ** 2).mean()
    return y_predict, loss

# data from https://www.kaggle.com/shivachandel/kc-house-data
# loading the data
data = pd.read_csv('kc_house_data.csv')
# preprocessing
X, Y = preprocessing(data)

x_data = X.to_numpy()
y_data = Y.to_numpy()

features = x_data.shape[1]
dataSize = x_data.shape[0]

# normalize the data to avoid big numbers and exploding gradients!
for i in range(0, features):
    x_data[:,i] = x_data[:,i]/x_data[:,i].mean()

y_data[:] = y_data[:] / y_data.mean()

# divide the data into train and test
trainingSize = int(0.8 * dataSize)

x_train = x_data[0:trainingSize, :]
y_train = y_data[0:trainingSize]

x_test = x_data[trainingSize:, :]
y_test = y_data[trainingSize:]

# initial point for parameters        
theta = np.zeros([features,1])

# parameter of gradient descent
gamma = 1e-5
epochs = 5000
batchSize = 100

trainLoss = []

t0 = time.time()

for e in range(epochs):
    for i in range(0, trainingSize, batchSize):
        
        # get the batches
        x_batch = x_train[i:i + batchSize, :]
        y_batch = y_train[i:i + batchSize]

        y_predict = np.matmul(x_batch, theta).flatten()
        error = y_batch - y_predict

        gradient = -2 * np.matmul(x_batch.T, np.expand_dims(error,-1))
        
        theta = theta - gamma * gradient
    
    # calculate the training loss
    loss = (error ** 2).sum()
    trainLoss.append(loss)
    print("epoch "+str(e)+": "+str(loss))
    
t1 = time.time()
delta = t1 - t0
print("training done")
print("time passed: "+str(delta))

# now get the prediction and calculate loss and r2 score on test data
y_pred, loss = predict(x_test, y_test, theta)
score = r2_score(y_test, y_pred)
print("r2-score is: "+str(score))
print("loss is: "+str(loss))

# plotting
plt.figure()
plt.grid(color='silver', linestyle='-', linewidth=0.2)
plt.plot(list(range(1, epochs+1)), trainLoss, color='blue')
plt.xlabel("Epochs")
plt.ylabel("Training Loss")
#plt.title("", fontsize='small')
plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
plt.tight_layout()
#plt.legend(['two robot', 'three robots'], loc='upper left')
#plt.savefig("compare.pdf")
plt.show()