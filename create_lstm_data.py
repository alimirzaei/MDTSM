
import numpy as np

dataset = np.load('dataset.npz')

X= dataset['X']

seq_number = 20
prediction_number = 20

X_input = []
Y_input = []

for x_entity in X:
    for i in range(prediction_number, x_entity.shape[1]-max(prediction_number, seq_number)):
        X_input.append(x_entity[:,i:i+seq_number])
        Y_input.append(x_entity[:,i-prediction_number:i+prediction_number])

X_input = np.array(X_input)
Y_input = np.array(Y_input)