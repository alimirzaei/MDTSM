import numpy as np


x = np.linspace(0, 100, 1000)

amp = [0.25, 0.5, 0.75, 1]
freq = [1,2,3,4]

X = []
Y = []
for a in amp:
    for f in freq:
        X.append(a*np.sin(f*x)+.1*np.random.randn(len(x)))
        Y.append([a,f])

X = np.array(X).reshape(16, 1, 1000)
Y = np.array(Y)

np.savez('dataset.npz', X=X, Y=Y)