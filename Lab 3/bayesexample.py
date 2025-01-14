import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

# Make some random data points in two sets
from sklearn.datasets import make_blobs
X, y = make_blobs(100, 2, centers=2, random_state=2, cluster_std=1.5)
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu')
#plt.show()

# Fit data to Gauss distribution
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)

# Generate new data and predict label
rng = np.random.RandomState(0)
Xnew = [-6, -14] + [14, 18] * rng.rand(2000, 2)
ynew = model.predict(Xnew)

# Plot data
plt.scatter(X[:,0], X[:,1], c=y, s=50, cmap='RdBu')
lim = plt.axis()
plt.scatter(Xnew[:,0], Xnew[:,1], c=ynew, s=20, cmap='RdBu', alpha=0.1)
plt.axis(lim)
#plt.show()

# Predict
yprob = model.predict_proba(Xnew)
yprob[-8:].round(2)
print(yprob[-8:].round(2))