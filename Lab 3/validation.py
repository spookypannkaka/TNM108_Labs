import pandas as pd
import numpy as np
from sklearn.calibration import cross_val_predict
from sklearn.feature_selection import RFE
from sklearn.model_selection import KFold, cross_val_score
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error

# Boston dataset is racist
#from sklearn.datasets import load_boston
#boston = load_boston()

data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

#boston = pd.read_csv('Boston.csv',header=None)
boston = raw_df

#X=boston.data
#Y=boston.target
#X = boston.iloc[:, :-1]
#Y = boston.iloc[:, -1]
X = data
Y = target

cv = 10
kf = KFold(n_splits=cv, shuffle=True, random_state=42)  # Shuffle the data before cross-validation

print('\nlinear regression')
lin = LinearRegression()
scores = cross_val_score(lin, X, Y, cv=kf) # cv=cv
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lin, X,Y, cv=kf)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))
print('\nridge regression')
ridge = Ridge(alpha=1.0)
scores = cross_val_score(ridge, X, Y, cv=kf) # cv=cv
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(ridge, X,Y, cv=kf) # cv=cv
print("MSE: %0.2f" % mean_squared_error(Y,predicted))
print('\nlasso regression')

lasso = Lasso(alpha=0.1)
scores = cross_val_score(lasso, X, Y, cv=kf) # cv=cv
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lasso, X,Y, cv=kf) # cv=cv
print("MSE: %0.2f" % mean_squared_error(Y,predicted))
print('\ndecision tree regression')
tree = DecisionTreeRegressor(random_state=0)
scores = cross_val_score(tree, X, Y, cv=kf) # cv=cv
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(tree, X,Y, cv=kf) # cv=cv
print("MSE: %0.2f" % mean_squared_error(Y,predicted))
print('\nrandom forest regression')
forest = RandomForestRegressor(n_estimators=50, max_depth=None, min_samples_split=2, random_state=0)
scores = cross_val_score(forest, X, Y, cv=kf) # cv=cv
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(forest, X,Y, cv=kf) # cv=cv
print("MSE: %0.2f" % mean_squared_error(Y,predicted))
print('\nlinear support vector machine')
svm_lin = svm.SVR(epsilon=0.2,kernel='linear',C=1)
scores = cross_val_score(svm_lin, X, Y, cv=kf) # cv=cv
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(svm_lin, X,Y, cv=kf) # cv=cv
print("MSE: %0.2f" % mean_squared_error(Y,predicted))
print('\nsupport vector machine rbf')
clf = svm.SVR(epsilon=0.2,kernel='rbf',C=1.)
scores = cross_val_score(clf, X, Y, cv=kf) # cv=cv
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(clf, X,Y, cv=kf) # cv=cv
print("MSE: %0.2f" % mean_squared_error(Y,predicted))
print('\nknn')
knn = KNeighborsRegressor()
scores = cross_val_score(knn, X, Y, cv=kf) # cv=cv
print("mean R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(knn, X,Y, cv=kf) # cv=cv
print("MSE: %0.2f" % mean_squared_error(Y,predicted))
ridge = Ridge()

'''best_features=4
rfe_lin = RFE(lin,best_features).fit(X,Y)
supported_features=rfe_lin.get_support(indices=True)
for i in range(0, 4):
    z=supported_features[i]
    print(i+1,boston.feature_names[z])'''

'''best_features = 4
rfe_lin = RFE(estimator=lin, n_features_to_select=best_features)
rfe_lin.fit(X, Y)
supported_features = rfe_lin.support_
print("Selected Features:")
for i in range(len(supported_features)):
    if supported_features[i]:
        print(i + 1, boston.columns[i])'''
from sklearn.feature_selection import RFE
best_features=4
print('feature selection on linear regression')
rfe_lin = RFE(estimator=lin, n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_lin.support_)
scores = cross_val_score(lin, X[:,mask], Y, cv=kf)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lin, X[:,mask],Y, cv=kf)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('feature selection ridge regression')
rfe_ridge = RFE(estimator=ridge,n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_ridge.support_)
scores = cross_val_score(ridge, X[:,mask], Y, cv=kf)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(ridge, X[:,mask],Y, cv=kf)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('feature selection on lasso regression')
rfe_lasso = RFE(estimator=lasso,n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_lasso.support_)
scores = cross_val_score(lasso, X[:,mask], Y, cv=kf)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(lasso, X[:,mask],Y, cv=kf)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('feature selection on decision tree')
rfe_tree = RFE(estimator=tree,n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_tree.support_)
scores = cross_val_score(tree, X[:,mask], Y, cv=kf)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(tree, X[:,mask],Y, cv=kf)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('feature selection on random forest')
rfe_forest = RFE(estimator=forest,n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_forest.support_)
scores = cross_val_score(forest, X[:,mask], Y, cv=kf)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(forest, X[:,mask],Y, cv=kf)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))

print('feature selection on linear support vector machine')
rfe_svm = RFE(estimator=svm_lin,n_features_to_select=best_features).fit(X,Y)
mask = np.array(rfe_svm.support_)
scores = cross_val_score(svm_lin, X[:,mask], Y, cv=kf)
print("R2: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
predicted = cross_val_predict(svm_lin, X,Y, cv=kf)
print("MSE: %0.2f" % mean_squared_error(Y,predicted))