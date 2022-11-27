import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, KFold,cross_val_score,cross_val_predict
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, RidgeCV,LassoCV,ElasticNetCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error

boston_dataset = load_boston() 
print(boston_dataset.keys())

print(boston_dataset.DESCR)

boston = pd.DataFrame(boston_dataset.data) 
print(boston.head())

boston['MEDV'] = boston_dataset.target 
correlation_matrix = boston.corr().round(2)
print(correlation_matrix)

X = boston_dataset.data[:, [5, 8]] 
y = boston_dataset.target

print ("X : ", len(X), " y : ", len(y))
# BURAYA NEDEN İKİ LENGHTİN AYNI OLDUĞUNU YAZACAKSIN
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5) 
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

linearRegression = LinearRegression() 
linearRegression.fit(X_train, y_train)
fx_test = linearRegression.predict(X_test)
mse = mean_squared_error(y_test, fx_test)
print('RMSE is {}'.format(mse))
(minv, maxv) = (y_test.min(),y_test.max())
fig,ax=plt.subplots()
ax.scatter(y_test,fx_test,marker="o",s=5) # points of size 5
ax.plot([minv, maxv],[minv, maxv]) #y=f(x) ideal line
ax.set_xlabel("real")
ax.set_ylabel("predicted")
plt.show()

firstResults = linearRegression.predict(X_test)
cv = KFold(n_splits=10, random_state=1, shuffle=True)
lastResults = cross_val_predict(linearRegression, X_test, y_test,cv=cv, n_jobs=-1)
firstResultsMse = mean_squared_error(y_test, firstResults)                          
lastResultsMse = mean_squared_error(y_test, lastResults)
print("Before : ", firstResultsMse , " After : ", lastResultsMse)


lassoRegression = Lasso()
lassoRegression.fit(X_train, y_train)
fx_test = lassoRegression.predict(X_test)
mse = mean_squared_error(y_test, fx_test)
print('RMSE is {}'.format(mse))
(minv, maxv) = (y_test.min(),y_test.max())
fig,ax=plt.subplots()
ax.scatter(y_test,fx_test,marker="o",s=5) # points of size 5
ax.plot([minv, maxv],[minv, maxv]) #y=f(x) ideal line
ax.set_xlabel("real")
ax.set_ylabel("predicted")
plt.show()


# lassoRegression tuned parameters
lasso_cv_model = LassoCV(alphas=None,cv=10,max_iter=10000,normalize=True)
lasso_cv_model.fit(X_train, y_train)
lassoRegression = Lasso(alpha = lasso_cv_model.alpha_)



ridgeRegression = Ridge()
ridgeRegression.fit(X_train, y_train)
fx_test = ridgeRegression.predict(X_test)
mse = mean_squared_error(y_test, fx_test)
print('RMSE is {}'.format(mse))
(minv, maxv) = (y_test.min(),y_test.max())
fig,ax=plt.subplots()
ax.scatter(y_test,fx_test,marker="o",s=5) # points of size 5
ax.plot([minv, maxv],[minv, maxv]) #y=f(x) ideal line
ax.set_xlabel("real")
ax.set_ylabel("predicted")
plt.show()

#ridgeRegression tuned Parameters
lambdalar = 10 ** np.linspace(10, -2, 100) * 0.5
ridgeRegression = RidgeCV(alphas=lambdalar,scoring="neg_mean_squared_error",normalize=True)

elasticNetRegression = ElasticNet()
elasticNetRegression.fit(X_train, y_train)
fx_test = elasticNetRegression.predict(X_test)
mse = mean_squared_error(y_test, fx_test)
print('RMSE is {}'.format(mse))
(minv, maxv) = (y_test.min(),y_test.max())
fig,ax=plt.subplots()
ax.scatter(y_test,fx_test,marker="o",s=5) # points of size 5
ax.plot([minv, maxv],[minv, maxv]) #y=f(x) ideal line
ax.set_xlabel("real")
ax.set_ylabel("predicted")
plt.show()


#elasticNetRegression tuned parameters
enet_cv_model = ElasticNetCV(cv=10, random_state=0).fit(X_train, y_train)
elasticNetRegression = ElasticNet(alpha=enet_cv_model.alpha_)


#son kısımdaki her bir cross_validation kısımlarını yaptıktan sonra tahmin değerlerinin hata kareler ortalamasına göre kıyaslayıp hangisini kullanıcağını yazmalısın 