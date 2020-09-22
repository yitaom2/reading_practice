import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import linear_model

X = pd.read_csv("diabetes_data_upload.csv")
le = preprocessing.LabelEncoder()
X = X.apply(le.fit_transform)

y = X["class"]
x = X.drop("class",axis=1)

def minimize_variable_value(x_test, y_test, weight, intercept):
    temp = np.dot(x_test, weight.reshape(-1,1)) + intercept
    return_value = np.sum(np.dot(np.array(y_test), temp) + np.log(1 + np.exp(temp)))
    return return_value

# 留一法
average_error_1 = 0
accuracy_1 = 0
for i in range(len(x)):
    x_test = x.iloc[i].to_numpy()
    y_test = y.iloc[i]
    x_train = x.drop(i, axis=0).to_numpy()
    y_train = y.drop(i, axis=0).to_numpy()
    lr = linear_model.LogisticRegression(solver='lbfgs', C=2)
    lr.fit(x_train, y_train)
    average_error_1 += minimize_variable_value(x_test, y_test, lr.coef_, lr.intercept_) / len(x)
    accuracy_1 += lr.score(np.array([x_test]), np.array([y_test])) / len(x)

# 十折交叉法
block_size = int(len(x) / 10)
average_error_10 = 0
accuracy_2 = 0
for i in range(10):
    x_test = x.iloc[list(range(i*block_size, (i+1)*block_size))].to_numpy()
    y_test = y.iloc[list(range(i*block_size, (i+1)*block_size))]
    x_train = x.drop(list(range(i*block_size, (i+1)*block_size)), axis=0).to_numpy()
    y_train = y.drop(list(range(i*block_size, (i+1)*block_size)), axis=0).to_numpy()
    lr = linear_model.LogisticRegression(solver='lbfgs', C=2)
    lr.fit(x_train, y_train)
    average_error_10 += minimize_variable_value(x_test, y_test, lr.coef_, lr.intercept_) / (10 * block_size)
    accuracy_2 += lr.score(x_test, y_test) / 10

print(average_error_1, average_error_10)
print(accuracy_1, accuracy_2)
"""
output
6.571718798678939 162.97464123669926
0.9288461538461508 0.9173076923076923
"""
