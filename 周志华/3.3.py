import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model

data = np.array([[0.697, 0.460, 1],
        [0.774, 0.376, 1],
        [0.634, 0.264, 1],
        [0.608, 0.318, 1],
        [0.556, 0.215, 1],
        [0.403, 0.237, 1],
        [0.481, 0.149, 1],
        [0.437, 0.211, 1],
        [0.666, 0.091, 0],
        [0.243, 0.267, 0],
        [0.245, 0.057, 0],
        [0.343, 0.099, 0],
        [0.639, 0.161, 0],
        [0.657, 0.198, 0],
        [0.360, 0.370, 0],
        [0.593, 0.042, 0],
        [0.719, 0.103, 0]])

x = data[:, :2]
y = data[:, 2]
X = np.array([row.copy() for row in data])
X[:, 2] = np.ones(x.shape[0])

# parameter initialized, no specific reason for initial numbers
B = np.random.rand(1, x.shape[1] + 1)
step = 0.5

# logistic regression
original_prediction = 1 / (1 + np.exp(-B.dot(X.T)))
for i in range(1000):
  y_ = 1 / (1 + np.exp(-B.dot(X.T)))
  derivative = (X.T).dot((y_ - y).T)
  B -= (derivative * step).T
final_prediction = 1 / (1 + np.exp(-B.dot(X.T)))

# final result compare with sklearn's training model
lr = linear_model.LogisticRegression(solver='lbfgs', C=1000)
lr.fit(x, y)
print([lr.coef_, lr.intercept_])
#array([[ 3.03909749, 11.95570987]]), array([-4.24959139])
print(B)
#array([[ 3.15828505, 12.52102681, -4.42880585]])

# suggested reading: https://github.com/han1057578619/MachineLearning_Zhouzhihua_ProblemSets/blob/master/ch3--线性模型/3.3/3.3-LogisticRegression.py
