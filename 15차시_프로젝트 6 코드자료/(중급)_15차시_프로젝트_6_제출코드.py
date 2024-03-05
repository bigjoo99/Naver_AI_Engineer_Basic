import numpy as np

xy = np.array([[1., 2., 3., 4., 5., 6.],
              [10., 20., 30., 40., 50., 60.]])

## 코드시작 ##

x_train = xy[0]    # ... 에 알맞은 코드를 작성해보세요.

y_train = xy[1]    # ... 에 알맞은 코드를 작성해보세요.


## 코드종료 ##

print(x_train, x_train.shape)
print(y_train, y_train.shape)


## 코드시작 ##

beta_gd = np.random.rand(1)
bias = np.random.rand(1)

## 코드종료 ##

print(beta_gd, bias)


learning_rate = 0.001

## 코드시작 ##
for i in range(1000):
    hypothesis = x_train * beta_gd + bias

    cost = np.mean((hypothesis - y_train) ** 2)
    
    w_gradient = np.mean((hypothesis - y_train) * x_train)
    b_gradient = np.mean(hypothesis - y_train)
    
    beta_gd = beta_gd - learning_rate * w_gradient
    bias = bias - learning_rate * b_gradient
    
    if i % 100 == 0:
        print(f"Epoch ({i}/1000)    cost: {cost:.6f},   w: {beta_gd.item():.6f},    b: {bias.item():.6f}")

## 코드종료 ##