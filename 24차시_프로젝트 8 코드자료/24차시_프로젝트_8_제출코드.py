import numpy as np # numpy import
X = np.array([0, 0, 1, 1, 0, 1, 0, 1]).reshape(2,4) # 입력
Y = np.array([0, 1, 1, 0]).reshape(1,4) # 정답

print(X)
print(Y)

# 가중치 초기화 함수
def init_parameters(num_hidden_units = 2):
  W1 = np.random.randn(2, num_hidden_units) # 첫번째 레이어 가중치
  B1 = np.zeros((num_hidden_units,1)) # 첫번째 레이어 바이어스
  W2 = np.random.randn(num_hidden_units, 1) # 두번째 레이어 가중치
  B2 = np.zeros((1, 1)) # 두번째 레이어의 바이어스
  return W1, B1, W2, B2 # 가중치 파라미터 리턴

####################################################
######## 미션 코드 : 어파인함수와 시그모이드 함수 ###########
####################################################

# Affine transform을 구현하세요.
def affine(W, X, B):
  return np.dot(W.T,X) + B # '...'에 코드를 채워주세요.

# sigmoid function을 구현하세요.
def sigmoid(z):
  return 1/ (1+ np.exp(-z)) # '...'에 코드를 채워주세요.

# 임의로 w, b, x를 만들고 affine 함수를 테스트 해봅니다
w = np.arange(4).reshape(2,2)
b = [[1],[2]]
x = [[1],[1]]

print(affine(w, x, b)) # affine test -> [[3],[6]]

print(sigmoid(0.1)) # sigmoide test -> 0.524979

###########################################
####### 미션 코드 : 이진 크로스 엔트로피  ########
###########################################
def binary_cross_entropy(Y, YHat):
  N = Y.shape[1] # 총 샘플의 수
  loss = (-1/N) * np.sum(Y * np.log(YHat) + (1-Y) * np.log(1-YHat)) # '...'에 코드를 채워주세요
  return loss

# 정답 확인
Y = np.array([0, 1, 1, 0]).reshape(1, 4) # 정답
YHat = np.array([0.5, 0.5, 0.5, 0.5]).reshape(1, 4) # 추정값

loss = binary_cross_entropy(Y, YHat)
print("2진 크로스엔트로피 비용:", loss)

########################################
####### 미션 코드 : 2레이어 순방향 연산 #######
########################################

def forward_loss(X, Y, _params):
  W1, B1, W2, B2 = _params

  # 첫번째 레이어연산
  Z1 = affine(W1,X,B1) # 1) affine 함수  - '...'에 채워주세요
  H = sigmoid(Z1) # 2) sigmoid 함수 - '...'에 채워주세요

  # 두번째 레이어 연산
  Z2 = affine(W2,H,B2) # 3) affine 함수 - '...'에 채워주세요
  YHat = sigmoid(Z2) # 4) sigmoid 함수  - '...'에 채워주세요

  # 손실함수 계산
  loss = binary_cross_entropy(Y, YHat) # 5) 이진크로스 엔트로피 함수 - '...'에 채워주세요

  return Z1, H, Z2, YHat, loss

np.random.seed(42) # random seed로 고정
W1, B1, W2, B2 = init_parameters(num_hidden_units = 2) # 파라미터 초기화
forward_loss(X, Y, [W1, B1, W2, B2])[-1] # loss출력 : 0.70492209

######################################
####### 미션 코드 : 역전파 구현하기 ##########
######################################

def get_gradients(X, Y, _params):
  W1, B1, W2, B2 = _params
  m = X.shape[1] # 샘플의 수
  # 포워드 함수 통과 후 출력
    # - Z1 : 첫번재 레이어 affine 결과
    # - H : 첫번재 레이어 sigmoid 통과한 결과
    # - Z2 : 두번재 레이어 affine 통과한 결과
    # - YHat : 두번재 레이어 sigmoid 통과한 결과
    # - loss : 크로스엔트로피 손실값
  Z1, H, Z2, YHat, loss = forward_loss(X, Y, _params)

  # 1) dLoss/dZ2 구현. 손실함수가 각 샘플 손실의 평균으로 계산되기 때문에 그대로 구현하였습니다.
  dLdZ2 = (1/m)*(YHat-Y) # 그림에서 1의 구현

  # 2) dLoss/dW2의 구현 - '...'을 구현하세요.
  dLdW2 = np.dot(H,dLdZ2.T) # 그림에서 2의 구현 (초록색 2번 참고)

  # 3) dLoss/dB2의 구현 - 샘플마다 gradient가 있음. 따라서 합쳐줘야 함.
  dLdB2 = np.sum(dLdZ2, axis=1, keepdims=True) # 그림에서 3의 구현

  # 4) dLoss/dH의 구현 - '...'을 구현하세요.
  dLdH = np.dot(W2,dLdZ2) #  그림에서 4의 구현

  # 5) dLoss/dZ1의 구현 - '...'을 구현하세요.
  dLdZ1 = dLdH * H * (1-H) # 그림에서 5의 구현

  # 6) dLoss/dW1의 구현 - '...'을 구현하세요.
  dLdW1 = np.dot(dLdZ1, X.T) # 그림에서 6의 구현

  # 7) dLoss/dB2의 구현 - '...'을 구현하세요.
  dLdB1 = np.sum(dLdZ1, axis=1, keepdims=True)

  return [dLdW1, dLdB1, dLdW2, dLdB2], loss


# 모델 학습하기
def optimize (X, Y, _params, learning_rate = 0.1, iteration = 1000):

    params = np.copy(_params) # 파라미터 복사
    loss_trace = [] # 손실 값 저장

    for epoch in range(iteration): # 학습 반복
        dparams, loss = get_gradients(X, Y, params) # 그레디언트 추출
        for param, dparam in zip(params, dparams):
            param += - learning_rate * dparam # 경사하강법 구현

        if (epoch % 100 == 0): # 손실값 저장
            loss_trace.append(loss)

    _, _, _, Y_hat_predict, _ = forward_loss(X, Y, params) # 학습된 모델로 추론

    return params,loss_trace, Y_hat_predict

X = np.array([0, 0, 1, 1, 0, 1, 0, 1]).reshape(2,4) # 입력
Y = np.array([0, 1, 1, 0]).reshape(1,4) # 정답

params = init_parameters(2) # 파라미터 세팅
new_params, loss_trace, Y_hat_predict = optimize(X, Y, params, 0.1, 150000) # 학습 및 추론

print(Y_hat_predict) # 정답 Y와 유사한 값이 나왔다면 학습이 잘 진행된 것 입니다.

import matplotlib.pyplot as plt

# Plot learning curve (with costs)
plt.plot(loss_trace)
plt.ylabel('loss')
plt.xlabel('iterations (per hundreds)')
plt.show()