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