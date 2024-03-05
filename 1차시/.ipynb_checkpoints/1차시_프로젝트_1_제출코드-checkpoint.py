import sys
print(sys.version)

import pandas as pd

## 코드시작 ##

df_csv = pd.read_csv('zen of python.csv')

## 코드종료 ##

print(df_csv)

import numpy as np

# 주어진 리스트
data = [10, 25, 40, 55, 70]

## 코드시작 ##
# 1. numpy 배열로 변환하여 출력하세요.
np_data = np.array(data)
print(np_data)

# 2. 리스트의 모든 원소의 합을 계산하여 출력하세요.
print(np.sum(np_data))

# 3. 리스트의 모든 원소의 평균을 계산하여 출력하세요.
print(np.mean(np_data))

# 4. 리스트의 최댓값과 최솟값을 출력하세요.
print(np.max(np_data))
print(np.min(np_data))

# 5. 각 원소를 10으로 나누는 연산을 수행하고 결과를 출력하세요.
np_data_divide = np_data/10
print(np_data_divide)

# 6. 각 원소를 3으로 거듭제곱하는 연산을 수행하고 결과를 출력하세요.

np_data_exp = np_data**3
print(np_data_exp)

## 코드종료 ##