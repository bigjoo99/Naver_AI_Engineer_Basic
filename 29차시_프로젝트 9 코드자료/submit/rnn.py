## 코드 시작 ##
batch_size = 100
num_epochs = 30
learning_rate = 0.00003
## 코드 종료 ##

torch.manual_seed(7777) # 일관된 weight initialization을 위한 random seed 설정

## 코드 시작 ##
model = SimpleLSTM()       # 위의 설명 1. 을 참고하여 ..... 을 채우세요.
## 코드 종료 ##

model = model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

val_every = 1
saved_dir = './saved/LSTM'