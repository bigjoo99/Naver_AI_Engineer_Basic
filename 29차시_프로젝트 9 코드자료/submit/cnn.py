torch.manual_seed(7777) # 일관된 weight initialization을 위한 random seed 설정

## 코드 시작 ##
model = SimpleCNN()          # 위의 설명 1. 을 참고하여 ..... 을 채우세요.
## 코드 종료 ##

model = model.to(device)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

model = model.to(device)
val_every = 1
saved_dir = './saved/SimpleCNN'


model_path = './saved/ResNet50/best_model.pt'
# model_path = './saved/pretrained/ResNet50/best_model.pt' # 모델 학습을 끝까지 진행하지 않은 경우에 사용

## 코드 시작 ##

checkpoint = torch.load(model_path, map_location=device)
state_dict = checkpoint['net']

new_model.load_state_dict(state_dict)


## 코드 종료 ##