# Import packages
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms,datasets
import time
import numpy as np

# 构建CNN网络结构，暂时先不用ResNet网络结构
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        # 如果没有BN层效果很糟糕！
        self.layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.classify = nn.Sequential(
            nn.Dropout(),
            nn.Linear(4 * 4 * 256,1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes))

    def forward(self, x):
        x = self.layers(x)
        # Tensor Flatten 拉直操作
        x = torch.flatten(x, 1)
        x = self.classify(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
model = CNN(num_classes=2).to(device)
print(model)
# Loss 函数定义 & 优化器的选择
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr= 0.001)

data_root = './dataset/dataset/PetImages'
# Data Preprocessing
transforms = transforms.Compose([
             transforms.Resize(64),
             transforms.ToTensor(),
             transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
])

# Divide 60% of dataset as training dataset,20% as validation dataset and 20% as test_set
training_dataset = datasets.ImageFolder(data_root,transform= transforms)
num_train = len(training_dataset)
indices = list(range(num_train))

# shuffle the training index
np.random.shuffle(indices)
# 20% and 40% slice to split the data
split = int(np.floor(0.2*num_train))

train_idx,valid_idx,test_idx = indices[2*split:],indices[split:2*split],indices[:split]
# We use subrandom sampler, of course you can select DataLoader(dataset,shuffle = True,batch_size = ) function
training_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
# Create training loader and valid loader
training_loader = DataLoader(training_dataset,sampler = training_sampler,batch_size = 32)
valid_loader = DataLoader(training_dataset,sampler = valid_sampler,batch_size = 32)

Accuracy =[]
Epoch =[i for i in range(100)]
# Initialize accuracy to save pth model
acc =None

# Start Training! Start 100 epoch to train classification model
for epoch in range(100):
    model.train()
    #===============================================Train========================================================
    for batch_id, (img,label) in enumerate(training_loader):
        start_time = time.time()
        print('Start Training')
        img,label = img.to(device),label.to(device)
        predict = model(img).to(device)
        loss = criterion(predict,label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_id%100 ==0:
            print('Iteration:%d'%batch_id,'Loss is :%f'%loss.item())

     #=======================================Validation==================================================
    correct = 0
    total_num = len(valid_loader)*32
    model.eval()
    for x,label in valid_loader:
        x,label = x.to(device),label.to(device)
        predict = model(x)
        pred = predict.argmax(dim = 1)
        correct +=torch.eq(pred,label).sum().item()
    accuracy = correct/total_num
    # save model
    if acc is None:
        acc = accuracy
        continue
    else:
        if accuracy>=acc:
            print('Saving Model ....')
            torch.save(model,'./model/catvsdogs.pth')

    end_time = time.time()
    loading_time = end_time - start_time
    print('Accuracy:{},Time cost:{}'.format(accuracy,loading_time))
    Accuracy.append(accuracy)

    plt.figure()
    plt.plot(Epoch,Accuracy,color = 'red',label = 'Accuracy')
    plt.show()

