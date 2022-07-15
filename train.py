from logging import critical
import numpy as np
import json

from sklearn import datasets
from language_utils import normalization_text,convert_to_vec,tokenize
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from model import NeuralNet

with open('intents.json','r') as f:
    intents = json.load(f)

all_words = []
tags=[]
xy=[]

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = normalization_text(pattern)
        # w = filter(w)
        w = tokenize(w)
        all_words.extend(w)
        xy.append((w,tag))

# remove duplicate
all_words = sorted(set(all_words))
tags = sorted(set(tags))


X_train =[]
Y_train =[]
for (pattern_sentence,tag) in xy:
    bag = convert_to_vec(pattern_sentence,all_words)
    X_train.append(bag)

    label =tags.index(tag)
    Y_train.append(label)

X_train = np.array(X_train)
# Tạo kiểu dữ liệu phù hợp
Y_train = np.array(Y_train,dtype=torch.LongTensor)

class ChatBotDataSet(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = Y_train
    
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples
    

#hyperparameters
batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

dataset = ChatBotDataSet()
train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

device =torch.device('cpu')
model = NeuralNet(input_size=input_size,hidden_size=hidden_size,num_classes=output_size)


#loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epochs):
    for(words,labels) in train_loader:
        words =words.to(device)
        labels =labels.to(device)
    
        #forward 
        outputs = model(words)
        loss = criterion(outputs,labels)

        #backward and optim step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if(epoch+1)%100==0:
        print(f'epoch {epoch+1}/{num_epochs},loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')



data = {
    "model_state":model.state_dict(),
    "input_size":input_size,
    "hidden_size":hidden_size,
    "output_size":output_size,
    "all_words":all_words,
    "tags":tags,
}

FILE = "data.pth"
torch.save(data,FILE)

print("training successed,file saved!")