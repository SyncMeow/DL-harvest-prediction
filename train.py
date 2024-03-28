import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import json
import set_split
import random

class NNModel(nn.Module):
    def __init__(self, input_dim=27, output_dim=1, num_hidden_layers=30 , hidden_dim=60, p_dropout=0.3):
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        self.dropout = nn.Dropout(p= p_dropout)

        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))

        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        for idx, layer in enumerate(self.hidden_layers):
            if (idx+1 % 5):
                x = self.dropout(x)
            x = torch.relu(layer(x))
        return self.output_layer(x)

class data_set(Dataset):
    def __init__(self, data, label):
        data = torch.tensor(data).float()
        data = F.normalize(data)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.tensor(self.data[index]).float(), torch.tensor(self.label[index]).float()

def fetch_data(PATH):
    set_split.split_raw_data(5)

    with open(PATH, "r", encoding = "utf-8") as r:
        jdata = json.load(r)

    DATA = []
    LABEL = []
    for value in jdata["data"]:
        DATA.append(value[0])
        LABEL.append(value[1])
    return DATA, LABEL

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

batch_size = 24
n_epoch = 100

DATA, LABEL = fetch_data("./train_data.json")
datasplit = int(len(DATA)/5)
DATA1, LABEL1 = DATA[datasplit:], LABEL[datasplit:]
DATA2, LABEL2 = DATA[:datasplit], LABEL[:datasplit]

train_dataset = data_set(DATA1, LABEL1)
val_dataset = data_set(DATA2, LABEL2)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = NNModel()
model.to(device)
learning_rate = 0.001
criterion = nn.MSELoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

def fit(epoch, model, dataloader, training = True):
    if training:
        model.train()
    else:
        model.eval()

    mode = "training" if training else "validation"
    running_loss = 0.0

    cnt = 0
    for index, (data, target) in enumerate(dataloader):
        cnt += 1
        data, target = data.to(device), target.to(device)

        output = model(data)
        loss = criterion(output, target)

        running_loss += loss.item()

        if (training):
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    #loss = running_loss / len(dataloader.dataset)
    loss = running_loss / cnt
    print(f'<Epoch {epoch}> {mode} loss: {round(loss, 4)}')
    return loss

#training
train_losses, val_losses = [], []

for epoch in range(1, n_epoch+1):
    epoch_train_loss = fit(epoch, model, train_dataloader, True)
    train_losses.append(epoch_train_loss)
    epoch_val_loss = fit(epoch, model, val_dataloader, False)
    val_losses.append(epoch_val_loss)

torch.save(model.state_dict(), './model.ckpt')
print("Model saved")

#visualize
plt.plot(range(1, len(train_losses)+1), train_losses, 'bo', label = 'training loss')
plt.plot(range(1, len(val_losses)+1), val_losses, 'r', label = 'validation loss')
plt.legend()

#test
model_for_test = NNModel(p_dropout=0.3)
model_for_test.load_state_dict(torch.load('./model.ckpt'))

TESTDATA, TESTLABEL = fetch_data("./test_data.json")

avg_error = 0

for i in range(len(TESTDATA)):
    with torch.no_grad():
        pred_y = model_for_test(torch.tensor(DATA[i]))

        error = abs(pred_y[0]-LABEL[i][0])/LABEL[i][0]

        print(f"{i}  y:{LABEL[i]}  pred_y:{pred_y[0]}  error:{error} ({round(error.item()*100, 2)}%)")
        avg_error += error.item()

print(f"average error: {round(avg_error*100/5, 2)}%")

plt.show()