from sklearn.preprocessing import StandardScaler, LabelEncoder
import csv
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

attr_label = ['Water', 'Fire', 'Earth', 'Light', 'Dark']
race_label = ['God', 'Human', 'Demon', 'Beast', 'Dragon', 'Elf', 'Machina', 'Material']
attr_encoder = LabelEncoder()
attr_encoder.fit(attr_label)
race_encoder = LabelEncoder()
race_encoder.fit(race_label)

folder_path = ''
""" read train.csv file """
train_filename = 'train.csv'
with open(folder_path+train_filename, newline='') as csvfile:
    train_csv = csv.DictReader(csvfile)
    X = []
    num_feats = ['Hit Point', 'Attack Point', 'Recovery', 'Total']
    Y = []
    for row in train_csv:
        num = []
        for key in num_feats:
            num.append(int(row[key]))
        num.append(attr_encoder.transform([row['Attribute']])[0])
        X.append(num)
        Y.append(race_encoder.transform([row['Race']])[0])
SC = StandardScaler()
SC.fit(X)
X = SC.transform(X)
x_train , y_train = X, Y

""" read test.csv file """
answer = {"id":[], "Race":[]} # store the answers
test_filename = 'test.csv'
with open(folder_path+test_filename, newline='') as csvfile:
    test_csv = csv.DictReader(csvfile)
    x_test = []
    num_feats = ['Hit Point', 'Attack Point', 'Recovery', 'Total']
    for row in test_csv:
        num = []
        answer['id'].append(row['id'])
        for key in num_feats:
            num.append(int(row[key]))
        num.append(attr_encoder.transform([row['Attribute']])[0])
        x_test.append(num)
SC2 = StandardScaler()
SC2.fit(x_test)
x_test = SC.transform(x_test)

""" preparing dataset """
class ToS_dataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.tensor(X, dtype= torch.float32)
        self.Y = Y
  
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

train_data_len = len(x_train)
train_ratio = 0.75
train_dataset = ToS_dataset(x_train[:int(train_data_len*train_ratio)], y_train[:int(train_data_len*train_ratio)])
train_loader = DataLoader(train_dataset, batch_size= 64, shuffle= True)
valid_dataset = ToS_dataset(x_train[int(train_data_len*train_ratio):], y_train[int(train_data_len*train_ratio):])
valid_loader = DataLoader(valid_dataset, batch_size= 64, shuffle= True)

# TODO: Generate test data and dataloader.
class ToS_test_data(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype= torch.float32)
  
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]
test_dataset = ToS_test_data(x_test)
test_loader = DataLoader(test_dataset, batch_size=1)

""" Define Model """
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # TODO: Define each layer of your model here
        # Use methods or classes from torch.nn
        self.model = nn.Sequential(
            nn.Linear(5, 512),
            nn.Dropout(0.4), 
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(128, 8))

    def forward(self, x):
        # TODO: Define the forward pass
        return self.model(x)

cpu_gpu = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(cpu_gpu)
model = DNN().to(cpu_gpu)

""" Define Loss Function """
criterion = nn.CrossEntropyLoss()

""" Define Optimizer """
optim = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=5e-4)

num_epochs = 150
history = {"besti": -1, "best_loss": 9999}
for epoch in range(num_epochs):
    epoch_loss = 0
    valid_loss = 0
    model.train()
    for step, (X, target) in enumerate(train_loader):
        optim.zero_grad()
        y = model(X.to(cpu_gpu))
        loss = criterion(y, target.view(-1).to(cpu_gpu, dtype=torch.int64))
        loss.backward()
        epoch_loss += loss.data
        optim.step()

    model.eval()
    with torch.no_grad():
      for step, (X, target) in enumerate(valid_loader):
            y = model(X.to(cpu_gpu))
            loss = criterion(y, target.view(-1).to(cpu_gpu, dtype=torch.int64))
            valid_loss += loss.data
    print("epoch: %3d,\t training loss: %.3f,\t validation loss: %.3f"%(epoch, epoch_loss.data, valid_loss.data))
    if valid_loss < history['best_loss']:
        history['besti'] = epoch
        history['best_loss'] = valid_loss.data
print(history)

""" Testing """
model.eval()
with torch.no_grad():
    for step, X in enumerate(test_loader):
        y = model(X.to(cpu_gpu))
        _, preds_tensor = torch.max(y,1)
        preds = np.squeeze(preds_tensor.cpu().numpy())
        answer['Race'].append(race_encoder.inverse_transform([preds])[0])

df = pd.DataFrame(answer)
df = df[["id", "Race"]]
df.to_csv("./submission.csv", index=False)
