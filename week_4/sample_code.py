from torch import nn, cuda, device, FloatTensor, as_tensor, int64, optim, argmax, no_grad
from torch.cuda import is_available
from sklearn import preprocessing
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import csv
cpu_gpu = device("cuda" if is_available() else "cpu")

""" for transform words and category numbers """

attribute_encoder = preprocessing.LabelEncoder()
attribute_encoder.fit(['Water', 'Fire', 'Earth', 'Light', 'Dark'])
race_encoder = preprocessing.LabelEncoder()
race_encoder.fit(['God', 'Human', 'Demon', 'Beast', 'Dragon', 'Elf', 'Machina', 'Material'])

""" read train.csv file """

train_filename = "./train.csv"
with open(train_filename, newline='') as csvfile:
    train_csv = csv.DictReader(csvfile)
    X = []
    num_feats = ['Hit Point', 'Attack Point', 'Recovery', 'Total']
    Y = []
    for row in train_csv:
        num = []
        for key in num_feats:
            num.append(int(row[key]))
        num.append(attribute_encoder.transform(([row['Attribute']])))
        X.append(num)
        Y.append(race_encoder.transform([row['Race']]))
            

""" read test.csv file """
answer = {"id":[], "Race":[]} # store the answers

# TODO: Load the test data.


""" Preparing Dataset """
class ToS(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
    
    def __len__(self):
        return len(self.X_num)
    
    def __getitem__(self, idx):
        # TODO: Return Input and Target (X, Y) pair.
        pass

data_length = len(X)
train_data = ToS(X[:int(data_length*0.8)], Y[:int(data_length*0.8)])
train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
valid_data = ToS(X[int(data_length*0.8):], Y[int(data_length*0.8):])
valid_loader = DataLoader(train_data, batch_size = 64, shuffle = True)

# TODO: Generate test data and dataloader.

""" Define Model """
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        # TODO: Define each layer of your model here
        # Use methods or classes from torch.nn
        self.model = nn.Sequential(nn.Linear(5, 512), nn.Dropout(0.6), nn.LeakyReLU(),  nn.Linear(512, 8))

    def forward(self, x):
        # TODO: Define the forward pass
        pass

model = DNN().to(cpu_gpu)
""" Define Loss Function """
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
""" Training """
num_epochs = 100
for epoch in tqdm(range(num_epochs)):
    epoch_loss = 0
    valid_loss = 0
    history = {"besti": -1, "best_loss": 999}
    model.train()
    for step, (X, target) in enumerate(train_loader):
        optimizer.zero_grad()
        y = model(X.to(cpu_gpu))
        loss = criterion(y, target.view(-1).to(cpu_gpu, dtype=int64))
        loss.backward()
        epoch_loss += loss.data
        optimizer.step()
    """ validation"""
    model.eval()
    with no_grad():
        # TODO: Validation part. Use the validation data
        pass
    print("epoch: %3d, training loss: %.3f"%(epoch, epoch_loss))
    print("epoch: %3d, validation loss: %.3f"%(epoch, valid_loss))
    if valid_loss < history['best_loss']:
        history['besti'] = step
        history['best_loss'] = valid_loss
    

""" Testing """
model.eval()
with no_grad():
    # TODO: Testing part. Use the test data
    # Remember to fill in "answer"
    pass

df = pd.DataFrame(answer)
df = df[["id", "Race"]]
df.to_csv("./submission.csv", index=False)