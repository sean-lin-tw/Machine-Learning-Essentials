# Defining and Loading Data
```
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    # reads data & preprocesses
    def __init__(self, file):
        self.data = ...

    # returns one sample at a time
    def __getitem__(self, index):
        return self.data[index]

    # returns the size of the dataset
    def __len__(self):
        return len(self.data)

dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size=5, shuffle=False)
```
# Tensor Operations
```
# creation
x = torch.tensor([[1, -1], [-1, 1]])
x = torch.from_numpy(np.array([[1, -1], [-1, 1]]))
x = torch.zeros([2, 2])
x = torch.ones([1, 2, 5])

# moving tensor to devices
x = x.to('cpu')
x = x.to('cuda')
```
# Defining a Network
- Linear Layer
- Non-linear layer
```
import torch.nn as nn

class MyModel(nn.Module):
    # Define model layers
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)
```

# Choosing a Loss Function
```
criterion = nn.MSELoss() # Mean Squared Error (for regression tasks)
criterion = nn.CrossEntropyLoss() # Cross Entropy (for classification tasks)
loss = criterion(model_output, expected_value)
```

# Choosing an Optimizer
- `torch.optim`: gradient-based optimization algorithms
```
optimizer = torch.optim.SGD(model.parameters(), lr, momentum = 0)
```
- For every batch of data:
 1. Call optimizer.zero_grad() to reset gradients of model parameters.
 2. Call loss.backward() to backpropagate gradients of prediction loss.
 3. Call optimizer.step() to adjust model parameters

# Saving and Loading the Model
```
torch.save(model.state_dict(), path)
ckpt = torch.load(path)
model.load_state_dict(ckpt)
```

# Overall Code Template
```
dataset = MyDataset(file)
training_set = DataLoader(dataset, 16, shuffle=True)
mode = MyModel().to(device) # construct model and move to device (cpu/cuda)
loss_func = nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), 0.1) 

# training loop
for epoch in range(n_epochs):
    model.train() # train mode
    for x, y in training_set:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        pred = model(x) # forward pass (compute output)
        loss = loss_func(pred, y) 
        loss.backward() # compute gradient (backpropagation)
        optimizer.step() # update model with optimizer

# validation loop
model.eval() # evaluation mode
total_loss = 0
for x, y in validation_set:
    x, y = x.to(device), y.to(device)
    with torch.no_grad(): # disable gradient calculation
        pred = model(x)
        loss = loss_func(pred, y)
    total_loss += loss.cpu().item() * len(x) # accumulate loss
    avg_loss = total_loss / len(validation_set.dataset) # compute averaged loss

# prediction loop
model.eval()
preds = []
for x in test_set:
    x = x.to(device)
    with torch.no_grad(): # disable gradient calculation
        pred = model(x)
        preds.append(pred.cpu())
```