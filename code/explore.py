# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3.10.4 ('venv-pytorch')
#     language: python
#     name: python3
# ---

# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

# %%
data_train = datasets.CIFAR100(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

data_test = datasets.CIFAR100(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)

loader_train = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=4)
loader_test = DataLoader(data_test, batch_size=64, shuffle=False, num_workers=4)

print(f"training data size: {len(data_train)}")
print(f"test data size: {len(data_test)}")

# %%
fig = plt.figure(figsize=(8, 4))
cols, rows = 4, 2
for i in range(1, cols * rows + 1):
    img, label = data_train[i]
    fig.add_subplot(rows, cols, i)
    plt.title(data_train.classes[label])
    plt.axis("off")
    plt.imshow(img.permute(1,2,0))
plt.show()

# %%
# standardize image data (y = (x - mean) / std) or dont?
sum = torch.zeros((3,))
sum_squared = torch.zeros((3,))
num = 0

loader_stat = DataLoader(data_train, batch_size=100, shuffle=False, num_workers=4)

for t in loader_stat:
    sum += t[0].permute(1,0,2,3).flatten(start_dim=1).sum(dim=(1))
    sum_squared += (t[0] ** 2).permute(1,0,2,3).flatten(start_dim=1).sum(dim=(1))
    num += len(t[0])

print("Sums: ", sum, sum_squared, num)


# %%
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 100)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)

net.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

for epoch in range(10000):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(loader_train, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
            running_loss = 0.0

print('Finished Training')

# %%
