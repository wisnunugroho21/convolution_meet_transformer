import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim import AdamW

import torchvision
import torchvision.transforms as transforms

from dataloader.PennFudanPedDataset import PennFudanPedDataset
from dataloader.ClrPennFudanPedDataset import ClrPennFudanPedDataset
from dataloader.CatsDataset import CatsDataset

import matplotlib.pyplot as plt

from copy import deepcopy

from model.main import MainModel

epochs = 10
batch_size = 4
PATH = '.'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train = True, download = True, transform = transform)
trainloader = DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 4)

testset = torchvision.datasets.CIFAR10(root='./data', train = False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size = batch_size, shuffle = False, num_workers = 4)

# -----------------------------------------------------------------------------------------------------------

net = MainModel(num_class = 10).to(device)

segoptimizer    = AdamW(net.parameters(), lr = 6e-5)
segloss         = nn.CrossEntropyLoss()

for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        segoptimizer.zero_grad()

        images, labels    = data
        images, labels    = images.to(device), labels.to(device)

        out = net(images)

        loss = segloss(out, labels)
        loss.backward()
        segoptimizer.step()

        running_loss += loss.item()
        if i % 200 == 199:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')
torch.save(net.state_dict(), PATH + '/net.pth')

# -------------------------------------------------------------------

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        out = net(images)
        _, predictions = out.max(1)

        total   += labels.size(0)
        correct += (predictions == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# -------------------------------------------------------------------

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

# again no gradients needed
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        outputs = net(images)
        _, predictions = outputs.max(1)

        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')