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

# -------------------------------------------------------------------

torch.save(net.state_dict(), PATH + '/net.pth')

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)

        out = net(images)
        out = out.transpose(1, 2).transpose(2, 3).argmax(-1)

        total   += (labels.shape[0] * labels.shape[1] * labels.shape[2])
        correct += (out == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

# -------------------------------------------------------------------

def display(display_list, title):
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])

        disImg  = display_list[i].detach().numpy()
        plt.imshow(disImg)
        plt.axis('off')
    plt.show()

# -------------------------------------------------------------------

images, labels  = testset[0]
images          = images.unsqueeze(0)
images, labels  = images.to(device), labels.to(device)

out = net(images)

disInput    = images.squeeze(0).transpose(0, 1).transpose(1, 2)
disOutput   = out.squeeze(0).transpose(0, 1).transpose(1, 2).argmax(-1)

display([disInput.cpu(), labels.cpu(), disOutput.cpu()], ['Input Image', 'True Mask', 'Predicted Mask'])

# -----------------------------------------------------------------------------

images, labels  = testset[10]
images          = images.unsqueeze(0)
images, labels  = images.to(device), labels.to(device)

out = net(images)

disInput    = images.squeeze(0).transpose(0, 1).transpose(1, 2)
disOutput   = out.squeeze(0).transpose(0, 1).transpose(1, 2).argmax(-1)

display([disInput.cpu(), labels.cpu(), disOutput.cpu()], ['Input Image', 'True Mask', 'Predicted Mask'])

# ----------------------------------------------------------------------------------

images, labels  = testset[14]
images          = images.unsqueeze(0)
images, labels  = images.to(device), labels.to(device)

out = net(images)

disInput    = images.squeeze(0).transpose(0, 1).transpose(1, 2)
disOutput   = out.squeeze(0).transpose(0, 1).transpose(1, 2).argmax(-1)

display([disInput.cpu(), labels.cpu(), disOutput.cpu()], ['Input Image', 'True Mask', 'Predicted Mask'])