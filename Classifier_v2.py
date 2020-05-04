import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# Using ImageFolder, load data
# The output of torchvision data sets are PILImage images of range [0,1].
# We transform them to Tensors of normalized range [-1,1]
# =============================================================================
transform = transforms.Compose(
    [transforms.ToTensor(),
      transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])

trainset = torchvision.datasets.ImageFolder(root = '/Users/ChrisYoon/Desktop/ME396P/Project',
                                            transform = transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers = 2)

# # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
# #                                         downlotad=True, transform=transform)
# # trainloader = torch.utils.data.DataLoader(trainset, batch_size = 4,
# #                                           shuffle=True, num_workers=2)


# # We load testset to test the classifier after it is trained 

# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                         download=True, transform=transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                           shuffle=False, num_workers = 2)
testset = torchvision.datasets.ImageFolder(root = '/Users/ChrisYoon/Desktop/ME396P/Test',
                                            transform = transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=True, num_workers = 2)
classes = ('Cubic', 'Hexagonal', 'Monoclinic', 'Orthorhombic', 'Rhombohedral', 'Triclinic')



# # =============================================================================
# # Here we will show some of the training images, just for fun 
# # =============================================================================

# # functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))


### 
# =============================================================================
# Now the real part: Define a convolutional neural network
# Copy the neural network from the Neural Networks section before and
# modify it to take 3-channel images (instead 1-channel images as it was defined)
# =============================================================================
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1 = nn.Linear(16*102*102, 120)
        self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = x.view(x.size(0), 332928)
        # x = x.view(x.size(0), 16*102*102)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()


# =============================================================================
# Define a loss function and optimizer:
# We will choose classification cross-entropy loss and SGD (Stochastic Gradient Descent) 
# with momentum
# =============================================================================
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


# =============================================================================
# Train the network
# We simply loop over our data iterator, and feed the inputs to the network and optimize
# =============================================================================
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 200 == 199:    # print every 20 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 200))
            running_loss = 0.0

print('Finished Training')

# =============================================================================
# Saving our trained model
# # =============================================================================
PATH = './crystalstruct.pth'
torch.save(net.state_dict(),  PATH)

# =============================================================================
# We have trained the network for 2 passes over the training dataset. Did it learn anything?
# To check this, let's have the classifier output a prediction of an image and compare it to
# ground-truth. 
# =============================================================================
dataiter = iter(testloader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

# Load the saved model (isn't super necessary here)
net = Net()
net.load_state_dict(torch.load(PATH))

# Now we see what the neural network thinks these examples are 
outputs = net(images)


# The outputs are 'energies' for the 10 clases. The higher the energy for a class,
# the more network thinks that the image is of that particular class. 
_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                              for j in range(4)))


# =============================================================================
# Let's look at how the network performs on the whole dataset 
# =============================================================================
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on all the test images: %d %%' % (
    100 * correct / total))


class_correct = list(0. for i in range(6))
class_total = list(0. for i in range(6))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(6):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))