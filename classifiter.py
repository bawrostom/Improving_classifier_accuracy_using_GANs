import os
import torch
from torch import nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from torchvision.io import read_image
from torch.utils.data import DataLoader

# setting the device to GPU if there is or to cpu if there is no gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f'the device used is: {device}')





class myDataset(Dataset):
    def __init__(self, transform=None, target_transform=None, mode='Train'):
      if mode == 'Train':
        #self.path_to_cow = '/content/drive/MyDrive/Colab_Notebooks/Newdata/Train/Cow'
        #self.path_to_horse = '/content/drive/MyDrive/Colab_Notebooks/Newdata/Train/Horse'
        self.path_to_cow = '/content/drive/MyDrive/myNewData/Cow'
        self.path_to_horse = '/content/drive/MyDrive/myNewData/Horse'
      else:
        self.path_to_cow = '/content/drive/MyDrive/Colab_Notebooks/Newdata/Test/Shape1'
        self.path_to_horse = '/content/drive/MyDrive/Colab_Notebooks/Newdata/Test/Shape2'
      self.img_dir_cow = [os.path.join(self.path_to_cow, i) for i in os.listdir(self.path_to_cow)]
      self.img_dir_horse = [os.path.join(self.path_to_horse, i) for i in os.listdir(self.path_to_horse)]
      self.img_dir = self.img_dir_cow + self.img_dir_horse
      self.nbre_images = len(self.img_dir)
      self.labels = np.zeros(self.nbre_images)
      self.labels[0:len(self.img_dir_cow)] = 1
      self.transform = transform
      self.target_transform = target_transform

    def __len__(self):
        return self.nbre_images

    def __getitem__(self, idx):
        image = read_image(self.img_dir[idx])
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# building the classifier
class MyNeuralNetwork(nn.Module): 
    def __init__(self): 
      super(MyNeuralNetwork, self).__init__() 
      self.conv1 = nn.Conv2d(3, 8, (3, 3)) # we define 2d convolution with the input image and 8 filters of size 3 * 3 and 3 chanels (RGB) output will be 8 chanels of 62 * 62 pixel 
      self.norm1 = nn.BatchNorm2d(8) # normalization helps to make sure that each channel from the given input chanels has roughly the same scale and distribution
      self.pool = nn.MaxPool2d(2) # this filter takes the max value between two pixels, after this operation the size of image will be hlaf so 31 * 31 
      self.conv2 = nn.Conv2d(8, 4, (3, 3)) # 2d convolution with 4 filters of size 3 * 3 * 8, after this operation we will have 4 chanels of 29 * 29 pixel
      self.norm2 = nn.BatchNorm2d(4) 
      self.pool2 = nn.MaxPool2d(2) # the size will be half so 14
      self.Relu = nn.ReLU() # we use Relu to eliminate negative values: max(0,x)
      self.tanh = nn.Tanh() # we use tanh to map the image values to a range between -1 and 1
      self.FC1 = nn.Linear(14*14*4,500) # the input layer of 980 feature map and 500 neurons at the second one (first hidden layer) 
      self.FC2 = nn.Linear(500,200) # fully connection of 500 to 200 neurons (second hidden layer)
      self.FC3 = nn.Linear(200,120) # fully connection of 200 to 120 neurons (third hidden layer)
      self.FC4 = nn.Linear(120,2) # fully connection of 120 to output layer 2 neurons, the number of our classes

    def forward(self, x): # the forward method from nn.Module class, to impliment the operations defined on inint
        x = self.Relu(self.conv1(x)) # first convolution then we eliminate negetive values
        x = self.norm1(x) # normalization
        x = self.pool(x) # max pooling
        x = self.Relu(self.conv2(x)) # second convolution then activation
        x = self.norm2(x) # normalization
        x = self.pool(x) # maxpooling
        x = x.view(-1,14 * 14 * 4) # flattening the feature map to a vector 
        x = self.FC1(x) # fit the feature map to the input layer
        x = self.Relu(x) # activation
        x = self.FC2(x) # output from activation to second layer
        x = self.Relu(x) # activation 
        x = self.FC3(x) # output of activation to the third layer 
        x = self.Relu(x) 
        x = self.FC4(x)
        x = self.Relu(x) 
        x = self.tanh(x) # we map the obtained values to a range between -1 and 1
        return x





disc = MyNeuralNetwork().to(device) # instantiating the classifier 


#transform to resize the input images to 64 * 64 
transform = transforms.Compose([
            transforms.Resize((64,64))
            ])


batch_size = 6 # we take 6 images each time

# defining the training data and train loader
training_data = myDataset(transform=transform,mode='Train')
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,shuffle=True, num_workers=1) # shuffle is true means we take elements of batch randomly

# defining the testing data and test loader
test_data = myDataset(transform=transform,mode='Test')
test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,shuffle=True, num_workers=1)


loss_fn = nn.CrossEntropyLoss() # we take the cross entropy loss as our loss function, where it calculates the difference between the probability of a class and real value, to have the probability of the class we do the softmax for our input to convert our class values to probabilities then we compute the negative log-likelihood 
optimizer = torch.optim.SGD(disc.parameters(), lr= 1e-2) # we use SGD optimizer to update the weights after calculating the gradients


#defining the training function
def train(dataloader, optimizer, model, loss_fn):
    size = dataloader.dataset.__len__() # the size is our dataset length
    correct =0 # initialize correct counter

    for batch, (X, y) in enumerate(dataloader): 
        X, y = X.type(torch.float).to(device), y.type(torch.long).to(device) # sending images and labels to device 

        # forward pass
        pred = model(X) # we do the forward pass with giving the image to classifier directly without calling forward method
        loss = loss_fn(pred, y) # computing the loss between predicted and real
        # Backpropagation
        optimizer.zero_grad() # setting the gradients to zero
        loss.backward() # computing the gradients
        optimizer.step() # updating the weights 

        # printing the predicted and real labels each 8 batches
        if batch % 8 == 0:
          print(f'The predicted: {pred.argmax(1)} \n ------------------')
          print(f'The real: {y} \n --------------------------')
        
        correct += (pred.argmax(1) == y).type(torch.float).sum().item() # accumulating the correct predictions 
    correct /= size 
    print(f"Train Error: \n Accuracy: {(100*correct):>0.1f}%")

# defining the testing function
def test(dataloader, model):
  size = dataloader.dataset.__len__() # size of dataset 
  num_batches= len(dataloader) 
  correct= 0
  with torch.no_grad():
    for X, y in dataloader:
      X, y= X.type(torch.float).to(device), y.type(torch.long).to(device) # sending the images and labels to device
      pred = model(X) # predicting 
      correct += (pred.argmax(1) == y).type(torch.float).sum().item() # accumulating the correct 
    correct /= size
    # printing the accuracy at the end of test
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}% \n")


# training for 10 epochs then testing

for epoch in range(30):
    train(train_dataloader, optimizer, disc, loss_fn)
print(f'\n testing now: \n')
test(test_dataloader, disc)
