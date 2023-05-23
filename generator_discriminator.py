from torch.nn.modules.batchnorm import BatchNorm2d
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
import torchvision.utils as vutils
from torchvision.transforms import ToPILImage


# setting the device to GPU if there is or to cpu if there is no gpu
device = "cuda" if torch.cuda.is_available() else "cpu"
print (f'the device used is: {device}')


# Size of noise
nz = 100


# This class is to load images for training and testing, it inherets from Dataset class so we don't de redifine evrything from scratch 
class myDataset(Dataset):
    def __init__(self, transform=None,mode='Train'):
      if mode == 'Train': #cheking if we are training or testing 
        self.path_to_cow = '/content/drive/MyDrive/Colab_Notebooks/Newdata/Train/Cow' # path to training cow images on my drive 
        self.path_to_horse = '/content/drive/MyDrive/Colab_Notebooks/Newdata/Train/Horse' # path to horse images
      else:
        self.path_to_cow = '/content/drive/MyDrive/Colab_Notebooks/Newdata/Test/Shape1' # path to cows for training
        self.path_to_horse = '/content/drive/MyDrive/Colab_Notebooks/Newdata/Test/Shape2' # path to horses for training 
      self.img_dir_cow = [os.path.join(self.path_to_cow, i) for i in os.listdir(self.path_to_cow)] # storing the name of cow images in a list
      self.img_dir_horse = [os.path.join(self.path_to_horse, i) for i in os.listdir(self.path_to_horse)] # storing the name of horse images in a list
      self.img_dir = self.img_dir_horse + self.img_dir_cow  # concatinating the two lists 
      self.nbre_images = len(self.img_dir) # returning the number of images 
      self.labels = np.zeros(self.nbre_images) # labling all images as zeros
      self.labels[0:len(self.img_dir_cow)] = 1 # we lable the cows to 1
      self.transform = transform # we spicify the transform to the value in input when constructing the this class
    # this method is abstract on Dataset class, we define it now to return the lenght of our dataset which is the number of images
    def __len__(self):
        return self.nbre_images
    # this method is abstract on Dataset class, we define it now a to return an image by its index 
    def __getitem__(self, idx):
        image = read_image(self.img_dir[idx]) # we take the image by its arrangement on image directory, read_image() is a method from Dataset class
        label = self.labels[idx] # the corresponding label (0 for horse, 1 for cow) is the one stored on label list on that index 
        if self.transform:
            image = self.transform(image) # if the transform option is set we do the transformation 
        return image, label # output of this function is a tuple of the image and its label 
    
# The discriminator is our classifier that classifies the images into two classes fake or real, it inherits from nn.Module class
class Discriminator(nn.Module):
    def __init__(self): # the default values for our class when it is constructed
        super(Discriminator, self).__init__() # we create a tomporary object of the super class nn.Module so we can access its atributes and methods
        self.main = nn.Sequential( # we define the sequnce of actions that will be done for the input image 
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # conv2d is a 2d convlution where the input is a 2 dimentional image, 3 layer (RGB) and it will be convolved with 64 filters of 4 by 4 pixels by 3, andt stride is 2 means the filter moves by two pixles on the image and 1 is the padding means we add 1 pixel for padding the size of image after this layer is 32*32
            nn.LeakyReLU(0.2, inplace=False), # Relu() sets negative values to 0 but LeakyRelu() lets some of them with a given slope 0.2, this to prevent desactivating a huge amount of neurons. For inplace means whether the activation replaces the current tensor or create a new one, putting it False will consume memory 
            nn.Conv2d(64, 128, 4, 2, 1, bias=False), # 64 input chanels from first layer, 128 filters so 128 output chanels, filter size 4 * 4 * 64, stride is 2 and padding is 1 the size of image after this layer is 16 * 16
            nn.BatchNorm2d(128), # normalization helps to make sure that each channel from the given input chanels has roughly the same scale and distribution
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),# 128 input chanels, 256 output chanels, 4 * 4 * 128 filter size and stride is 2 with 1 pixel padding the size of image fter this layer will be 8 * 8
            nn.BatchNorm2d(256), 
            nn.LeakyReLU(0.2, inplace=True), 
            nn.Conv2d(256, 512, 4, 2, 1, bias=False), # 256 input chanels, 512 output with 4 * 4 * 256 filter size, stride 2 and 1 pixel padding the size of image after this layer will be 4 * 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False), # 512 input and 1 chanel output chanel, filter of 4 * 4 * 512 and stride = 1 with no padding  the size of image after this layer is 1 pixel
            nn.Sigmoid() # we use this activation function to have values in range of 0 and 1 and it corresponds to the two classes if 1 cow else 0 horse.
        )
    def forward(self, input): # the forward method is the responsible for forward propagation and it is abstruct in nn.Module class we have to define it now and it returns the output of our image as input to the sequence defined in init
        return self.main(input)


#The generator is the responsible of generating 64 * 64 pixel images starting from a noise of size 1*10
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # we do convolutional layers, the output after each layer is calculated by this formula: (input height - 1) * stride - 2 * padding + kernel size 
            nn.ConvTranspose2d( nz, 512, 4, 1, 0, bias=False), # conv transposed takes an input of random noise vector (100) and generates a feature map of 4 * 4 and 512 chanels
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),         
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False), # this convolution layer will generate a feature map of 8 * 8 and 256 chanels 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False), # the feature map will be 15 * 15 and 128 chanels
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False), # the feature map will be 32 * 32 and 64 chanels
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),        
            nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False), # the feature map will be 64 * 64 and 3 chanels and it is as our training images 
            nn.Tanh() # we use tanh to map the imgage values to a range between -1 and 1 
        )

    def forward(self, input): # the forward method is the responsible for forward propagation and it is abstruct in nn.Module class we have to define it now and it returns the generated image starting from noise input tensor 
        return self.main(input)



# Now after we finished defining our classes: Discriminator, Ginirator, myDataset we will instantiate them

disc = Discriminator().to(device) # we instantiate the discriminator object
gen = Generator().to(device) # we instantiate our generator

#transform is used to manipulate the input data so we can have more feature maps from one image. 
transform = transforms.Compose([
            transforms.Resize((64,64)), # in our case we are using it just to resize the input image like the generated ones from 256 * 256 to 64 * 64 
            ])


batch_size = 1 # we can not take all the images at onece and perform the caculations because of memory, so we take batches 

#defining test and train data: 
training_data = myDataset(transform=transform) # we instantiate our training dataset by letting the mode as default 'Train'
train_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size,shuffle=True, num_workers=4) # we instantiate our training dataloader using the training dataset and by setting btach size, shuffle to true wich means we take the elements of batch rundomly each time and number of workers is when using gpu how match parallel operations to run simultaneously


#defining the loss function:
loss_fn = nn.BCELoss() # BCELoss function is defined as: -(y​⋅log(x)​+(1−y)⋅log(1−x​))
# we select our optimizer which is the responsible of updating the weights
opt_disc = torch.optim.SGD(disc.parameters(), lr= 1e-3) # discriminator optimizer
opt_gen = torch.optim.SGD(gen.parameters(), lr= 1e-3) # generator optimizer


#defining training fucntion:
def train(dataloader, opt_disc,opt_gen ,disc ,gen , loss_fn):
  size= dataloader.dataset.__len__() # the size of our dataset
  for batch , (real,y) in enumerate(dataloader,0): # we loop througth the dataloader where real are the images and y the labels
    print(f'batch number: {train}')
    #training the discriminator

    real, y= real.type(torch.float).to(device) , y.type(torch.long).to(device)  # we send real and y to the device
    noise =  torch.rand(1, nz, 1, 1, device=device) # generating random noise of size 100
    fake = gen(noise.to(device)) # generating the image starting from random noise

    

    disc_real = disc(real) # we give our real image to the discriminator to classify it as real or fak
    loss_real = loss_fn(disc_real, torch.ones_like(disc_real)) # we have: -(y​⋅log(x)​+(1−y​)⋅log(1−x​)) we want to maximize log(D(x)), we give x as the output from disc and y 1 so we eleminate the second term 
    disc_fake = disc(fake) # feeding fake image to discriminator
    loss_fake = loss_fn(disc_fake, torch.zeros_like(disc_fake)) # we have:  -(y​⋅log(x)​+(1−y​)⋅log(1−x​)) ,we need to minimize the log(D(G(z))) so we can maximize log(1-D(G(z))). We give x as output from disc and y as zeros to eliminate the first term so we can maximze the second term 
    loss = (loss_real + loss_fake) / 2 # calculating the average loss

 
    disc.zero_grad() # setting the gradient to zero
    loss.backward(retain_graph = True ) # calculating the gradient with respect to all the parameters
    opt_disc.step() # using the previous calculated gradients we do our optimization for discriminator 
    
    #train the generator: 
    output = disc(fake) # we feed our fake to the discriminator 
    lossG = loss_fn(output, torch.ones_like(output)) # we have:  -(y​⋅log(x)​+(1−y​)⋅log(1−x​)), we give x as output from disc and y as zeros to eliminate the first term so we can maximze the second term 
    gen.zero_grad() # setting the gradient to zero
    lossG.backward() # calculating the gradients with respect to all the parameters
    opt_gen.step() # optimizing the generator using the calculated gradients
    # we print each 10 batches an image generated by the generator 
    if batch % 10 == 9: 
      %matplotlib inline
      noise = torch.rand(1, nz, 1, 1, device=device)
      fake = gen(noise.to(device))
      fake = fake.cpu().detach().squeeze().permute(1,2,0).numpy() 
      plt.imshow(fake) 
      plt.show()
    # we save the generated images to drive using ToPILImage library 
    noise = torch.rand(1, nz, 1, 1, device=device) 
    fake_np = gen(noise.to(device)).detach().cpu().numpy()
    fake_tensor = torch.from_numpy(fake_np)
    fake_pil = ToPILImage()(vutils.make_grid(fake_tensor, padding=2, normalize=True))
    fake_pil.save(f'/content/drive/MyDrive/generated_horses/fake_{j}.png')


# Now we set 500 epochs so we train our model 500 times 

j=0
for i in range(500):
  j+=1
  print(f'we are on the epoch: {i}')
  train(train_dataloader, opt_disc,opt_gen ,disc ,gen , loss_fn)
