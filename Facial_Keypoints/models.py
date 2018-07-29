## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        ## output size = (W-F)/S +1 = (224-4)/1 +1 = 221
        # the output Tensor for one image, will have the dimensions: (32, 220, 220)
        self.conv1 = nn.Conv2d(1, 32, 4)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # Batch Normalization layer
        self.bn_1 = nn.BatchNorm2d(32)

        # second conv layer: 32 inputs, 64 outputs, 5x5 conv
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 106
        # the output tensor will have dimensions: (64, 106, 106)
        # after another pool layer this becomes (64, 53, 53); 53  is rounded down
        self.conv2 = nn.Conv2d(32, 64, 5)

        self.bn_2 = nn.BatchNorm2d(64)

        # third conv layer: 64 inputs, 128 outputs, 7x7 conv
        ## output size = (W-F)/S +1 = (53-7)/1 +1 = 47
        # the output tensor will have dimensions: (64, 47, 47)
        # after another pool layer this becomes (64, 23, 23); 23.5  is rounded down
        self.conv3 = nn.Conv2d(64, 128, 7)

        self.bn_3 = nn.BatchNorm2d(128)

        # third conv layer: 128 inputs, 256 outputs, 7x7 conv
        ## output size = (W-F)/S +1 = (23-7)/1 +1 = 17
        # the output tensor will have dimensions: (256, 17, 17)
        # after another pool layer this becomes (256, 8, 8); 8.5  is rounded down
        self.conv4 = nn.Conv2d(128, 256, 7)

        self.bn_4 = nn.BatchNorm2d(256)

        # 64 outputs * the 12*12 filtered/pooled map size
        self.fc1 = nn.Linear(256 * 8 * 8, 1000)

        # dropout with p=0.4
        self.fc1_drop = nn.Dropout(p=0.4)

        # finally, create 10 output channels (for the 10 classes)
        self.fc2 = nn.Linear(1000, 800)

        # finally, create 10 output channels (for the 10 classes)
        self.fc3 = nn.Linear(800, 500)

        # finally, create 10 output channels (for the 10 classes)

        self.fc4 = nn.Linear(500, 136)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.bn_1(self.fc1_drop(self.pool(F.relu(self.conv1(x)))))
        #print("Shape of X after convolution 1: ", x.shape)
        x = self.bn_2(self.fc1_drop(self.pool(F.relu(self.conv2(x)))))
        #print("Shape of X after convolution 2: ", x.shape)
        x = self.bn_3(self.fc1_drop(self.pool(F.relu(self.conv3(x)))))
        x = self.bn_4(self.fc1_drop(self.pool(F.relu(self.conv4(x)))))

        # a modified x, having gone through all the layers of your model, should be returned
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        #print("Shape of X : ",x.shape)

        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
