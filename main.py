import torch
from torch import nn, optim
from torch.autograd.variable import Variable
import os
from model import Generater
import scipy.io

from PIL import Image

import numpy as np
#from utils import Logger



from torchvision import transforms, datasets

# def mnist_data():
#     compose = transforms.Compose(
#         [transforms.ToTensor(),
#          transforms.Normalize((.5, .5, .5), (.5, .5, .5))
#         ])
#     out_dir = './dataset'
#     return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)
#
#
# # Load data
# data = mnist_data()
# # Create loader with data, so that we can iterate over it
# data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
#num_batches = len(data_loader)
def load_data():
    # images_A = scipy.io.loadmat('./CT-MRI_data/CT/CT.mat')
    # images_A = images_A['data']
    path, dirs, files = os.walk("./dataset/MRI").__next__()
    file_count = len(files)
    images_B = []
    for i in range(file_count):
        image= scipy.io.loadmat('./dataset/MRI/'+files[i])
        image = image['data']

        image = np.hstack([image, np.zeros([511, 53])])
        image = np.hstack([np.zeros([511, 54]),image])
        image = np.append(image, np.zeros([1,512]), axis=0)
        image *= 255.0 / image.max()
        image = image.reshape(1,512,512)
        images_B.append(image)

    images_B=np.array(images_B)
    path, dirs, files = os.walk("./dataset/CT").__next__()
    file_count = len(files)
    images_A = []

    for i in range(file_count):
        image= scipy.io.loadmat('./dataset/CT/'+files[i])
        image = image['data']
        image = np.hstack([image, np.zeros([511, 53])])
        image = np.hstack([np.zeros([511, 54]), image])
        image = np.append(image, np.zeros([1, 512]), axis=0)
        image *= 255.0 / image.max()
        image = image.reshape(1, 512, 512)
        images_A.append(image)
    # images_A=np.swapaxes(images_A, 0, 2)
    # images_A=np.swapaxes(images_A, 1, 2)
    return images_A,images_B

data_A,data_B = load_data()
test_image = data_A[0:10]

data_loader_A = torch.utils.data.DataLoader(data_A, batch_size=1, shuffle=True)
data_loader_B = torch.utils.data.DataLoader(data_B, batch_size=1, shuffle=True)

class DiscriminatorNet_A(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self):
        super(DiscriminatorNet_A, self).__init__()

        ndf = 64
        self.hidden0 = nn.Sequential(
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden6 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden6 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.out(x)
        x = x.reshape(1,1) #4->1

        return x
    # def __init__(self):
    #     super(DiscriminatorNet_A, self).__init__()
    #     n_features = 206955
    #     n_out = 1
    #
    #     self.hidden0 = nn.Sequential(
    #         nn.Linear(n_features, 1024),
    #         nn.LeakyReLU(0.2),
    #         nn.Dropout(0.3)
    #     )
    #     self.hidden1 = nn.Sequential(
    #         nn.Linear(1024, 512),
    #         nn.LeakyReLU(0.2),
    #         nn.Dropout(0.3)
    #     )
    #     self.hidden2 = nn.Sequential(
    #         nn.Linear(512, 256),
    #         nn.LeakyReLU(0.2),
    #         nn.Dropout(0.3)
    #     )
    #     self.out = nn.Sequential(
    #         torch.nn.Linear(256, n_out),
    #         torch.nn.Sigmoid()
    #     )
    #
    # def forward(self, x):
    #     x = self.hidden0(x)
    #     x = self.hidden1(x)
    #     x = self.hidden2(x)
    #     x = self.out(x)
    #     return x


discriminator_A = DiscriminatorNet_A()

class DiscriminatorNet_B(torch.nn.Module):
    """
    A three hidden-layer discriminative neural network
    """

    def __init__(self):
        super(DiscriminatorNet_B, self).__init__()

        ndf = 64
        self.hidden0 = nn.Sequential(
            nn.Conv2d(1, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden1 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden3 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden4 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden6 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.hidden6 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.out = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        x = self.hidden5(x)
        x = self.hidden6(x)
        x = self.out(x)
        x = x.reshape(1, 1)
        return x


discriminator_B = DiscriminatorNet_B()

def images_to_vectors(images):
    #print(images.size(0))
    return images.view(images.size(0), 206955)

def vectors_to_images(vectors):

    return vectors.view(vectors.size(0), 1, 511, 405)


class GeneratorNet_A(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self):
        super(GeneratorNet_A, self).__init__()
        ngf = 64
        self.hidden0 = nn.Sequential(
            nn.Conv2d(1, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(ngf)
        )
        self.hidden1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(ngf*2)
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(ngf*4)
        )
        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)

        )
        self.hidden4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf , 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)

        x = self.out(x)
        return x


#generator_A = GeneratorNet_A()
generator_A = Generater()

class GeneratorNet_B(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """

    def __init__(self):
        super(GeneratorNet_B, self).__init__()
        ngf = 64
        self.hidden0 = nn.Sequential(
            nn.Conv2d(1, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(ngf)
        )
        self.hidden1 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(ngf * 2)
        )
        self.hidden2 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(ngf * 4)
        )
        self.hidden3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)

        )
        self.hidden4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        self.out = nn.Sequential(
            nn.ConvTranspose2d(ngf, 1, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)

        x = self.out(x)
        return x
    # def __init__(self):
    #     super(GeneratorNet_B, self).__init__()
    #     n_features = 206955
    #     n_out = 206955
    #
    #     self.hidden0 = nn.Sequential(
    #         nn.Linear(n_features, 256),
    #         nn.LeakyReLU(0.2)
    #     )
    #     self.hidden1 = nn.Sequential(
    #         nn.Linear(256, 512),
    #         nn.LeakyReLU(0.2)
    #     )
    #     self.hidden2 = nn.Sequential(
    #         nn.Linear(512, 1024),
    #         nn.LeakyReLU(0.2)
    #     )
    #
    #     self.out = nn.Sequential(
    #         nn.Linear(1024, n_out),
    #         nn.Tanh()
    #     )
    #
    # def forward(self, x):
    #     x = self.hidden0(x)
    #     x = self.hidden1(x)
    #     x = self.hidden2(x)
    #     x = self.out(x)
    #     return x


#generator_B = GeneratorNet_B()
generator_B = Generater()
def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n

d_optimizer_A = optim.Adam(discriminator_A.parameters(), lr=0.0002)
d_optimizer_B = optim.Adam(discriminator_B.parameters(), lr=0.0002)
g_optimizer_A = optim.Adam(generator_A.parameters(), lr=0.0002)
g_optimizer_B = optim.Adam(generator_B.parameters(), lr=0.0002)

loss = nn.BCELoss()
generator_loss = torch.nn.L1Loss()#(reduction = 'sum')
Mseloss = nn.MSELoss()#(reduction = 'sum')
def gradient(image):
    a = np.array([[1, 0, -1],[2,0,-2],[1,0,-1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv1.weight = nn.Parameter(torch.from_numpy(a).float().unsqueeze(0).unsqueeze(0))
    G_x = conv1(Variable(image)).data.view(1, 512, 512)
    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    conv2.weight = nn.Parameter(torch.from_numpy(b).float().unsqueeze(0).unsqueeze(0))
    G_y = conv2(Variable(image)).data.view(1, 512, 512)
    G = torch.sqrt(torch.pow(G_x, 2) + torch.pow(G_y, 2))
    return G
def gradient_calculate(image):
    image = image.data.numpy()

    image = image.reshape(512, 512)
    #image = Image.fromarray(image, 'L')
    gradient = np.gradient(image)[0]
    scipy.io.savemat("./result/" + str(i), {'image': gradient})

    return gradient

def gradient_loss(fake,real):
    fake_gra= gradient_calculate(fake)
    fake_gra = torch.from_numpy(fake_gra)
    real_gra = gradient_calculate(real)
    real_gra = torch.from_numpy(real_gra)
    return Mseloss(real_gra,fake_gra)

def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data


def train_discriminator_A(optimizer_A,optimazer_B, real_A,fake_A,real_B, fake_B,recycle_A,recycle_B):
    N = real_A.size()[0]
    # Reset gradients
    optimizer_A.zero_grad()
    #optimazer_B.zero.grad()

    # 1.1 Train on Real Data
    prediction_real_A = discriminator_A(real_A.detach())

    #prediction_real_B = discriminator_B(real_B)
    # Calculate error and backpropagate
    #print(prediction_real)
    error_real_A = loss(prediction_real_A, ones_target(N)) * 0.5
    #error_real_B = loss(prediction_real_A, ones_target(N))
    #error_real_B.backward()
    # 1.2 Train on Fake Data
    prediction_fake_A = discriminator_A(fake_A.detach())
    #prediction_fake_B = discriminator_B(fake_B)
    # Calculate error and backpropagate
    error_fake_A = loss(prediction_fake_A, zeros_target(N)) * 0.5
    #error_fake_B = loss(prediction_fake_B, zeros_target(N))
    error = error_fake_A+error_real_A
    error.backward()
    #error_fake_B.backward()
    # 1.3 Update weights with gradients
    optimizer_A.step()
    #optimazer_B.step()

    # Return error and predictions for real and fake inputs
    return (error_real_A + error_fake_A)*0.5, prediction_real_A, prediction_fake_A

def train_discriminator_B(optimizer_A,optimazer_B, real_A,fake_A,real_B, fake_B,recycle_A,recycle_B):
    N = real_A.size()[0]
    # Reset gradients
    #optimizer_A.zero_grad()
    optimazer_B.zero_grad()

    # 1.1 Train on Real Data
    #prediction_real_A = discriminator_A(real_A)
    prediction_real_B = discriminator_B(real_B.detach())
    # Calculate error and backpropagate
    # print(prediction_real)
    #error_real_A = loss(prediction_real_A, ones_target(N))
    error_real_B = loss(prediction_real_B, ones_target(N))*0.5
    #error_real_A.backward()

    # 1.2 Train on Fake Data
    #prediction_fake_A = discriminator_A(fake_A)
    prediction_fake_B = discriminator_B(fake_B.detach())
    # Calculate error and backpropagate
    #error_fake_A = loss(prediction_fake_A, zeros_target(N))
    error_fake_B = loss(prediction_fake_B, zeros_target(N))*0.5
    #error_fake_A.backward()
    error = error_fake_B + error_real_B
    error.backward()

    # 1.3 Update weights with gradients
    #optimizer_A.step()
    optimazer_B.step()

    # Return error and predictions for real and fake inputs
    return (error_real_B + error_fake_B)*0.5, prediction_real_B, prediction_fake_B  # not sure


def train_generator_A(optimizer_A,optimizer_B, real_A,fake_A,fake_B,recycle_A,recycle_B):
    N = real_A.size()[0]
    # Reset gradients

    optimizer_A.zero_grad()
    # Sample noise and generate fake data
    prediction_A = discriminator_A(fake_A.detach())
    error_A = loss(prediction_A, ones_target(N))*10
    # calculate generator_error and backpropogate

    error_gen_A = generator_loss(recycle_A, real_A)
    error_gen_B = generator_loss(recycle_B, real_B)

    #voxel-wise
    error_voxel =  generator_loss(real_A,fake_A)

    #GDL loss
    error_gradient = gradient_loss(fake_A,real_A)
    #print('error_gradient',error_gradient)
    error = error_gen_A+error_gen_B
    total_error = error+error_A+error_voxel+error_gradient
    #print('error', error_A,error_gen_A,error_gen_B,error_voxel,error_gradient)
    total_error.backward()

    # Update weights with gradients
    optimizer_A.step()
    #optimizer_B.step()
    # Return error
    return total_error  #not finished

def train_generator_B(optimizer_A,optimizer_B, fake_A,real_B,fake_B,recycle_A,recycle_B):
    N = real_A.size()[0]

    # Reset gradients
    #optimizer_A.zero_grad()
    optimizer_B.zero_grad()
    # Sample noise and generate fake data
    #prediction_A = discriminator_A(fake_A)
    prediction_B = discriminator_B(fake_B.detach())
    # Calculate error and backpropagate
    #error_A = loss(prediction_A, ones_target(N))
    error_B = loss(prediction_B, ones_target(N))*10
    #error_A.backward()
    #error_B.backward()

    # calculate generator_error and backpropogate

    error_gen_A = generator_loss(recycle_A.detach(), real_A.detach())
    error_gen_B = generator_loss(recycle_B.detach(), real_B.detach())
    error = error_gen_A+error_gen_B
    error_voxel = generator_loss(real_B, fake_B)
    # GDL loss
    error_gradient = gradient_loss(fake_B, real_B)
    total_error=error+error_B+error_voxel+error_gradient

    total_error.backward()

    # Update weights with gradients
    #optimizer_A.step(
    optimizer_B.step()
    # Return error
    return total_error  #not finished

num_test_samples = 16
test_noise = noise(num_test_samples)
i = 0
# Create logger instance
#logger = Logger(model_name='VGAN', data_name='MNIST')
# Total number of epochs to train
num_epochs = 50
for epoch in range(num_epochs):
    for n_batch, data in enumerate(zip(data_loader_A,data_loader_B),0):
        # 1. Train Discriminator
        real_A,real_B = data
        real_A = Variable(real_A.float())#real_data = Variable(images_to_vectors(real_batch))
        real_B = Variable(real_B.float())
        # Generate fake data and detach
        # (so gradients are not calculated for generator)
        fake_A = generator_A(real_B).detach()    #.detach()
        fake_B = generator_B(real_A).detach()
        image = fake_B.data.numpy()
        image = image.reshape(512, 512)
        scipy.io.savemat("./result_MRI/" + str(i), {'image': image})
        image = fake_A.data.numpy()
        image = image.reshape(512, 512)
        scipy.io.savemat("./result_CT/" + str(i), {'image': image})
        print('get', i)
        i = (i + 1)%251

        #recycle
        recycle_A = generator_A(fake_B).detach()
        recycle_B = generator_B(fake_A).detach()
        # Train D
        d_error_A, d_pred_real_A, d_pred_fake_A = \
            train_discriminator_A(d_optimizer_A, d_optimizer_B, real_A, fake_A, real_B, fake_B, recycle_A,
                                  recycle_B)  # fake data
        d_error_B, d_pred_real_B, d_pred_fake_B = \
            train_discriminator_B(d_optimizer_A,d_optimizer_B, real_A,fake_A,real_B, fake_B,recycle_A,recycle_B)#fake data

        # 2. Train Generator
        # Generate fake data
        #fake_data = generator(noise(N))
        # Train G
        g_error_A = train_generator_A(g_optimizer_A, g_optimizer_B, real_A,fake_A, fake_B,recycle_A,recycle_B)  # fake_data)
        g_error_B = train_generator_B(g_optimizer_A,g_optimizer_B, fake_A,real_B,fake_B,recycle_A,recycle_B)#fake_data)
        print(epoch, d_error_B, g_error_B)
        #Log batch error
        #logger.log(d_error, g_error, epoch, n_batch, num_batches)
        #Display Progress every few batches
        #if (n_batch) % 100 == 0:
            # test_images = real_data#vectors_to_images(generator(test_noise))
            # test_images = test_images.data
           # print(epoch, d_error_B, g_error_B)
            # logger.log_images(
            #     test_images, num_test_samples,
            #     epoch, n_batch, num_batches
            # );
            # Display status Logs
            # logger.display_status(
            #     epoch, num_epochs, n_batch, num_batches,
            #     d_error, g_error, d_pred_real, d_pred_fake
            # )

    
            # np.savetxt("./result/" + str(i) + ".txt", image)
