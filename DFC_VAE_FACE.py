from __future__ import print_function
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import pickle as pk
import sys
from glob import glob
import numpy as np
from PIL import Image
import torch.nn.functional as F
import torch.nn.init as init
import torchvision.datasets as dset

parser = argparse.ArgumentParser(description='PyTorch VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 2, 'pin_memory': True}# if args.cuda else {}





'''
train_loader = range(30)
test_loader = range(20)
totensor = transforms.ToTensor()
def load_batch(batch_idx, istrain):
    if istrain:
        template = './data/test/%s'
        path = './data/test/'
    else:
        template = './data/test/%s'
        path = './data/test/'
    dir=os.listdir(path)
    data = []
    count = 0
    i=0
    while count<64:
        idx=(batch_idx*64+i)%len(dir)
    #    print(template%dir[idx])
        img = Image.open(template%dir[idx])
       # img = img.resize((128,128))
        img = np.array(img)
        img = totensor(img)
        if len(img)==3 :
            data.append(img)
            count+=1
            i+=1
        else:
            i+=1
    return torch.stack(data, dim=0)
'''


trainroot='./data/train'
testroot='./data/test'
trainset = dset.ImageFolder(root=trainroot,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.RandomHorizontalFlip(p=0.5),
                               #transforms.Grayscale(num_output_channels=3),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               #AddGaussianNoise(0., 0.1),
                           ]))
# Create the dataloader
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=kwargs['num_workers'])
testset = dset.ImageFolder(root=testroot,
                           transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.RandomHorizontalFlip(p=0.5),
                               #transforms.Grayscale(num_output_channels=3),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               #AddGaussianNoise(0., 0.1),
                           ]))
# Create the dataloader
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size,
                                         shuffle=True, num_workers=kwargs['num_workers'])



class VAE(nn.Module):

    def __init__(self, latent_size=32, beta=1):
        super(VAE, self).__init__()

        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.encoder = nn.Sequential(
            self._conv(3, 32),
            self._conv(32, 32),
            self._conv(32, 64),
            self._conv(64, 64),
        )
        self.fc_mu = nn.Linear(256, latent_size)
        self.fc_var = nn.Linear(256, latent_size)

        # decoder
        self.decoder = nn.Sequential(
            self._deconv(64, 64),
            self._deconv(64, 32),
            self._deconv(32, 32, 1),
            self._deconv(32, 3),
            nn.Sigmoid()
        )
        self.fc_z = nn.Linear(latent_size, 256)

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(-1, 256)
        return self.fc_mu(x), self.fc_var(x)

    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        z = self.fc_z(z)
        z = z.view(-1, 64, 2, 2)
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=4, stride=2
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    # out_padding is used to ensure output size matches EXACTLY of conv2d;
    # it does not actually add zero-padding to output :)
    def _deconv(self, in_channels, out_channels, out_padding=0):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels, out_channels,
                kernel_size=4, stride=2, output_padding=out_padding
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, (x*0.5)+0.5, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #loss=recon_loss(recon_x,x)
        #print(recon_loss,kl_diverge)
        return (recon_loss+self.beta*kl_diverge)/x.shape[0] #(recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size

    def save_model(self, file_path, num_to_keep=1):
        utils.save(self, file_path, num_to_keep)

    def load_model(self, file_path):
        utils.restore(self, file_path)

    def load_last_model(self, dir_path):
        return utils.restore_latest(self, dir_path)

class DFCVAE(VAE):

    def __init__(self, latent_size=100, beta=1):
        super(DFCVAE, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.latent_size = latent_size
        self.beta = beta

        # encoder
        self.e1 = self._conv(3, 64)
        self.e2 = self._conv1(64, 64)
        self.e3 = self._conv(64, 128)
        self.e4 = self._conv1(128, 128)
        self.e5 = self._conv(128, 256)
        self.e6 = self._conv(256, 256)
        #self.e7 = self._conv(256, 512)
        #self.e8 = self._conv1(512, 512)
        self.fc_mu_1 = nn.Linear(256*4*4, 256*4)
        self.fc_var_1 = nn.Linear(256*4*4, 256*4)
        self.norm = nn.BatchNorm1d(256*4)
        self.fc_mu = nn.Linear(256*4, latent_size)
        self.fc_var = nn.Linear(256*4, latent_size)

        # decoder
        #self.d1 = self._upconv(512, 512)
        #self.d2 = self._upconv(512, 256)
        self.d3 = self._upconv(256, 256)
        self.d4 = self._upconv(256, 128)
        self.d5 = self._upconv(128, 128)
        self.d6 = self._upconv(128, 64)
        self.d7 = self._upconv(64, 64)
        self.d8 = self._upconv(64, 3)
        self.fc_z_1 = nn.Linear(latent_size, 256*4)
        self.fc_z = nn.Linear(256*4, 256*4*4)

    def encode(self, x):
        x = F.leaky_relu(self.e1(x))
        #print(x.shape)
        x = F.leaky_relu(self.e2(x))
        #print(x.shape)
        x = F.leaky_relu(self.e3(x))
        #print(x.shape)
        x = F.leaky_relu(self.e4(x))
        #print(x.shape)
        x = F.leaky_relu(self.e5(x))
        x = F.leaky_relu(self.e6(x))
        #x = F.leaky_relu(self.e7(x))
        #x = F.leaky_relu(self.e8(x))
        #print(x.shape)
        x = x.view(-1, 256*16)
        return self.fc_mu(self.norm(F.leaky_relu(self.fc_mu_1(x)))), self.fc_var(self.norm(F.leaky_relu(self.fc_var_1(x))))
        #return self.fc_mu(x), self.fc_var(x)
    def sample(self, mu, logvar):
        std = torch.exp(0.5*logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)
    
    def decode(self, z):
        z = self.fc_z(self.norm(F.leaky_relu(self.fc_z_1(z))))
        #z = self.fc_z_1(z)
        z = z.view(-1, 256, 4, 4)
        #print(z.shape)
        
        #z = F.leaky_relu(self.d1(z))
        #z = F.leaky_relu(self.d2(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d3(z))
        z = F.leaky_relu(self.d4(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d5(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d6(F.interpolate(z, scale_factor=2)))
        z = F.leaky_relu(self.d7(z))
        z = F.leaky_relu(self.d8(F.interpolate(z, scale_factor=2)))
        #print(z.shape)
        return torch.sigmoid(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.sample(mu, logvar)
        rx = self.decode(z)
        return rx, mu, logvar

    def _conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )
    def _conv1(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1, padding=1
            ),
            nn.BatchNorm2d(out_channels),
        )

    def _upconv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(
                in_channels, out_channels,
                kernel_size=3, stride=1
            ),
            nn.BatchNorm2d(out_channels),
        )




model = DFCVAE(latent_size=800)


if args.cuda:
    model.cuda()

optimizer = optim.Adam(model.parameters(), lr=1e-3)

def train(epoch):
    model.train()
    train_loss = 0
    batch_idx = 0
    for data,_ in train_loader:
        #data = load_batch(batch_idx, True)
        #data = Variable(data)
        data = data.to("cuda")
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss = train_loss + loss.item()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), (len(train_loader)*64),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

        batch_idx = batch_idx + 1
    #torchvision.utils.save_image(data.data, './imgs/Epoch_{}_data.jpg'.format(epoch), nrow=8, padding=2)
    torchvision.utils.save_image(recon_batch.data, './imgs/train_recon/Epoch_{}_recon.jpg'.format(epoch), nrow=8, padding=2)
    torchvision.utils.save_image(data.data*0.5+0.5, './imgs/train_target/Epoch_{}_recon_origin.jpg'.format(epoch), nrow=8, padding=2)
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / (len(train_loader)*args.batch_size)))
    return train_loss / (len(train_loader)*args.batch_size)

def test(epoch):
    model.eval()
    test_loss = 0
    batch_idx = 0
    for data,_ in test_loader:
        #data = load_batch(batch_idx, False)
        #data = Variable(data, volatile=True)
        if args.cuda:
            data = data.cuda()
        recon_batch, mu, logvar = model(data)
        test_loss = test_loss + model.loss(recon_batch, data, mu, logvar).item()

    #torchvision.utils.save_image(data.data, './imgs/Epoch_{}_data_test.jpg'.format(epoch), nrow=4, padding=2)
    torchvision.utils.save_image(recon_batch.data, './imgs/test/Epoch_{}_recon_test.jpg'.format(epoch), nrow=4, padding=2)

    test_loss = test_loss / (len(test_loader)*args.batch_size)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


def perform_latent_space_arithmatics(items): # input is list of tuples of 3 [(a1,b1,c1), (a2,b2,c2)]
    load_last_model()
    model.eval()
    data = [im for item in items for im in item]
    data = [totensor(i) for i in data]
    data = torch.stack(data, dim=0)
    data = Variable(data, volatile=True)
    if args.cuda:
        data = data.cuda()
    z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
    it = iter(z.split(1))
    z = zip(it, it, it)
    zs = []
    numsample = 11
    for i,j,k in z:
        for factor in np.linspace(0,1,numsample):
            zs.append((i-j)*factor+k)
    z = torch.cat(zs, 0)
    recon = model.decode(z)

    it1 = iter(data.split(1))
    it2 = [iter(recon.split(1))]*numsample
    result = zip(it1, it1, it1, *it2)
    result = [im for item in result for im in item]

    result = torch.cat(result, 0)
    torchvision.utils.save_image(result.data, '../imgs/vec_math.jpg', nrow=3+numsample, padding=2)


def latent_space_transition(items): # input is list of tuples of  (a,b)
    load_last_model()
    model.eval()
    data = [im for item in items for im in item[:-1]]
    data = [totensor(i) for i in data]
    data = torch.stack(data, dim=0)
    data = Variable(data, volatile=True)
    if args.cuda:
        data = data.cuda()
    z = model.get_latent_var(data.view(-1, model.nc, model.ndf, model.ngf))
    it = iter(z.split(1))
    z = zip(it, it)
    zs = []
    numsample = 11
    for i,j in z:
        for factor in np.linspace(0,1,numsample):
            zs.append(i+(j-i)*factor)
    z = torch.cat(zs, 0)
    recon = model.decode(z)

    it1 = iter(data.split(1))
    it2 = [iter(recon.split(1))]*numsample
    result = zip(it1, it1, *it2)
    result = [im for item in result for im in item]

    result = torch.cat(result, 0)
    torchvision.utils.save_image(result.data, '../imgs/trans.jpg', nrow=2+numsample, padding=2)


def rand_faces(num=5):
    load_last_model()
    model.eval()
    z = torch.randn(num*num, model.latent_variable_size)
    z = Variable(z, volatile=True)
    if args.cuda:
        z = z.cuda()
    recon = model.decode(z)
    torchvision.utils.save_image(recon.data, '../imgs/rand_faces.jpg', nrow=num, padding=2)

def load_last_model():
    models = glob('./models/Epoch_*.pth')
    model_ids = [(int(f.split('_')[0]), f) for f in models]
    start_epoch, last_cp = max(model_ids, key=lambda item:item[0])
    print('Last checkpoint: ', last_cp)
    model.load_state_dict(torch.load(last_cp))
    return start_epoch, last_cp

def resume_training():
    #start_epoch, _ = load_last_model()
    start_epoch=0
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(epoch)
        train_loss = train(epoch)
        if epoch%10 == 0:
            test(epoch)
        #torch.save(model.state_dict(), '../models/Epoch_{}_Train_loss_{:.4f}_Test_loss_{:.4f}.pth'.format(epoch, train_loss, test_loss))
        torch.save(model.state_dict(), './models/Epoch_{}.pth'.format(epoch, train_loss))

def last_model_to_cpu():
    _, last_cp = load_last_model()
    model.cpu()
    torch.save(model.state_dict(), './models/cpu_'+last_cp.split('/')[-1])

if __name__ == '__main__':
    resume_training()
    # last_model_to_cpu()
    # load_last_model()
    #rand_faces(8)
    # da = load_pickle(test_loader[0])
    # da = da[:120]
    # it = iter(da)
    # l = zip(it, it, it)
    # # latent_space_transition(l)
    # perform_latent_space_arithmatics(l)