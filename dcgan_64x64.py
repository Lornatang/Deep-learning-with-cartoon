# Copyright 2019 Lorna Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""Use DCGAN to process 64 x 64 pixel images"""

import argparse
import os
import random

import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

from model.cnn_64x64 import Discriminator
from model.cnn_64x64 import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./datasets', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--batch_size', type=int, default=64, help='inputs batch size')
parser.add_argument('--img_size', type=int, default=64, help='the height / width of the inputs image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr_D', type=float, default=0.00005, help='learning rate of D, default=0.00005')
parser.add_argument('--lr_G', type=float, default=0.0002, help='learning rate of G, default=0.00002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for adam. default=0.999')
parser.add_argument("--n_critic", type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument("--clip_value", type=float, default=0.01, help='lower and upper clip value for disc. weights')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='./checkpoints/dcgan_64x64_G.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='./checkpoints/dcgan_64x64_D.pth', help="path to netD (to continue training)")
parser.add_argument('--out_images', default='./dcgan_64x64_imgs', help='folder to output images')
parser.add_argument('--checkpoints_dir', default='./checkpoints', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

try:
  os.makedirs(opt.out_images)
except OSError:
  pass

if opt.manualSeed is None:
  opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
  print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if opt.cuda else "cpu")

fixed_noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)


def main():
  """ train model
  """
  try:
    os.makedirs(opt.checkpoints_dir)
  except OSError:
    pass
  ################################################
  #               load train dataset
  ################################################
  dataset = dset.ImageFolder(root=opt.dataroot,
                             transform=transforms.Compose([
                               transforms.Resize(opt.img_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
                             ]))

  assert dataset
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                           shuffle=True, num_workers=int(opt.workers))

  if torch.cuda.device_count() > 1:
    netG = torch.nn.DataParallel(Generator())
  else:
    netG = Generator()
  if os.path.exists(opt.netG):
    netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, loc: storage))

  if torch.cuda.device_count() > 1:
    netD = torch.nn.DataParallel(Discriminator())
  else:
    netD = Discriminator()
  if os.path.exists(opt.netD):
    netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, loc: storage))

  # set train mode
  netG.train()
  netG = netG.to(device)
  netD.train()
  netD = netD.to(device)
  print(netG)
  print(netD)

  ################################################
  #           Binary Cross Entropy
  ################################################
  criterion = nn.BCELoss()

  ################################################
  #            Use Adam optimizer
  ################################################
  optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_D, betas=(opt.beta1, opt.beta2))
  optimizerG = optim.Adam(netG.parameters(), lr=opt.lr_G, betas=(opt.beta1, opt.beta2))

  ################################################
  #               print args
  ################################################
  print("########################################")
  print(f"train dataset path: {opt.dataroot}")
  print(f"batch size: {opt.batch_size}")
  print(f"image size: {opt.img_size}")
  print(f"Epochs: {opt.n_epochs}")
  print(f"Noise size: {opt.nz}")
  print("########################################")
  print("Starting trainning!")
  for epoch in range(opt.n_epochs):
    for i, data in enumerate(dataloader):
      ##############################################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ##############################################
      # train with real
      netD.zero_grad()
      real_data = data[0].to(device)
      batch_size = real_data.size(0)

      # real data label is 1, fake data label is 0.
      real_label = torch.full((batch_size,), 1, device=device)
      fake_label = torch.full((batch_size,), 0, device=device)

      output = netD(real_data).view(-1)
      errD_real = criterion(output, real_label)
      errD_real.backward()

      # train with fake
      noise = torch.randn(batch_size, opt.nz, 1, 1, device=device)
      fake = netG(noise)
      output = netD(fake.detach()).view(-1)
      errD_fake = criterion(output, fake_label)
      errD_fake.backward()
      loss_D = errD_real + errD_fake
      optimizerD.step()

      ##############################################
      # (2) Update G network: maximize log(D(G(z)))
      ##############################################
      netG.zero_grad()
      output = netD(fake).view(-1)
      loss_G = criterion(output, real_label)
      loss_G.backward()
      optimizerG.step()
      print(f"Epoch->[{epoch + 1:03d}/{opt.n_epochs:03d}] "
            f"Progress->{i / len(dataloader) * 100:4.2f}% "
            f"Loss_D: {loss_D.item():.4f} "
            f"Loss_G: {loss_G.item():.4f} ", end="\r")

      if i % 50 == 0:
        vutils.save_image(real_data, f"{opt.out_images}/real_samples.png", normalize=True)
        with torch.no_grad():
          fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"{opt.out_images}/fake_samples_epoch_{epoch + 1}.png", normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), opt.netG)
    torch.save(netD.state_dict(), opt.netD)


if __name__ == '__main__':
  main()
