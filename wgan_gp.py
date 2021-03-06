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


import argparse
import os
import random

import torch.autograd as autograd
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.optim.adam import Adam

from model.cnn import Discriminator
from model.cnn import Generator

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='./datasets', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=8)
parser.add_argument('--batch_size', type=int, default=64, help='inputs batch size')
parser.add_argument('--img_size', type=int, default=64, help='the height / width of the inputs image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam. default=0.999')
parser.add_argument("--n_critic", type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='./checkpoints/wgan_gp_G.pth', help="path to netG (to continue training)")
parser.add_argument('--netD', default='./checkpoints/wgan_gp_D.pth', help="path to netD (to continue training)")
parser.add_argument('--out_images', default='./wgan_gp_imgs', help='folder to output images')
parser.add_argument('--checkpoints_dir', default='./checkpoints', help='folder to output model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

if opt.manualSeed is None:
  opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
  print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:0" if opt.cuda else "cpu")

fixed_noise = torch.randn(opt.batch_size, opt.nz, 1, 1, device=device)

try:
  os.makedirs(opt.out_images)
except OSError:
  pass


def calculate_gradient_penatly(netD, real_imgs, fake_imgs):
  """Calculates the gradient penalty loss for WGAN GP"""
  eta = torch.FloatTensor(real_imgs.size(0), 1, 1, 1).uniform_(0, 1).to(device)
  eta = eta.expand(real_imgs.size(0), real_imgs.size(1), real_imgs.size(2), real_imgs.size(3)).to(device)

  interpolated = eta * real_imgs + ((1 - eta) * fake_imgs)
  interpolated.to(device)

  # define it to calculate gradient
  interpolated = torch.autograd.Variable(interpolated, requires_grad=True)

  # calculate probaility of interpolated examples
  prob_interpolated = netD(interpolated)

  # calculate gradients of probabilities with respect to examples
  gradients = autograd.grad(
    outputs=prob_interpolated,
    inputs=interpolated,
    grad_outputs=torch.ones(prob_interpolated.size()).to(device),
    create_graph=True,
    retain_graph=True,
    only_inputs=True,
  )[0]

  gradients_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

  return gradients_penalty


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
  #            Use RMSprop optimizer
  ################################################
  optimizerD = Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
  optimizerG = Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))

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
      # get data
      real_imgs = data[0].to(device)
      batch_size = real_imgs.size(0)

      # Sample noise as generator input
      z = torch.randn(batch_size, opt.nz, 1, 1, device=device)

      ##############################################
      # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ##############################################

      optimizerD.zero_grad()

      # Generate a batch of images
      fake_imgs = netG(z)

      real_validity = netD(real_imgs)
      fake_validity = netD(fake_imgs)

      # Gradient penalty
      gradient_penalty = calculate_gradient_penatly(netD, real_imgs.data, fake_imgs.data)

      # Loss measures generator's ability to fool the discriminator
      loss_D = -torch.mean(real_validity) + torch.mean(fake_validity) + gradient_penalty * 10

      loss_D.backward()
      optimizerD.step()

      optimizerG.zero_grad()

      ##############################################
      # (2) Update G network: maximize log(D(G(z)))
      ##############################################
      if i % opt.n_critic == 0:
        # Generate a batch of images
        fake_imgs = netG(z)

        # Train on fake images
        loss_G = -torch.mean(netD(fake_imgs))

        loss_G.backward()
        optimizerG.step()

        print(f"Epoch->[{epoch + 1:03d}/{opt.n_epochs:03d}] "
              f"Progress->{i / len(dataloader) * 100:4.2f}% "
              f"Loss_D: {loss_D.item():.4f} "
              f"Loss_G: {loss_G.item():.4f} ", end="\r")

      if i % 50 == 0:
        vutils.save_image(real_imgs, f"{opt.out_images}/real_samples.png", normalize=True)
        with torch.no_grad():
          fake = netG(fixed_noise).detach().cpu()
        vutils.save_image(fake, f"{opt.out_images}/fake_samples_epoch_{epoch + 1}.png", normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), opt.netG)
    torch.save(netD.state_dict(), opt.netD)


if __name__ == '__main__':
  main()
