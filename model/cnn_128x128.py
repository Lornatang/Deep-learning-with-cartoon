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


"""Use DCGAN to process 128 x 128 pixel images"""

import torch
import torch.nn as nn


# custom weights initialization called on netG and netD
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    self.main = nn.Sequential(
      # inputs is Z, going into a convolution
      nn.ConvTranspose2d(100, 128 * 16, 4, 1, 0, bias=False),
      nn.BatchNorm2d(128 * 16),
      nn.ReLU(True),
      # state size. (ngf*16) x 4 x 4
      nn.ConvTranspose2d(128 * 16, 128 * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128 * 8),
      nn.ReLU(True),
      # state size. (ngf*8) x 8 x 8
      nn.ConvTranspose2d(128 * 8, 128 * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128 * 4),
      nn.ReLU(True),
      # state size. (ngf*4) x 16 x 16
      nn.ConvTranspose2d(128 * 4, 128 * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128 * 2),
      nn.ReLU(True),
      # state size. (ngf*2) x 32 x 32
      nn.ConvTranspose2d(128 * 2, 128, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128),
      nn.ReLU(True),
      # state size. (ngf) x 64 x 64
      nn.ConvTranspose2d(128, 3, 4, 2, 1, bias=False),
      nn.Tanh()
      # state size. (nc) x 128 x 128
    )

  def forward(self, inputs):
    """ forward layer
    Args:
      inputs: input tensor data.
    Returns:
      forwarded data.
    """
    if torch.cuda.is_available():
      outputs = nn.parallel.data_parallel(self.main, inputs)
    else:
      outputs = self.main(inputs)
    return outputs


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.main = nn.Sequential(
      # inputs is (nc) x 128 x 128
      nn.Conv2d(3, 128, 4, 2, 1, bias=False),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf) x 64 x 64
      nn.Conv2d(128, 128 * 2, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128 * 2),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*2) x 32 x 32
      nn.Conv2d(128 * 2, 128 * 4, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128 * 4),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*4) x 16 x 16
      nn.Conv2d(128 * 4, 128 * 8, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128 * 8),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*8) x 8 x 8
      nn.Conv2d(128 * 8, 128 * 16, 4, 2, 1, bias=False),
      nn.BatchNorm2d(128 * 16),
      nn.LeakyReLU(0.2, inplace=True),
      # state size. (ndf*16) x 4 x 4
      nn.Conv2d(128 * 16, 1, 4, 1, 0, bias=False),
      nn.Sigmoid()
    )

  def forward(self, inputs):
    """ forward layer
    Args:
      inputs: input tensor data.
    Returns:
      forwarded data.
    """
    if torch.cuda.is_available():
      outputs = nn.parallel.data_parallel(self.main, inputs)
    else:
      outputs = self.main(inputs)
    return outputs