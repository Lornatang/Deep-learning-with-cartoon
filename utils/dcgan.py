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


"""Use DCGAN to process 64x64 pixel images"""

import torch
import torchvision.utils as vutils
import model.cnn_64x64
import model.cnn_128x128


def generate_64x64():
  """ random generate fake image.
  """
  if torch.cuda.device_count() > 1:
    netG = torch.nn.DataParallel(model.cnn_64x64.Generator())
  else:
    netG = model.cnn_64x64.Generator()

  netG.load_state_dict(torch.load("./checkpoints/dcgan_64x64_G.pth", map_location=lambda storage, loc: storage))
  netG.eval()

  with torch.no_grad():
    z = torch.randn(16, 100, 1, 1)
    fake = netG(z).detach().cpu()
    vutils.save_image(fake, f"./static/dcgan_64x64.png", normalize=True, nrow=4)


def generate_128x128():
  """ random generate fake image.
  """
  if torch.cuda.device_count() > 1:
    netG = torch.nn.DataParallel(model.cnn_128x128.Generator())
  else:
    netG = model.cnn_128x128.Generator()

  netG.load_state_dict(torch.load("./checkpoints/dcgan_128x128_G.pth", map_location=lambda storage, loc: storage))
  netG.eval()

  with torch.no_grad():
    z = torch.randn(16, 100, 1, 1)
    fake = netG(z).detach().cpu()
    vutils.save_image(fake, f"./static/dcgan_128x128.png", normalize=True, nrow=4)
