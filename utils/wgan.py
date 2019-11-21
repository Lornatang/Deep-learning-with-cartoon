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


"""Use WGAN to process 64x64 pixel images"""

import torch
import torchvision.utils as vutils
import model.cnn
import model.cnn_128x128


def generate():
  """ random generate fake image.
  """
  if torch.cuda.device_count() > 1:
    netG = torch.nn.DataParallel(model.cnn.Generator())
  else:
    netG = model.cnn.Generator()

  netG.load_state_dict(torch.load("./checkpoints/wgan_G.pth", map_location=lambda storage, loc: storage))
  netG.eval()

  with torch.no_grad():
    z = torch.randn(64, 100, 1, 1)
    fake = netG(z).detach().cpu()
    vutils.save_image(fake, f"./static/wgan_imgs.png", normalize=True)
