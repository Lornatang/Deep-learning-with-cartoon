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


""" DCGAN implments for noise generate cartoon image"""

from django.shortcuts import render
from rest_framework.views import APIView

from utils import dcgan
from utils import wgan
from utils import wgan_gp


def index(request):
  """Get the image based on the base64 encoding or url address
          and do the pencil style conversion
  Args:
      request: Post request in url.
      - image_code: 64-bit encoding of images.
      - url:        The URL of the image.
  Returns:
      Base64 bit encoding of the image.
  Notes:
      Later versions will not return an image's address,
      but instead a base64-bit encoded address
  """
  return render(request, "index.html")


class GAN(APIView):
  """ use dcgan generate animel sister
  """

  @staticmethod
  def get(request):
    """ Get the image based on the base64 encoding or url address
        and do the pencil style conversion
    Args:
        request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.
    Returns:
        Base64 bit encoding of the image.
    Notes:
        Later versions will not return an image's address,
        but instead a base64-bit encoded address
    """
    ret = {
      "status_code": 20000,
      "message": None,
      "dcgan_imgs": None,
      "wgan_imgs": None,
      "wgan_gp_imgs": None}
    return render(request, "gan.html", ret)

  @staticmethod
  def post(request):
    """ Get the image based on the base64 encoding or url address
        and do the pencil style conversion
    Args:
        request: Post request in url.
        - image_code: 64-bit encoding of images.
        - url:        The URL of the image.
    Returns:
        Base64 bit encoding of the image.
    Notes:
        Later versions will not return an image's address,
        but instead a base64-bit encoded address
    """
    # Get the url for the image
    dcgan_imgs = "./static/dcgan_imgs.png"
    wgan_imgs = "./static/wgan_imgs.png"
    wgan_gp_imgs = "./static/wgan_gp_imgs.png"
    dcgan.generate()
    wgan.generate()
    wgan_gp.generate()

    ret = {
      "status_code": 20000,
      "message": "OK",
      "dcgan_imgs": dcgan_imgs,
      "wgan_imgs": wgan_imgs,
      "wgan_gp_imgs": wgan_gp_imgs}
    return render(request, "gan.html", ret)
