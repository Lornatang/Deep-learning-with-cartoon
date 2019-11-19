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

"""
====================================WARNING====================================
Do not delete this file unless you know how to refactor it!
====================================WARNING====================================
"""

from django.conf.urls import url
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from django.urls import path

from api.views_cats import GAN
from api.views_cats import index

# noinspection PyInterpreter
urlpatterns = [
  path('', index),
  url(r'^api/gan/$', GAN.as_view(), name="GAN"),
  url('index/', index, name="index"),
]
urlpatterns += staticfiles_urlpatterns()
