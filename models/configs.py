# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections
from enum import Enum

class ConfigsVit(Enum):
    ViT_B_16=1
    ViT_B_32=2
    ViT_L_16=3
    ViT_L_32=4
    ViT_H_14=5
    R50_ViT_B_16=6
    testing=7





