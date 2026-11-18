#!/bin/bash
# Reference implementations the tests compare against (MAE and Prithvi-EO-2.0 / TerraTorch).
cd "$(dirname "$0")/ref"
curl -sL https://raw.githubusercontent.com/facebookresearch/mae/main/util/pos_embed.py -o mae_pos_embed.py
curl -sL https://raw.githubusercontent.com/facebookresearch/mae/main/models_mae.py -o mae_models_mae.py
curl -sL https://raw.githubusercontent.com/torchgeo/terratorch/main/terratorch/models/backbones/prithvi_mae.py -o prithvi_mae.py
ls -l
