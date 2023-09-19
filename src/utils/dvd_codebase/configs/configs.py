"""
Copyright (c) Facebook, Inc. and its affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import logging
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--debug', default=0, type=int, help='')

# Data /projects/hochmeister/CATER-videos/features/per_video'
#parser.add_argument('--fea-dir', default='/scratch/hochmeister/CATER-videos/features/monet_pretrained_on_clevr/per_video', type=str, help='Image feature files (.pkl)')
#parser.add_argument('--data-dir', default='/scratch/hochmeister/DVDData/small_subset/', type=str,help='Path to training feature files')
parser.add_argument('--output-dir', default='/scratch/abdessaied/projects/olvit/msc2022_hochmeister/checkpoints/avsd_code_testing', type=str,help='output path of model and params')
parser.add_argument('--num-workers', default=20, type=int, help='')
parser.add_argument('--device', default='0', type=str, help='')

# Training 
parser.add_argument('--num-epochs', '-e', default=15, type=int,help='Number of epochs')
#parser.add_argument('--batch-size', '-b', default=85, type=int,help='Batch size in training')
# others
parser.add_argument('--verbose', '-v', default=0, type=int,help='verbose level')

args, unknown = parser.parse_known_args()

print(args)

# Presetting
if args.verbose >= 1:
    logging.basicConfig(level=logging.DEBUG, 
        format='%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s')
else:
    logging.basicConfig(level=logging.INFO, 
        format='%(asctime)s %(levelname)s: %(message)s')
