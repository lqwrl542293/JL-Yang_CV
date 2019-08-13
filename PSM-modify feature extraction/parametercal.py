from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import math
from dataloader import listflowfile as lt
from dataloader import SecenFlowLoader as DA
from models import *
from tensorboardX import SummaryWriter



model = basic(192)
model = nn.DataParallel(model)
model.cuda()
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))