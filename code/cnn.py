import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision import transforms

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

