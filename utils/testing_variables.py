import torch
from utils.models import *
from utils.helper_functions import *
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Model Testing")
    parser.add_argument('--checkpoint', type=str, required=False, help='Path to the model checkpoint file')
    parser.add_argument('--data_folder', type=str, required=True, help='Path to the dataset folder')
    return parser.parse_args()
args = parse_args()