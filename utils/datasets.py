import torch
from torch.utils.data import Dataset
import os
import cv2
from utils.testing_variables import *
import random

class irtpvc(Dataset):
    def __init__(self, split='train', repeat_factor=500, train_ratio=26, val_ratio=6, test_ratio=6):
        self.sequence_dir = args.data_folder
        self.gt_dir = args.data_folder + '/labels/'
        self.sequence_folders = sorted(os.listdir(self.sequence_dir + 'pca/'))
        self.repeat_factor = repeat_factor

        random.seed(42)
        total_folders = len(self.sequence_folders)
        assert train_ratio + val_ratio + test_ratio <= total_folders

        random.shuffle(self.sequence_folders)

        self.train_folders = self.sequence_folders[:train_ratio]
        self.val_folders = self.sequence_folders[train_ratio:train_ratio + val_ratio]
        self.test_folders = self.sequence_folders[train_ratio + val_ratio:]

        if split == 'train':
            self.selected_folders = self.train_folders
        elif split == 'val':
            self.selected_folders = self.val_folders
        elif split == 'test':
            self.selected_folders = self.test_folders
        else:
            raise ValueError("Invalid split. Choose from 'train', 'val', or 'test'.")

    def __len__(self):
        self.dataset_length = len(self.selected_folders)
        return self.dataset_length * self.repeat_factor

    def __getitem__(self, index):
        folder_index = int(index / self.repeat_factor)

        pca = torch.load(os.path.join(self.sequence_dir + 'pca/', self.selected_folders[folder_index]))
        tsr = torch.load(os.path.join(self.sequence_dir + 'tsr/', self.selected_folders[folder_index]))

        gt_mask_bgr = cv2.imread(self.gt_dir + self.selected_folders[folder_index][:-3].upper() + ".png")
        gt_mask = cv2.cvtColor(gt_mask_bgr, cv2.COLOR_BGR2GRAY) / 255.0

        return pca, tsr, torch.from_numpy(gt_mask).float()