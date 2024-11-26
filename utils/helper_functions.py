import random
import torchvision.transforms.functional as TF
from torchvision.transforms import RandomCrop
import torch
from utils.training_variables import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Spatial_Augmentation:
    def __init__(self, p=0.5, crop_size=(128, 128), shear_range=(-15, 15), scale_range=(0.8, 1.2)):
        self.p = p
        self.crop_size = crop_size
        self.shear_range = shear_range
        self.scale_range = scale_range

    def __call__(self, image, mask):
        if random.random() > self.p:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() > self.p:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < self.p:
            angle = random.uniform(-15, 15)
            scale = random.uniform(self.scale_range[0], self.scale_range[1])
            image = TF.affine(image, angle=angle, translate=[0, 0], scale=scale, shear=[0, 0], interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.affine(mask, angle=angle, translate=[0, 0], scale=scale, shear=[0, 0], interpolation=TF.InterpolationMode.NEAREST)

        crop = RandomCrop(self.crop_size)
        i, j, h, w = crop.get_params(image, output_size=self.crop_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        return image, mask

def calculate_iou(predictions, labels):
    predictions = (predictions > 0.5).to(torch.bool).to(device)
    labels = labels.to(torch.bool).to(device)
    intersection = (predictions & labels).sum().float()
    union = (predictions | labels).sum().float()
    iou = intersection / union if union != 0 else torch.tensor(1.0)
    return iou.item()

def calculate_iou_multiclass(predictions, labels, num_classes=6):
    predictions = torch.argmax(predictions, dim=1).to(device)
    iou_per_class = []
    for cls in range(num_classes):
        if (labels == cls).sum() == 0:
            continue
        pred_class = (predictions == cls).to(device)
        label_class = (labels == cls).to(device)
        intersection = (pred_class & label_class).sum().float()
        union = (pred_class | label_class).sum().float()
        iou = intersection / union if union != 0 else torch.tensor(1.0)
        iou_per_class.append(iou)

    iou_per_class = torch.tensor(iou_per_class, device=device)
    miou = torch.nanmean(iou_per_class).item()
    return miou

def calculate_recall(pred, gt, num_classes=6, include_background=True, device=None):
    if device is None:
        device = pred.device

    pred = torch.argmax(pred, dim=1).to(device)
    gt = gt.to(device)

    recall_per_class = []

    for cls in range(num_classes):
        if not include_background and cls == 0:
            continue

        if (gt == cls).sum() == 0:
            continue

        pred_mask = (pred == cls)
        gt_mask = (gt == cls)

        true_positive = (pred_mask & gt_mask).float().sum().to(device)
        false_negative = (~pred_mask & gt_mask).float().sum().to(device)

        if (true_positive + false_negative) == 0:
            recall = torch.tensor(0.0, device=device)
        else:
            recall = true_positive / (true_positive + false_negative)

        recall_per_class.append(recall)

    if len(recall_per_class) > 0:
        mean_recall = torch.mean(torch.stack(recall_per_class))
    else:
        mean_recall = torch.tensor(0.0, device=device)

    return mean_recall.item()

def calculate_precision(pred, gt, num_classes=6, include_background=True, device=None):
    if device is None:
        device = pred.device

    pred = torch.argmax(pred, dim=1).to(device)
    gt = gt.to(device)

    precision_per_class = []

    for cls in range(num_classes):
        if not include_background and cls == 0:
            continue

        if (gt == cls).sum() == 0:
            continue

        pred_mask = (pred == cls)
        gt_mask = (gt == cls)

        true_positive = (pred_mask & gt_mask).float().sum().to(device)
        false_positive = (pred_mask & ~gt_mask).float().sum().to(device)

        if (true_positive + false_positive) == 0:
            precision = torch.tensor(0.0, device=device)
        else:
            precision = true_positive / (true_positive + false_positive)

        precision_per_class.append(precision)

    if len(precision_per_class) > 0:
        mean_precision = torch.mean(torch.stack(precision_per_class))
    else:
        mean_precision = torch.tensor(0.0, device=device)

    return mean_precision.item()