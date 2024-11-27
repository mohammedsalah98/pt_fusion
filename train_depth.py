import torch
from torch.utils.data import DataLoader
from utils.datasets import *
from utils.dataloaders import *
import warnings
from tqdm.auto import tqdm, trange
import matplotlib.pyplot as plt
from utils.helper_functions import *

warnings.filterwarnings("ignore")

model = att_UNetMultiModal_depth_MultiClass(in_channels_pca=10, in_channels_tsr=5, num_classes=6)
criterion = nn.L1Loss()
criterion_2 = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=5e-4)
weighing_factor = 0.5
num_epochs = 30
batch_size = 8

train_dataset = irtpvc_augmented(split='train', augment_factor=500, train_ratio=26, val_ratio=6, test_ratio=6)
val_dataset = irtpvc_augmented(split='val', augment_factor=500, train_ratio=26, val_ratio=6, test_ratio=6)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: collate_PCA_TSR_depth_fusion(x, transform=True))
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=lambda x: collate_PCA_TSR_depth_fusion(x, transform=True))

model = model.to(device)

train_losses = []
val_losses = []
test_losses = []

train_ious = []
val_ious = []
test_ious = []

train_depths = []
val_depths = []
test_depths = []

plt.ion()
fig_loss, ax_loss = plt.subplots()
line1_loss, = ax_loss.plot([], [], label="Training Loss")
line2_loss, = ax_loss.plot([], [], label="Validation Loss")
ax_loss.set_xlim(0, num_epochs)
ax_loss.set_ylim(0, 2)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training, Validation, and Testing Loss")

fig_iou, ax_iou = plt.subplots()
line1_iou, = ax_iou.plot([], [], label="Training IoU")
line2_iou, = ax_iou.plot([], [], label="Validation IoU")
ax_iou.set_xlim(0, num_epochs)
ax_iou.set_ylim(0, 1)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("IoU")

fig_depth, ax_depth = plt.subplots()
line1_depth, = ax_depth.plot([], [], label="Training Depth")
line2_depth, = ax_depth.plot([], [], label="Validation Depth")
ax_depth.set_xlim(0, num_epochs)
ax_depth.set_ylim(0, 1)
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Depth")

for epoch in trange(num_epochs, desc='training'): 
    # Train
    model.train()
    training_loss = 0.0
    training_depth = 0.0
    training_iou = 0.0
    num_batches = 0
    with tqdm(train_loader, unit='batch', desc=f'Epoch {epoch+1} [Training]') as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            pca, tsr, gt, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[-1].to(device)
            output, output_depth = model(pca, tsr)

            loss = (weighing_factor*criterion(output_depth, gt)) + criterion_2(output, mask)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            training_loss += loss.item()
            training_iou += calculate_iou_multiclass(output, mask)
            training_depth += criterion(output_depth, gt).item()
            num_batches += 1

            pfix = {"train_loss": training_loss / num_batches, "train_IoU": f"{(training_iou / num_batches):.5f}", \
                    "train_depth": training_depth / num_batches}
            tepoch.set_postfix(pfix)
    
    train_losses.append(training_loss / num_batches)
    train_ious.append(training_iou / num_batches)
    train_depths.append(training_depth / num_batches)
    
    # Validate
    model.eval()
    val_loss = 0.0
    val_iou = 0.0
    val_depth = 0.0
    num_val_batches = 0
    with torch.no_grad():
        with tqdm(val_loader, unit='batch', desc=f'Epoch {epoch+1} [Validation]') as vepoch:
            for batch_idx, batch in enumerate(vepoch):
                pca, tsr, gt, mask = batch[0].to(device), batch[1].to(device), batch[2].to(device), batch[-1].to(device)
                output, output_depth = model(pca, tsr)

                loss = (weighing_factor*criterion(output_depth, gt)) + criterion_2(output, mask)

                val_loss += loss.item()
                val_iou += calculate_iou_multiclass(output, mask)
                val_depth += criterion(output_depth, gt).item()
                num_val_batches += 1

                pfix = {"val_loss": val_loss / num_val_batches, "val_IoU": f"{(val_iou / num_val_batches):.5f}", \
                        "val_depth": val_depth / num_val_batches}
                vepoch.set_postfix(pfix)
    
    val_losses.append(val_loss / num_val_batches)
    val_ious.append(val_iou / num_val_batches)
    val_depths.append(val_depth / num_val_batches)

    line1_loss.set_xdata(range(1, len(train_losses) + 1))
    line1_loss.set_ydata(train_losses)
    
    line2_loss.set_xdata(range(1, len(val_losses) + 1))
    line2_loss.set_ydata(val_losses)
    
    ax_loss.set_ylim(0, max(max(train_losses), max(val_losses), max(test_losses)) + 0.1)
    fig_loss.canvas.draw()
    fig_loss.canvas.flush_events()

    line1_iou.set_xdata(range(1, len(train_ious) + 1))
    line1_iou.set_ydata(train_ious)
    
    line2_iou.set_xdata(range(1, len(val_ious) + 1))
    line2_iou.set_ydata(val_ious)
    
    ax_iou.set_ylim(0, max(max(train_ious), max(val_ious), max(test_ious)) + 0.1)
    fig_iou.canvas.draw()
    fig_iou.canvas.flush_events()

    line1_depth.set_xdata(range(1, len(train_depths) + 1))
    line1_depth.set_ydata(train_depths)
    
    line2_depth.set_xdata(range(1, len(val_depths) + 1))
    line2_depth.set_ydata(val_depths)
    
    ax_depth.set_ylim(0, max(max(train_depths), max(val_depths), max(test_depths)) + 0.1)
    fig_depth.canvas.draw()
    fig_depth.canvas.flush_events()

plt.ioff()
plt.show()