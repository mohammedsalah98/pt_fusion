# PT-Fusion
## A Multi-Modal Attention Network for Segmentation and Depth Estimation of Subsurface Defects in Pulse Thermography

[![PT-Fusion:](https://github.com/mohammedsalah98/pt_fusion/blob/main/thumbnail.png)](https://drive.google.com/file/d/17Gw1JwUtIPZwAZ9cj_FlGoW12sXYYHZk/view?usp=sharing)

#
You can find the PDF of the paper [here]().
If you use this code in an academic context please cite this publication:

```bibtex
@ARTICLE{pt_fusion,
  author={Salah, Mohammed and Werghi, Naoufel and Svetinovic, Davor and Abdulrahman, Yusra},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={PT-Fusion: A Multi-Modal Attention Network for Segmentation and Depth Estimation of Subsurface Defects in Pulse Thermography}, 
  year={2024},
  volume={},
  pages={},
  doi={}}
```

## Code Structure and IRT-PVC Dataset Outline:
![Alt text](https://github.com/mohammedsalah98/pt_fusion/blob/main/dataset.png)

## Supported platforms

Tested on the following platforms:

- Ubuntu 18.04 and 20.04 LTS

## Prerequisites
You need the model checkpoints and dataset to be in your working directory:

[Checkpoints](https://drive.google.com/drive/folders/1i5LGqa5_GO9XCohDdU-1M8rXYSEteuyP?usp=sharing)
[Testing Dataset](https://drive.google.com/drive/folders/1i5LGqa5_GO9XCohDdU-1M8rXYSEteuyP?usp=sharing)

## Running the code
### Step 1: Create a conda environment
Create pt_fusion conda environment:
```
git clone https://github.com/mohammedsalah98/pt_fusion.git
cd pt_fusion
conda env create -f environment.yml
conda activate pt_fusion
```

### Step 2: Running the code:

#### Multi-Class Segmentation:
```
cd pt_fusion
python test_multi.py --checkpoint /path/to/checkpoint --data_folder /path/to/dataset
```

#### Segmentation & Depth Estimation:
```
python test_depth.py --checkpoint /path/to/checkpoint --data_folder /path/to/dataset
```

#### Parameters:
- ``--checkpoint``: Path to downloaded checkpoint
- ``--data_folder``: Path to dataset

#### Training:
[Training Dataset](https://drive.google.com/drive/folders/1i5LGqa5_GO9XCohDdU-1M8rXYSEteuyP?usp=sharing)
```
python train_segmentation.py --data_folder /path/to/dataset
python train_depth.py --data_folder /path/to/dataset
```

For example, if you place the checkpoint in the checkpoints folder and the dataset in pt_fusion directory, the command should look like this:
```
python test_depth.py --checkpoint checkpoints/attention_fusionUnet_depth.pth --data_folder dataset/
```

## Disclaimer
We will soon release a code for benchmarks against state-of-the-art models.