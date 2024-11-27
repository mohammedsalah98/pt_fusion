# PT-Fusion
## A Multi-Modal Attention Network for Segmentation and Depth Estimation of Subsurface Defects in Pulse Thermography

[![PT-Fusion:](https://github.com/mohammedsalah98/E_Calib/blob/master/video_thumbnail.png)](https://youtu.be/4giQn6rt-48)

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

## Code Structure and ECam_ACircles Dataset Outline:
![Alt text](https://github.com/mohammedsalah98/E_Calib/blob/master/ECam_ACircles.png)

## Supported platforms

Tested on the following platforms:

- Ubuntu 18.04 and 20.04 LTS

## Prerequisites
You need the model checkpoints and dataset to be in your working directory:
[Checkpoints]()
[Dataset]()

## Running the code
### Step 1: Create a conda environment
Create pt_fusion conda environment:
```
git clone https://github.com/mohammedsalah98/pt_fusion.git
cd pt_fusion
conda env create -f environment.yml
conda activate pt_fusion

### Step 2: Running the testing code:

- Multi-Class Segmentation:
```
cd pt_fusion
python test_multi.py --checkpoint /path/to/checkpoint --data_folder /path/to/dataset
```

- Multi-Class Segmentation:
```
cd pt_fusion
python test_depth.py --checkpoint /path/to/checkpoint --data_folder /path/to/dataset
```
#### Parameters:
- ``--checkpoint``:
- ``--data_folder``:

## Disclaimer
We will soon release a code for benchmarks against state-of-the-art models.