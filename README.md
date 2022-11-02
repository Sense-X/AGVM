# AGVM

**This repo is the official implementation of "[Large-batch Optimization for Dense Visual Predictions](https://arxiv.org/abs/2210.11078) (NeurIPS 2022)".** Since we adopted private frameworks (POD and LinkLink) to conduct the experiments previously, the results open-sourced with mmdetection will be slightly different from the results in our paper. The optimized version of DDP will be released in the future.

## Usage

### Installation

**Step 0.** Please refer to [mmdetection get started](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/get_started.md) for installation and dataset preparation.

**Step 1.** Install AGVM from source:

```bash
git clone https://github.com/Sense-X/AGVM.git
cd AGVM
make install
```

### Training

Please refer to this [doc](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/1_exist_data_model.md) for examples of training.

## Results
The box mAP of Faster R-CNN:
| Batch Size | 32   | 256                                                          | 512                                                          |
| ---------- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Baseline   | 37.1 | 36.7                                                         | 36.2                                                         |
| AGVM       | -    | 37.1 ([config](configs/object_detection/agvm/faster_rcnn_res50_bs256.py)) | 36.8 ([config](configs/object_detection/agvm/faster_rcnn_res50_bs512.py)) |

The seg mAP of Mask R-CNN:
| Batch Size | 32   | 256                                                          | 512                                                          |
| ---------- | ---- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Baseline   | 34.8 | 34.4                                                         | 33.9                                                         |
| AGVM       | -    | 35.0  ([config](configs/instance_segmentation/agvm/mask_rcnn_res50_bs256.py)) | 34.6 ([config](configs/instance_segmentation/agvm/mask_rcnn_res50_bs512.py)) |

## Citation

```
@article{xue2022large,
  title = {Large-batch Optimization for Dense Visual Predictions},
  author = {Zeyue Xue and Jianming Liang and Guanglu Song and Zhuofan Zong and Liang Chen and Yu Liu and Ping Luo},
  year = {2022},
  journal = {arXiv:2210.11078}
}
```

