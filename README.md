# AGVM

**This repo is the official implementation of "Large-batch Optimization for Dense Visual Predictions".** Since we adopted private frameworks (POD and LinkLink) to conduct the experiments previously, the results open-sourced with mmdetection will be slightly different from the results in our paper. 


## Training





## Results
We report the box mAP of Faster R-CNN and seg mAP of Mask R-CNN.
| Batch Size        | 32   | 256  | 512  |
|-------------------|------|------|------|
| Faster R-CNN Base | 37.1 | 36.7 | 36.2 |
| Faster R-CNN AGVM | -    | 37.1 | 36.8 |
| Mask R-CNN Base   | 34.8 | 34.4 | 33.9 |
| Mask R-CNN AGVM   | -    | 35.0 | 34.6 |
