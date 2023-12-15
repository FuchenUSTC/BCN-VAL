# Action Recognition Evaluation Pipeline on BCN Feature


# Required Environment
- python 3.8.8
- Torch 1.7


# Video feature of ActivityNet validation set
- Download the video feature of ActivityNet validation set from https://pan.baidu.com/s/1CqYLlkA9mMNWSrsVhb4Jjg (pass code: r8vd) 
- Download the weights folder from the same URL and put it into this repo (`./weights/ckpt_epoch_32.pth`)
- The video feature of each video clip is extracted by BCN model (implemented by Caffe) with frame stride 8
- For more details about BCN model, please refer https://github.com/FuchenUSTC/BCN
- Please refer to `./dataset/anet/anet_val_npy.csv` for more details about the clip number of each video
- All the video number is `4,926` (5K), all the clip number is `2,066,253` (2M)


# Configuration
- Modify the `eva_root_path` in `./config/mlp-anet-infer.yml` as the path of the validation feature
- Modify the `clip_stride` in `./config/mlp-anet-infer.yml` for different number of sampling clips for evaluation
- When clip_stride is `-1`, the number of sampled clips equals to video number (5K)
- When clip_stride is `20`, the number of sampled clips is `103,312` (100K) 
- When clip_stride is `2`, the number of sampled clips is about `1M`


# Evaluation
If the environment and configuration have been set, please run
```
bash run_eval.sh
```
The Top-1, Top-3 and Top-5 classification accuracies are recorded on the log folder `./output/mlp-anet-infer`


# Model and weights



# Logs
Please refer to the logs `./output/mlp-anet-infer-5K/log.txt` (Top-1: 0.9275) and `./output/mlp-anet-infer-100K/log.txt` (Top-1: 0.9252) for more details about the performances on different number of clips (5K and 100K)
