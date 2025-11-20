# CameraFlow

## Traning In Process!!! No TODOs Anymore!!!

 - 看效果如何要不要 vllm 部署 UnifiedReward-2.0-qwen-7b


## Code

源代码文件。核心文件存放位置：

 - Code/setup.py 注释内容依次为 NPU 环境配置所需步骤，未注释内容为原仓库依赖。
 - Code/scripts/train_recam.py 主训练函数，为方便调试增加 denoise 和 sample 的缓存功能
 - Code/flow_grpo/rewards.py 奖励函数 
 - Code/config/grpo.py 重点关注 my_recam_8npu()，配置超参与奖励函数
 - Code/scripts/recam 因 Pipeline 类为自行实现，存放相关代码


单结点 8 卡训练
```
bash scripts/single_node/grpo.sh
```

4 结点 32 卡训练，各节点依次启动
```
bash scripts/multi_node/recam/main.sh
```


## Evaluation

处理 WebVID 开源数据集可以直接执行生成，结构目录如下
```
.WebVID
├── cameras
│   └── camera_extrinsics.json
│
├── scripts
│   ├── batch-download-from-csv.py
│   ├── generate_metadata-csv.py
│   ├── select-name-more-than-25
│   └── results_2M_val.csv
│
├── videos
│   ├── video01.mp4
│   ├── video02.mp4
│   └── ...
│
└── metadata.csv
```


已生成 10 个相机轨迹的视频，结构目录如下
```
.results
├── cam_type1
│   ├── video01.mp4
│   ├── video02.mp4
│   └── ...
│
├── cam_type2
│   ├── video01.mp4
│   ├── video02.mp4
│   └── ...
│
├── ...
│
└── cam_type10
    ├── video01.mp4
    ├── video02.mp4
    └── ...
```

执行 CameraFlow/evaluator.py 调用已有 clip 文件计算 -T -F -V 指标、计算FID FVD (-V 要求10个相机轨迹的视频)
```
 conda activate wan
 CUDA_VISIBLE_DEVICES=1 python evaluator.py
```

执行 GIM/demo.py 计算 source 和 out video 逐帧之间可信的像素对应数。
```
 conda env  create -f environment.yaml
 conda activate gim
 CUDA_VISIBLE_DEVICES=5 python demo.py \
  --model gim_dkm  \
  --recam_folder .results/cam_type1/   \
  --original_folder .WebVID/videos/    \
  --num_frames 80
```
所需模型如下：
```
color150.mat
encoder_epoch_20.pth
gim_dkm_100h.ckpt
gim_roma_100h.ckpt
object150_info.csv
resnet50-0676ba61.pth
```
执行 VBench/evaluate.py 计算视频评价指标。
```
conda activate wan
pip install transformers==4.33.2

CUDA_VISIBLE_DEVICES=1 \
python evaluate.py --videos_path  .results/cam_type1  \
  --dimension    aesthetic_quality   imaging_quality    temporal_flickering   motion_smoothness  subject_consistency  background_consistency   dynamic_degree \
  --mode=custom_input \
  --load_ckpt_from_local True
```
所需模型在 pretrained 文件夹下，使用完毕恢复 transformers 库版本
```
pip install transformers==4.46.2
```

执行 CamAccuracy/glomap.py 实现调用 Glomap 前处理 + 指标计算后处理一条龙服务

TODO: 可能出现3分钟都处理不出来的怪东西；结果保存在 csv 文件中，但是字符串而非 float
```
 conda activate wan
 CUDA_VISIBLE_DEVICES=1 python glomap.py
```

## MultiCamDataset Preprocess

发布者并未给出 caption 的获得方式。gen_metadata_csv.py 是一个基于 Qwen2.5-VL-3B-Instruct-AWQ 推理的脚本，需要强制在单卡4090上运行，否则会报错 tensor 位置不同。

该版本并非稳定版，首先需要手动处理脚本检测出的空行，最后再将所有 !!!! 感叹号按照同一场景下的其他视频的 caption 复制粘贴过来。
同时，人工检查发现幻觉较为严重，例如所有视频均为单人在原地跳舞，但却会被大量识别出两个人、摄像头靠近主体会被解释为人正跑向相机、摄像头向上俯视会被解释为摔倒等等。

```
conda activate wlh-py
cd /data/wlh/ReCamMaster/MultiCamVideo-Dataset
CUDA_VISIBLE_DEVICES=0  python gen_metadata_csv.py
```

数据集文件夹架构如下：（在训练前还要先逐一用 VAE 处理提取为同路径同名 cam01.pth 文件）
```
MultiCamVideo-Dataset
├── train
│   ├── f18_aperture10
│   │   ├── scene1    # one dynamic scene
│   │   │   ├── videos
│   │   │   │   ├── cam01.mp4    # synchronized 81-frame videos at 1280x1280 resolution
│   │   │   │   ├── cam02.mp4
│   │   │   │   ├── ...
│   │   │   │   └── cam10.mp4
│   │   │   └── cameras
│   │   │       └── camera_extrinsics.json    # 81-frame camera extrinsics of the 10 cameras 
│   │   ├── ...
│   │   └── scene3400
│   │       └── ...
│   ├── f24_aperture5
│   │   └── ...
│   ├── f35_aperture2.4
│   │   └── ...
│   └── f50_aperture2.4
│       └── ...
│── val
│    └── 10basic_trajectories
│        ├── videos
│        │   ├── cam01.mp4
│        │   ├── cam02.mp4
│        │   ├── ...
│        │   └── cam10.mp4
│        └── cameras
│            └── camera_extrinsics.json    # 10 different trajectories for validation
└── metadata.csv
```

metadata.csv 的示例内容如下：
```
|                 filename                   |                            text                            |
|:------------------------------------------:|:----------------------------------------------------------:|
|  f18_aperture10/scene10/videos/cam04.mp4   |  The video depicts a person standing on a wooden deck....  |
|  f18_aperture10/scene../videos/...         |  ... ...                                                   |
```
