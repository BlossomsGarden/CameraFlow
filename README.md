# CameraFlow

## Train

源代码文件。核心文件存放位置：

 - /setup.py 注释内容依次为 NPU 环境配置所需步骤，未注释内容为原仓库依赖。
 - /scripts/train_recam.py 主训练函数，为方便调试增加 denoise 和 sample 的缓存功能
 - /flow_grpo/rewards.py 奖励函数 
 - /flow_grpo/server_path 奖励函数模型启动处，查看下文 Server 部分
 - /config/grpo.py 重点关注 my_recam_8npu()，配置超参与奖励函数
 - /scripts/recam/ 因 Pipeline 类为自行实现，存放相关代码


单结点 8 卡训练
```
bash scripts/single_node/grpo.sh
```

4 结点 32 卡训练，各节点依次启动
```
bash scripts/multi_node/recam/main.sh
```

启动训练时数据集文件夹架构如下，只是用 f24 文件夹。
dataset_tools/ 中给出供训练用的精简版 csv 文件，谨慎使用，具体查看 Dataset Preprocess部分。
```
MultiCamVideo-Dataset
├── train
│   └── f24_aperture5
│   │   ├── scene1    # one dynamic scene
│   │   │   ├── videos
│   │   │   │   ├── cam01.mp4                 # synchronized 81-frame videos at 1280x1280 resolution
│   │   │   │   ├── cam01.mp4.tensor.pth      # pre-encoded input
│   │   │   │   ├── cam02.mp4
│   │   │   │   ├── cam02.mp4.tensor.pth
│   │   │   │   ├── ...
│   │   │   │   ├── cam10.mp4
│   │   │   │   └── cam10.mp4.tensor.pth
│   │   │   └── cameras
│   │   │       └── camera_extrinsics.json    # 81-frame camera extrinsics of the 10 cameras 
│   │   ├── ...
│   │   └── scene3400
│   │       └── ...
├── metadata-f24-test.csv
└── metadata-f24-train.csv
```


## Evaluate

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

执行 eval/CamAccuracy/evaluator.py 调用已有 clip 文件计算 -T -F -V 指标、计算FID FVD (-V 要求10个相机轨迹的视频)
```
 conda activate wan
 CUDA_VISIBLE_DEVICES=1 python evaluator.py
```

执行 eval/GIM/demo.py 计算 source 和 out video 逐帧之间可信的像素对应数。
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
执行 eval/VBench/evaluate.py 计算视频评价指标。
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

执行 eval/CamAccuracy/glomap.py 实现调用 Glomap 前处理 + 指标计算后处理一条龙服务

TODO: 可能出现3分钟都处理不出来的怪东西；结果保存在 csv 文件中，但是字符串而非 float
```
 conda activate wan
 CUDA_VISIBLE_DEVICES=1 python glomap.py
```

## Dataset Preprocess

下载完毕后初始数据集文件夹架构如下：（在训练前还要先逐一用 VAE 处理提取为同路径同名 cam01.pth 文件）
```
MultiCamVideo-Dataset
├── train
│   ├── f18_aperture10 （场景非常暗，不推荐使用）
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

但发布者并未给出 metadata.csv，这里分别给出基于 Qwen3-VL-8B-Instruct 和 Qwen.5-VL-3B-Instruct-AWQ 的多进程处理示例和处理结果。

 - dataset_tools/create_csv_script.py 用于批量生成并行处理脚本的文件，其任务是 4 个结点 32 个进程同时处理 f24
 - dataset_tools/run_worker*.sh 用于在4个结点上启动上个步骤中生成的处理脚本，每张显卡一个
 - dataset_tools/merge_csv.py 用于生成完毕后将 32 个 csv 文件合并成一个
 - dataset_tools/post-handle-csv.py 用于处理空行和非 UTF-8 字符。

PS: 该版本并非稳定版，首先需要检验是否存在 !!!! 感叹号，若有则要手动将同一场景下的其他视频的 caption 粘过来。

PS: 谨慎使用 qwen2.5vl 的 csv 文件，会变得不幸。（所有视频都是单人在原地跳舞，但却被大量识别出两个人、摄像头靠近主体会被解释为人正跑向相机、摄像头向上俯视会被解释为摔倒等）f18的场景特别黑。


处理完毕后 metadata.csv 的示例内容如下：
```
|                 filename                   |                            text                            |
|:------------------------------------------:|:----------------------------------------------------------:|
|  f18_aperture10/scene10/videos/cam04.mp4   |  The video depicts a person standing on a wooden deck....  |
|  f18_aperture10/scene../videos/...         |  ... ...                                                   |
```

## Server

由于 unifiedreward、da3、flowgrpo 要求不同版本 transformers，本仓库采用额外部署接口开放请求端口的方式执行奖励函数计算。
在 flow_grpo/server_path 下的 server 文件是增加了 load/unload model 接口等魔改过的。
为提高评估速度，每个结点 8 张显卡上部署 8 个模型，依次开放 8 个端口。计算奖励时进程会自取 LOCAL_RANK 请求所在显卡上部署的模型对应的接口：
```
cd flow_grpo\server_patch
conda activate da3
bash start_da3_servers.sh
conda activate unifiedreward
bash start_qwen3vl_servers.sh
```


执行 server_tools/unified_reward/ 进行单视频质量评估，从 4 个维度出发，给出 1-5 分之间的两位小数分数：
```
unified_reward
├── api_request.py              # 基于 infer_2.0_qwen3vl_8b.py 封装的 c/s 接口
├── api_server.py
├── face-1.mp4                  # 供测试的 group_size=4 视频组，模型需给出有差异的得分
├── face-2.mp4
├── face-3.mp4
├── face-4.mp4
├── infer_2.0_qwen2.5vl_7b.py   # 调用 CodeGoat24/UnifiedReward-2.0-qwen-7b 模型
└── infer_2.0_qwen3vl_8b.py     # 调用 CodeGoat24/UnifiedReward-2.0-qwen3vl-8b 模型
```

执行 server_tools/depth_anything_3 进行相机外参准确性估计，输入单视频及其条件外参，给出 TransErr 和 RotErr：
```
depth_anything_3
├── docs
│   └── API.md                  # 官方 api 文档
├── src
│   ├── depth_anything_3        # 官方源码，因 910B 无法安装 pycolmap 改过部分逻辑
│   │   ├── ...
│   │   ├── api.py              # 核心调用
│   │   └── ...
│   ├── api_request.py          # 基于 infer.py 封装的 c/s 接口
│   ├── api_server.py
│   ├── cam01.mp4               # 从 Dataset 中取出的视频和外参，测试指标范围和性能
│   ├── cam02.mp4
│   ├── camera_extrinsics.json
│   └── infer.py                # 一个调用demo
└── requirements.txt            # 910B 安装指南
```

执行 server_tools/qwen3 进行看图说话，启用 flash_attention_2 能显著节省显存。

