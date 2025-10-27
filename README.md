# CameraFlow

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

执行 CameraFlow/glomap.py 实现调用 Glomap 前处理 + 指标计算后处理一条龙服务
TODO: 可能出现3分钟都处理不出来的怪东西；结果保存在 csv 文件中，但是字符串而非 float
```
 conda activate wan
 CUDA_VISIBLE_DEVICES=1 python glomap.py
```
