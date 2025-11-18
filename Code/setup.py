from setuptools import setup, find_packages
#
# rm -rf /home/ma-user/anaconda3/envs/recam/*
#
# scp -r -P 2222 \
#    /home/ma-user/anaconda3/envs/PyTorch-2.1.0/* \
#    ma-user@ma-job-2f9da41e-4f0c-407a-8b1c-2ba5f4073500-worker-1.ma-job-2f9da41e-4f0c-407a-8b1c-2ba5f4073500:/home/ma-user/anaconda3/envs/recam
#
# import torch
# import torch_npu
# torch.__version__
# torch_npu.npu.is_available()
 

# conda create -n flowgrpo  --clone  PyTorch-2.1.0
# pip install transformers accelerate  numpy==1.26.4 pandas scikit-learn  scikit-image  albumentations opencv-python pillow tqdm  pydantic requests matplotlib
# pip install deepspeed peft aiohttp fastapi uvicorn absl-py sentencepiece einops
# pip install ml-collections  diffusers==0.33.1 wandb==0.18.7 urllib3==1.26.18 modelscope ftfy
# pip install imageio[ffmpeg]

setup(
    name="flow-grpo",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio",
        "transformers==4.40.0",
        "accelerate==1.4.0",
        "diffusers==0.33.1", 
        
        "numpy==1.26.4",
        "pandas==2.2.3",
        "scipy==1.15.2",
        "scikit-learn==1.6.1",
        "scikit-image==0.25.2",
        
        "albumentations==1.4.10",  
        "opencv-python==4.11.0.86",
        "pillow==10.4.0",
        
        "tqdm==4.67.1",
        "wandb==0.18.7",
        "pydantic==2.10.6",  
        "requests==2.32.3",
        "matplotlib==3.10.0",
        
        # "flash-attn==2.7.4.post1",
        "deepspeed==0.16.4",  
        "peft==0.10.0",       
        "bitsandbytes==0.45.3",
        
        "aiohttp==3.11.13",
        "fastapi==0.115.11", 
        "uvicorn==0.34.0",
        
        "huggingface-hub==0.29.1",  
        "datasets==3.3.2",
        "tokenizers==0.19.1",
        
        "einops==0.8.1",
        "nvidia-ml-py==12.570.86",
        "xformers",
        "absl-py",
        "ml_collections",
        "sentencepiece",
        "openai",
    ],
    extras_require={
        "dev": [
            "ipython==8.34.0",
            "black==24.2.0",
            "pytest==8.2.0"
        ]
    }
)
