import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

# 实际上是一个基础配置模板，它提供了所有训练配置的默认值，所以每个函数都会先执行它
def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = f'logs/geneval/sd3.5-M'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def clipscore_sd3():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.02
    config.sample.global_std = True
    config.sample.same_latent = True
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/clipscore/sd3.5-M'
    config.reward_fn = {
        "clipscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_fast():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 24
    config.sample.mini_num_image_per_prompt = 9
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M-fast'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

# 执行 grpo.py 的脚本是你！！！
def general_ocr_sd3_8gpu():
    gpu_number = 8
    config = compressibility()


    # sd3.5 medium
    config.pretrained.model = "/home/ma-user/modelarts/user-job-dir/wlh/Model/stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    # 计算逻辑：总样本数16 / (GPU数4 × 每GPU batch size 8 / 每提示图像数16) = 8个批次
    
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    

    # 此处指定奖励函数和权重，使用OCR识别准确率作为奖励
    config.reward_fn = { "ocr": 1.0}
    

    # 此处指定提示函数类型
    # 决定输入什么内容给模型进行训练
    # 可用的提示函数类型：
    # "general_ocr"：从OCR数据集加载文本提示，config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    # "geneval"：从GenEval数据集加载结构化提示，config.dataset = os.path.join(os.getcwd(), "dataset/geneval")
    # "imagenet_animals"：从ImageNet动物类别生成提示
    # "simple_animals"：从简单动物列表生成提示
    config.prompt_fn = "general_ocr"
    # 此处配置数据集路径，要与上面的提示函数类型一一对应
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_4gpu():
    gpu_number=4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3_1gpu():
    gpu_number = 1
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(8/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


# 你他妈学习一下T2V是怎么部署的
def general_ocr_wan2_1():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # config.pretrained.model = "hf_cache/Wan2.1-T2V-14B-Diffusers"
    config.pretrained.model = "hf_cache/Wan2.1-T2V-1.3B-Diffusers"
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale=4.5
    config.run_name = "wan_flow_grpo"
    
    config.height = 240
    config.width = 416
    config.frames = 33
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 4 # 12
    config.sample.num_batches_per_epoch = 2
    config.sample.sample_time_per_prompt = 1
    config.sample.test_batch_size = 2

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt // 2 if (config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt) > 1 else 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.004
    config.train.learning_rate = 1e-4
    config.train.clip_range=1e-3
    # kl reward
    # KL reward and KL loss are two ways to incorporate KL divergence. KL reward adds KL to the reward, while KL loss, introduced by GRPO, directly adds KL loss to the policy loss. We support both methods, but KL loss is recommended as the preferred option.
    config.sample.kl_reward = 0
    # We also support using SFT data in RL training for supervised learning to prevent quality drop, but this option was unused
    config.train.sft=0.0
    config.train.sft_batch_size=3
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std=False
    config.train.ema=True
    config.mixed_precision = "bf16"
    config.diffusion_loss = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 30
    config.save_dir = f'logs/video_ocr/{config.run_name}'
    config.resume_from = None
    config.reward_fn = {
        "video_ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def my_recam_8npu():
    config = compressibility()
    config.run_name = "recam_flow_grpo"
    config.save_dir = f'logs/{config.run_name}'
    config.mixed_precision = "bf16"
    config.pretrained.wan_model = "/home/ma-user/modelarts/user-job-dir/wlh/Model/Wan-AI/Wan2.1-T2V-1.3B"
    config.pretrained.recam_model = "/home/ma-user/modelarts/user-job-dir/wlh/Model/recammaster/step20000-Kling.ckpt"
    config.allow_tf32 = True
    config.use_lora = True
    config.train.lora_path = None
    config.height = 480
    config.width = 832
    config.num_frames = 81
    config.reward_fn = {
        # "my_reward": 0.3,
        "clip_score": 0.3,
        "pick_score": 0.1,
        "optical_reward": 0.2,
        "gt_reward": 0.4,
    }
    config.dataset = "/home/ma-user/modelarts/user-job-dir/wlh/Data/MultiCamVideo-Dataset"
    
    config.sample.train_batch_size = 2  # sample batch size
    config.sample.test_batch_size = 1   # eval batch size

    config.sample.k = 4  
    config.sample.num_steps = 30        # sampling denoising steps
    config.sample.eval_num_steps = 50   # SDE inference steps
    config.sample.guidance_scale=5.0    # CFG
    config.sample.num_batches_per_epoch = 2     # 每个epoch里只采样2次batch

    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 30


    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2 if config.sample.num_batches_per_epoch > 1 else 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.5

    # Gradient checkpointing: saves memory by recomputing activations during backward pass
    # Enable this to reduce memory usage by 50-70% at the cost of ~30% slower training
    # Optional: offload checkpoints to CPU for even more memory savings (slower but saves more memory)
    config.train.gradient_checkpointing = True
    config.train.gradient_checkpointing_offload = False


    # kl loss
    config.train.beta = 0.004
    config.train.learning_rate = 1e-4
    config.train.clip_range=1e-3
    

    # Whether to use the std of all samples or the current group's.
    config.sample.global_std=False
    config.train.ema=True
    config.diffusion_loss = True
    

    config.per_prompt_stat_tracking = True
    return config


def get_config(name):
    return globals()[name]()
