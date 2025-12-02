# GRPO中k=4时，4个相同条件副本的分布位置（基于你的8卡NPU配置）

## 你的配置

根据 `config/grpo.py` 中的 `my_recam_8npu()` 函数：

```python
config.sample.train_batch_size = 2      # 每个NPU的batch size
config.sample.k = 4                     # group_size (每个条件的副本数)
config.sample.num_batches_per_epoch = 2 # 每个epoch采样2个batch
# 使用8卡NPU训练，所以 num_replicas = 8
```

**注意**：代码中使用的是 `config.sample.num_image_per_prompt`，你需要确保它等于 `config.sample.k`：
```python
config.sample.num_image_per_prompt = config.sample.k  # 应该设置为4
```

## 关键计算

基于你的配置：
- `num_replicas = 8` (8卡NPU)
- `batch_size = 2` (每个NPU)
- `k = 4` (group_size，每个条件的副本数)
- `num_batches_per_epoch = 2`

**每个batch的计算**：
- `total_samples = num_replicas * batch_size = 8 * 2 = 16` (每个batch的总样本数)
- `m = total_samples // k = 16 // 4 = 4` (每个batch有4个唯一样本，每个重复4次)

**每个epoch的计算**：
- 每个epoch总样本数 = `num_batches_per_epoch * total_samples = 2 * 16 = 32`
- 每个epoch唯一样本数 = `num_batches_per_epoch * m = 2 * 4 = 8`

## 采样器的工作机制

`DistributedKRepeatSampler` 的 `__iter__()` 是一个**无限循环生成器**：

```python
def __iter__(self):
    while True:  # 无限循环
        # 每次循环生成 total_samples = 16 个索引
        # 其中每个唯一样本重复k=4次
        yield per_card_samples[self.rank]  # 返回当前NPU的2个索引
```

**关键点**：
- 每次调用 `next(train_iter)` 时，生成器会执行一次 `yield`，然后**继续循环**
- 由于 `set_epoch` 在每次循环中都被调用（第1312行），每次 `next()` 都会生成**新的索引序列**

## 每个batch的epoch值不同

看第1312行：
```python
train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
```

这意味着：
- Batch 0: `epoch = epoch * 2 + 0`
- Batch 1: `epoch = epoch * 2 + 1`

**每个batch的epoch值都不同**，所以每个batch生成的索引序列都不同！

## 4个副本的详细分布

### Batch 0 的分布

当 `set_epoch(epoch * 2 + 0)` 时：

1. **采样器生成16个索引**：
   - 随机选择4个唯一样本（假设索引为 [100, 200, 300, 400]）
   - 每个唯一样本重复4次：`[100, 100, 100, 100, 200, 200, 200, 200, 300, 300, 300, 300, 400, 400, 400, 400]`
   - 打乱顺序（假设）：`[100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400, 100, 200, 300, 400]`

2. **分配到8个NPU**：
   - NPU0: 索引 [100, 200] → 生成2个视频 (videos.shape = [2, 3, 81, 480, 832])
   - NPU1: 索引 [300, 400] → 生成2个视频
   - NPU2: 索引 [100, 200] → 生成2个视频
   - NPU3: 索引 [300, 400] → 生成2个视频
   - NPU4: 索引 [100, 200] → 生成2个视频
   - NPU5: 索引 [300, 400] → 生成2个视频
   - NPU6: 索引 [100, 200] → 生成2个视频
   - NPU7: 索引 [300, 400] → 生成2个视频

3. **关键观察**：
   - 索引100的4个副本分布在：NPU0, NPU2, NPU4, NPU6（每个NPU1个）
   - 索引200的4个副本分布在：NPU0, NPU2, NPU4, NPU6（每个NPU1个）
   - 索引300的4个副本分布在：NPU1, NPU3, NPU5, NPU7（每个NPU1个）
   - 索引400的4个副本分布在：NPU1, NPU3, NPU5, NPU7（每个NPU1个）

**所以，同一个条件的4个副本分布在不同的NPU上！**

### Batch 1 的分布

当 `set_epoch(epoch * 2 + 1)` 时：

1. **采样器生成新的16个索引**：
   - 随机选择4个新的唯一样本（假设索引为 [500, 600, 700, 800]）
   - 每个唯一样本重复4次并打乱

2. **分配到8个NPU**：
   - 类似Batch 0的分配方式
   - 索引500、600、700、800各有4个副本，分布在不同的NPU上

## Samples的累积和合并

### Step 1: 每个batch的结果被append（第1424行）

```python
# 每个batch循环中
samples.append({
    "prompts": prompts,
    "prompt_embeds": prompt_embeds,
    "source_latents": all_latents[:, :, 21:, ...],
    "target_cameras": target_camera,
    "timesteps": timesteps,
    "latents": latent_trajectory[:, :-1],
    "next_latents": latent_trajectory[:, 1:],
    "log_probs": log_probs,
    "rewards": rewards,
})
```

**此时**：
- `samples` 是一个列表，包含2个字典（对应2个batch）
- 每个字典的tensor形状是 `[batch_size, ...]` = `[2, ...]`（单个NPU的batch）

### Step 2: 所有batch的samples被合并（第1484行）

```python
samples = {
    k: torch.cat([s[k] for s in samples], dim=0)  # 在batch维度上拼接
    for k in samples[0].keys()
}
```

**此时（单个NPU）**：
- `samples["latents"].shape = [4, ...]` (2个batch * 2个样本/batch = 4个样本)
- `samples["log_probs"].shape = [4, num_steps]`
- `samples["target_cameras"].shape = [4, 21, 12]`

**关键**：此时，同一个条件的4个副本**仍然分布在不同NPU上**，还没有聚集！

### Step 3: Gather所有NPU的samples（第1555行）

```python
gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
```

**此时（所有NPU合并后）**：
- `gathered_rewards["avg"].shape = [32, num_train_timesteps]` (8个NPU * 4个样本/NPU = 32个样本)
- 其中，索引100的4个副本现在聚集在一起（在gathered_rewards中的某个位置）
- 索引200的4个副本也聚集在一起
- 索引300、400、500、600、700、800的4个副本各自聚集

## 完整的流程示例

假设 Epoch 0：

### Batch 0 (epoch = 0 * 2 + 0 = 0)

**采样器生成**：
- 唯一样本：索引 [100, 200, 300, 400]
- 每个重复4次，打乱后分配到8个NPU

**每个NPU生成**：
- NPU0: 索引 [100, 200] → videos.shape = [2, 3, 81, 480, 832]
- NPU1: 索引 [300, 400] → videos.shape = [2, 3, 81, 480, 832]
- NPU2: 索引 [100, 200] → videos.shape = [2, 3, 81, 480, 832]
- ... (其他NPU类似)

**Samples累积**：
- 每个NPU的samples列表：`[{batch0_data}, ...]`（只有1个batch的数据）

### Batch 1 (epoch = 0 * 2 + 1 = 1)

**采样器生成**：
- 唯一样本：索引 [500, 600, 700, 800]
- 每个重复4次，打乱后分配到8个NPU

**每个NPU生成**：
- NPU0: 索引 [500, 600] → videos.shape = [2, 3, 81, 480, 832]
- NPU1: 索引 [700, 800] → videos.shape = [2, 3, 81, 480, 832]
- ... (其他NPU类似)

**Samples累积**：
- 每个NPU的samples列表：`[{batch0_data}, {batch1_data}]`（2个batch的数据）

### Samples合并（第1484行）

**单个NPU（例如NPU0）**：
- `samples["latents"].shape = [4, ...]` (batch0的2个 + batch1的2个 = 4个)
- `samples["log_probs"].shape = [4, num_steps]`
- `samples["rewards"]["avg"].shape = [4]`

### Gather（第1555行）

**所有NPU合并后**：
- `gathered_rewards["avg"].shape = [32, num_train_timesteps]`
  - 索引100的4个副本：在位置 [0, 2, 4, 6]（假设）
  - 索引200的4个副本：在位置 [1, 3, 5, 7]（假设）
  - 索引300的4个副本：在位置 [8, 10, 12, 14]（假设）
  - 索引400的4个副本：在位置 [9, 11, 13, 15]（假设）
  - 索引500的4个副本：在位置 [16, 18, 20, 22]（假设）
  - 索引600的4个副本：在位置 [17, 19, 21, 23]（假设）
  - 索引700的4个副本：在位置 [24, 26, 28, 30]（假设）
  - 索引800的4个副本：在位置 [25, 27, 29, 31]（假设）

## 为什么你在单个batch中只看到2个视频？

你看到的 `videos.shape = [2, 3, 81, 480, 832]` 是：
- **单个batch、单个NPU**的结果
- 这2个视频可能是：
  - 同一个条件的2个副本（如果它们来自同一个唯一样本索引）
  - 或者不同条件的各1个副本

**要看到完整的4个副本，需要：**
1. 查看合并后的samples（第1484行之后）- 已添加调试打印
2. 查看gather后的gathered_rewards（第1555行之后）- 已添加调试打印

## 验证方法

运行代码后，你会看到以下调试信息：

1. **每个batch的prompts**（第1445行）：
   ```
   [Batch 0] Prompts (前30字符): ['prompt1...', 'prompt2...']
   [Batch 1] Prompts (前30字符): ['prompt3...', 'prompt4...']
   ```

2. **合并后的samples形状**（第1495行）：
   ```
   合并后的samples形状（验证k个副本的分布）:
     latents: torch.Size([4, num_steps, 16, 21, 60, 104])
     log_probs: torch.Size([4, num_steps])
     target_cameras: torch.Size([4, 21, 12])
     总样本数（当前GPU）: 4
     期望的总样本数: 4
     k (group_size): 4
   ```

3. **Gather后的总样本数**（第1558行）：
   ```
   Gather后的总样本数（所有GPU）: 32
   k (group_size): 4
   期望的唯一样本数: 8
   每个唯一样本应该有 4 个副本
   ```

## 总结

**4个相同条件的副本不在同一个batch、同一个NPU中，而是：**

1. **分布在同一个batch的不同NPU上**（因为 `total_samples = 16 >= k = 4`）
2. **每个NPU在单个batch中只看到2个样本**（`batch_size = 2`）
3. **同一个条件的4个副本分布在4个不同的NPU上**（每个NPU1个副本）
4. **最终在gather后（第1555行），所有NPU的samples合并，同一个条件的k个副本才会完全聚集**

**你看到的 `videos.shape = [2, 3, 81, 480, 832]` 是单个batch、单个NPU的结果，这是正常的！**

要验证4个副本的存在，查看gather后的gathered_rewards，那里会有32个样本，其中每4个是同一个条件的副本。
