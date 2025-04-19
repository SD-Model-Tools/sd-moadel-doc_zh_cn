## SDXL 训练

未来，该文档将被移动到训练文档中。以下是关于 SDXL 训练脚本的简要说明。

### SDXL 训练脚本

- `sdxl_train.py` 是一个用于 SDXL 微调的脚本，其用法与 `fine_tune.py` 几乎相同，但也支持 DreamBooth 数据集。
  - 新增了 `--full_bf16` 选项。感谢 KohakuBlueleaf！
    - 该选项启用全 bfloat16 训练（包括梯度）。此选项有助于减少 GPU 内存使用。
    - 全 bfloat16 训练可能不稳定，请谨慎使用。
  - 现在，`sdxl_train.py` 支持为每个 U-Net 块设置不同的学习率。使用 `--block_lr` 选项指定，格式为用逗号分隔的 23 个值，如 `--block_lr 1e-3,1e-3 ... 1e-3`。
    - 这 23 个值分别对应 `0: time/label embed, 1-9: input blocks 0-8, 10-12: mid blocks 0-2, 13-21: output blocks 0-8, 22: out`。
- `prepare_buckets_latents.py` 现在支持 SDXL 微调。

- `sdxl_train_network.py` 是一个用于 SDXL LoRA 训练的脚本，其用法与 `train_network.py` 几乎相同。

- 两个脚本都有以下附加选项：
  - `--cache_text_encoder_outputs` 和 `--cache_text_encoder_outputs_to_disk`：缓存文本编码器的输出。此选项有助于减少 GPU 内存使用。但不能与随机打乱或丢弃标题的选项一起使用。
  - `--no_half_vae`：禁用半精度（混合精度）VAE。SDXL 的 VAE 在某些情况下似乎会产生 NaN。此选项有助于避免 NaN。

- `--weighted_captions` 选项尚不支持这两个脚本。

- `sdxl_train_textual_inversion.py` 是一个用于 SDXL Textual Inversion 训练的脚本，其用法与 `train_textual_inversion.py` 几乎相同。
  - 不支持 `--cache_text_encoder_outputs`。
  - 有两种标题选项：
    1. 使用标题进行训练。所有标题必须包含标记字符串。标记字符串将被替换为多个标记。
    2. 使用 `--use_object_template` 或 `--use_style_template` 选项。标题将从模板生成。现有的标题将被忽略。
  - 下面是嵌入的格式。

- `--min_timestep` 和 `--max_timestep` 选项已添加到每个训练脚本中。这些选项可用于使用不同的时间步训练 U-Net。默认值为 0 和 1000。

### SDXL 实用脚本

- 新增了 `tools/cache_latents.py`。该脚本可用于提前将潜变量缓存到磁盘。
  - 选项与 `sdxl_train.py` 几乎相同。有关用法，请参见帮助消息。
  - 请按以下方式启动脚本：
    `accelerate launch  --num_cpu_threads_per_process 1 tools/cache_latents.py ...`
  - 该脚本应该可以在多 GPU 上工作，但尚未在我的环境中进行测试。

- 新增了 `tools/cache_text_encoder_outputs.py`。该脚本可用于提前将文本编码器输出缓存到磁盘。
  - 选项与 `cache_latents.py` 和 `sdxl_train.py` 几乎相同。有关用法，请参见帮助消息。

- 新增了 `sdxl_gen_img.py`。该脚本可用于使用 SDXL（包括 LoRA、Textual Inversion 和 ControlNet-LLLite）生成图像。有关用法，请参见帮助消息。

### SDXL 训练技巧

- SDXL 的默认分辨率为 1024x1024。
- 微调可以在 24GB GPU 内存下以批量大小为 1 进行。对于 24GB GPU，建议使用以下选项__进行 24GB GPU 内存下的微调__：
  - 仅训练 U-Net。
  - 使用梯度检查点。
  - 使用 `--cache_text_encoder_outputs` 选项和缓存潜变量。
  - 使用 Adafactor 优化器。RMSprop 8bit 或 Adagrad 8bit 可能有效。AdamW 8bit 似乎无效。
- LoRA 训练可以在 8GB GPU 内存下进行（推荐 10GB）。为了减少 GPU 内存使用，建议使用以下选项：
  - 仅训练 U-Net。
  - 使用梯度检查点。
  - 使用 `--cache_text_encoder_outputs` 选项和缓存潜变量。
  - 使用 8 位优化器或 Adafactor 优化器之一。
  - 使用较低的维度（对于 8GB GPU 为 4 到 8）。
- 强烈建议为 SDXL LoRA 使用 `--network_train_unet_only` 选项。因为 SDXL 有两个文本编码器，训练结果可能不符合预期。
- PyTorch 2 似乎比 PyTorch 1 使用的 GPU 内存略少。
- `--bucket_reso_steps` 可以设置为 32，而不是默认值 64。小于 32 的值不适用于 SDXL 训练。

Adafactor 优化器的示例配置（固定学习率）：
```toml
optimizer_type = "adafactor"
optimizer_args = [ "scale_parameter=False", "relative_step=False", "warmup_init=False" ]
lr_scheduler = "constant_with_warmup"
lr_warmup_steps = 100
learning_rate = 4e-7 # SDXL 原始学习率
```

### SDXL Textual Inversion 嵌入格式

```python
from safetensors.torch import save_file

state_dict = {"clip_g": embs_for_text_encoder_1280, "clip_l": embs_for_text_encoder_768}
save_file(state_dict, file)
```

### ControlNet-LLLite

新增了 ControlNet-LLLite，一种用于 SDXL 的新型 ControlNet 方法。详细信息请参见 [文档](./train_lllite.md)。