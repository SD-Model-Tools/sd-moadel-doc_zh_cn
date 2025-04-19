# SDXL LoRA 学习脚本 `sdxl_train_network.py` 的使用方法

本文档介绍如何使用 `sd-scripts` 仓库中的 `sdxl_train_network.py` 脚本来训练 SDXL (Stable Diffusion XL) 模型的 LoRA (Low-Rank Adaptation) 模型。

## 1. 简介

`sdxl_train_network.py` 是一个用于训练 SDXL 模型附加网络（如 LoRA）的脚本。其基本用法与 `train_network.py` 相同，但需要针对 SDXL 模型进行特定设置。

本指南重点介绍 SDXL LoRA 训练，并说明与 `train_network.py` 的主要区别以及 SDXL 特有的设置项。

**前提条件：**

*   已克隆 `sd-scripts` 仓库并设置好 Python 环境。
*   准备好用于训练的数据集。（有关数据集准备，请参阅[数据集准备指南](./config.md)。）
*   已阅读 [LoRA 训练脚本 `train_network.py` 的使用方法](./train_network.md)。

## 2. 准备工作

开始训练前，需要以下文件：

1.  **训练脚本：** `sdxl_train_network.py`
2.  **数据集定义文件 (.toml)：** 一个 TOML 格式的文件，用于描述训练数据集的配置。

### 关于数据集定义文件

数据集定义文件 (`.toml`) 的基本写法与 `train_network.py` 相同。请参阅[数据集配置指南](./config.md) 和 [LoRA 训练脚本 `train_network.py` 的使用方法](./train_network.md)。

SDXL 通常使用高分辨率数据集和宽高比桶 (`enable_bucket = true`)。

下面以 `my_sdxl_dataset_config.toml` 为例进行说明。

## 3. 开始训练

通过在终端运行 `sdxl_train_network.py` 来开始训练。

以下是一个基本的 SDXL LoRA 训练命令行示例：

```bash
accelerate launch --num_cpu_threads_per_process 1 sdxl_train_network.py 
 --pretrained_model_name_or_path="<SDXL基础模型的路径>" 
 --dataset_config="my_sdxl_dataset_config.toml" 
 --output_dir="<训练结果输出目录>" 
 --output_name="my_sdxl_lora" 
 --save_model_as=safetensors 
 --network_module=networks.lora 
 --network_dim=32 
 --network_alpha=16 
 --learning_rate=1e-4 
 --unet_lr=1e-4 
 --text_encoder_lr1=1e-5 
 --text_encoder_lr2=1e-5 
 --optimizer_type="AdamW8bit" 
 --lr_scheduler="constant" 
 --max_train_epochs=10 
 --save_every_n_epochs=1 
 --mixed_precision="bf16" 
 --gradient_checkpointing 
 --cache_text_encoder_outputs 
 --cache_latents
```

与 `train_network.py` 的运行示例相比，主要区别如下：

*   运行的脚本是 `sdxl_train_network.py`。
*   `--pretrained_model_name_or_path` 需要指定 SDXL 的基础模型。
*   `--text_encoder_lr` 被拆分为 `--text_encoder_lr1` 和 `--text_encoder_lr2`（因为 SDXL 有两个 Text Encoder）。
*   推荐使用 `bf16` 或 `fp16` 作为 `--mixed_precision`。
*   推荐使用 `--cache_text_encoder_outputs` 和 `--cache_latents` 以减少 VRAM 使用量。

接下来，介绍与 `train_network.py` 不同的主要命令行参数。

### 3.1. 主要命令行参数（差异部分）

#### 模型相关

*   `--pretrained_model_name_or_path="<模型路径>"` **[必需]**
    *   指定用于训练的 **SDXL 模型**。可以是 Hugging Face Hub 的模型 ID（例如 `"stabilityai/stable-diffusion-xl-base-1.0"`），也可以是本地的 Diffusers 格式模型目录或 `.safetensors` 文件路径。
*   `--v2`, `--v_parameterization`
    *   这些参数适用于 SD1.x/2.x 模型。使用 `sdxl_train_network.py` 时，由于是针对 SDXL 模型，因此**通常不需要指定这些参数**。

#### 数据集相关

*   `--dataset_config="<配置文件路径>"`
    *   与 `train_network.py` 相同。
    *   SDXL 中通常会使用高分辨率数据或桶功能（在 `.toml` 中设置 `enable_bucket = true`）。

#### 输出和保存相关

*   与 `train_network.py` 相同。

#### LoRA 参数

*   与 `train_network.py` 相同。

#### 训练参数

*   `--learning_rate=1e-4`
    *   全局学习率。若未指定 `unet_lr`、`text_encoder_lr1` 和 `text_encoder_lr2`，则使用该值。
*   `--unet_lr=1e-4`
    *   U-Net 部分 LoRA 模块的学习率。若未指定，则使用 `--learning_rate` 的值。
*   `--text_encoder_lr1=1e-5`
    *   **Text Encoder 1 (OpenCLIP ViT-G/14) 的 LoRA 模块**的学习率。通常建议比 U-Net 的学习率小。
*   `--text_encoder_lr2=1e-5`
    *   **Text Encoder 2 (CLIP ViT-L/14) 的 LoRA 模块**的学习率。通常建议比 U-Net 的学习率小。
*   `--optimizer_type="AdamW8bit"`
    *   与 `train_network.py` 相同。
*   `--lr_scheduler="constant"`
    *   与 `train_network.py` 相同。
*   `--lr_warmup_steps`
    *   与 `train_network.py` 相同。
*   `--max_train_steps`, `--max_train_epochs`
    *   与 `train_network.py` 相同。
*   `--mixed_precision="bf16"`
    *   混合精度训练的设置。SDXL 中推荐使用 `bf16` 或 `fp16`。请根据 GPU 的支持情况选择，以减少 VRAM 使用并提升训练速度。
*   `--gradient_accumulation_steps=1`
    *   与 `train_network.py` 相同。
*   `--gradient_checkpointing`
    *   与 `train_network.py` 相同。由于 SDXL 内存消耗较大，推荐启用此选项。
*   `--cache_latents`
    *   将 VAE 的输出缓存到内存（或指定 `--cache_latents_to_disk` 时缓存到磁盘）。可以跳过 VAE 计算，从而减少 VRAM 使用并加速训练。但会禁用图像增强（如 `--color_aug`、`--flip_aug`、`--random_crop` 等）。SDXL 训练推荐使用此选项。
*   `--cache_latents_to_disk`
    *   与 `--cache_latents` 配合使用，将缓存存储到磁盘。在首次加载数据集时，会将 VAE 输出缓存到磁盘。在后续训练中可以跳过 VAE 计算，适合数据量较大的情况。
*   `--cache_text_encoder_outputs`
    *   将 Text Encoder 的输出缓存到内存（或指定 `--cache_text_encoder_outputs_to_disk` 时缓存到磁盘）。可以减少 VRAM 使用并加速训练。但会禁用文本增强（如 `--shuffle_caption`、`--caption_dropout_rate` 等）。
    *   **注意：** 使用此选项时，Text Encoder 的 LoRA 模块将无法训练（必须指定 `--network_train_unet_only`）。
*   `--cache_text_encoder_outputs_to_disk`
    *   与 `--cache_text_encoder_outputs` 配合使用，将缓存存储到磁盘。
*   `--no_half_vae`
    *   即使在使用混合精度 (`fp16`/`bf16`) 时，也让 VAE 以 `float32` 运行。SDXL 的 VAE 在 `float16` 下可能不稳定，因此在指定 `fp16` 时建议启用此选项。
*   `--clip_skip`
    *   SDXL 通常不需要使用，推荐不指定。
*   `--fused_backward_pass`
    *   将梯度计算和优化器步骤合并，以减少 VRAM 使用量。适用于 SDXL（目前仅支持 `Adafactor` 优化器）。

#### 其他

*   `--seed`、`--logging_dir`、`--log_prefix` 等参数与 `train_network.py` 相同。

### 3.2. 开始训练

配置好必要参数后，运行命令即可开始训练。训练进度会输出到控制台。基本流程与 `train_network.py` 相同。

## 4. 使用训练好的模型

训练完成后，在 `--output_dir` 指定的目录下会保存以 `--output_name` 命名的 LoRA 模型文件（例如 `.safetensors`）。

该文件可用于支持 SDXL 的 GUI 工具，如 AUTOMATIC1111/stable-diffusion-webui 和 ComfyUI。

## 5. 补充：与 `train_network.py` 的主要区别

*   **目标模型：** `sdxl_train_network.py` 专门用于 SDXL 模型。
*   **Text Encoder：** SDXL 有两个 Text Encoder，因此学习率的指定（`--text_encoder_lr1`、`--text_encoder_lr2`）有所不同。
*   **缓存功能：** `--cache_text_encoder_outputs` 在 SDXL 中效果显著，推荐使用。
*   **推荐设置：** 由于 SDXL 内存占用较大，推荐使用 `bf16` 或 `fp16` 混合精度、`gradient_checkpointing`，以及缓存功能（`--cache_latents`、`--cache_text_encoder_outputs`）。若使用 `fp16`，建议搭配 `--no_half_vae` 让 VAE 以 `float32` 运行。

更多详细选项，请参考脚本的帮助信息 (`python sdxl_train_network.py --help`) 或仓库中的其他文档。