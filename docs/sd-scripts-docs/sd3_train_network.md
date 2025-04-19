# `sd3_train_network.py` 用的Stable Diffusion 3/3.5模型的LoRA训练指南

本文档将介绍如何使用`sd-scripts`存储库中的`sd3_train_network.py`来训练Stable Diffusion 3 (SD3) 和 Stable Diffusion 3.5 (SD3.5) 模型的LoRA（Low-Rank Adaptation）模型。

## 1. 简介

`sd3_train_network.py`是一个用于训练Stable Diffusion 3/3.5模型的LoRA等附加网络的脚本。SD3采用了一种名为MMDiT（多模态扩散Transformer）的新架构，与传统的Stable Diffusion模型结构不同。使用此脚本可以创建针对SD3/3.5模型的LoRA模型。

本指南面向了解LoRA训练基本步骤的用户，并假设您具有使用`train_network.py`进行训练的经验。有关基本用法和通用选项，请参阅[`train_network.py`的使用指南](./train_network.md)。

**前提条件：**

*   已克隆`sd-scripts`存储库并设置Python环境。
*   已准备好用于训练的数据集。（有关数据集准备，请参阅[数据集配置指南](./config.md)。）
*   已准备好待训练的SD3/3.5模型文件。

## 2. 与`train_network.py`的区别

`sd3_train_network.py`基于`train_network.py`，并针对SD3/3.5模型进行了修改。主要区别如下：

*   **目标模型：** 适用于Stable Diffusion 3 Medium / Large (3.0 / 3.5) 模型。
*   **模型结构：** 使用MMDiT（基于Transformer）代替U-Net。使用CLIP-L、CLIP-G和T5-XXL作为文本编码器。VAE与SDXL兼容，但输入缩放处理不同。
*   **参数：** 有用于指定SD3/3.5模型、文本编码器组和VAE的参数。但是，如果是单个`.safetensors`文件，则可以在内部自动分离，因此无需指定单独的路径。
*   **部分参数不兼容：** 用于Stable Diffusion v1/v2的参数（例如：`--v2`、`--v_parameterization`、`--clip_skip`）不用于SD3/3.5的训练。
*   **SD3特有的参数：** 添加了文本编码器的注意力掩码和丢弃率、位置嵌入调整（适用于SD3.5）、时间步采样和损失加权等参数。

## 3. 准备

开始训练之前，需要以下文件：

1.  **训练脚本：** `sd3_train_network.py`
2.  **SD3/3.5模型文件：** 用于训练的SD3/3.5模型的`.safetensors`文件。建议使用单文件格式（Diffusers/ComfyUI/AUTOMATIC1111格式）。
    *   如果文本编码器或VAE是单独的文件，请在相应的参数中指定路径。
3.  **数据集定义文件（.toml）：** 一个TOML格式的文件，用于描述训练数据集的配置。（详情请参阅[数据集配置指南](./config.md)。）
    *   这里使用`my_sd3_dataset_config.toml`作为示例。

## 4. 开始训练

通过在终端中运行`sd3_train_network.py`来开始训练。基本命令行结构与`train_network.py`类似，但需要指定SD3/3.5特有的参数。

下面是一个基本的命令行示例：

```bash
accelerate launch --num_cpu_threads_per_process 1 sd3_train_network.py 
 --pretrained_model_name_or_path="<SD3模型的路径>" 
 --dataset_config="my_sd3_dataset_config.toml" 
 --output_dir="<训练结果的输出目录>" 
 --output_name="my_sd3_lora" 
 --save_model_as=safetensors 
 --network_module=networks.lora 
 --network_dim=16 
 --network_alpha=1 
 --learning_rate=1e-4 
 --optimizer_type="AdamW8bit" 
 --lr_scheduler="constant" 
 --sdpa  
 --max_train_epochs=10 
 --save_every_n_epochs=1 
 --mixed_precision="fp16" 
 --gradient_checkpointing 
 --apply_lg_attn_mask 
 --apply_t5_attn_mask 
 --weighting_scheme="sigma_sqrt" 
 --blocks_to_swap=32
```

※实际上应该写在一行或使用适当的换行符（`\` 或 `^`）。

### 4.1. 主要命令行参数说明（`train_network.py`新增或更改的部分）

除了[`train_network.py`指南](./train_network.md)中介绍的参数外，还需要指定以下SD3/3.5特有的参数。有关通用参数（`--output_dir`、`--output_name`、`--network_module`、`--network_dim`、`--network_alpha`、`--learning_rate`等），请参阅上述指南。

#### 模型相关参数

*   `--pretrained_model_name_or_path="<SD3模型的路径>"` **[必填]**
    *   指定用于训练的SD3/3.5模型的`.safetensors`文件路径。假设为单文件格式（Diffusers/ComfyUI/AUTOMATIC1111格式）。
*   `--clip_l`、`--clip_g`、`--t5xxl`、`--vae`：
    *   如果基础模型是单文件格式，通常不需要指定这些参数（会自动从模型内部加载）。
    *   如果文本编码器或VAE作为单独文件提供，请指定相应的`.safetensors`文件路径。

#### SD3/3.5训练参数

*   `--t5xxl_max_token_length=<整数>`
    *   指定T5-XXL文本编码器使用的最大Token长度。SD3的默认值为`256`。可能需要根据数据集的字幕长度进行调整。
*   `--apply_lg_attn_mask`
    *   对CLIP-L和CLIP-G的输出应用与填充Token相对应的注意力掩码（零填充）。
*   `--apply_t5_attn_mask`
    *   对T5-XXL的输出应用与填充Token相对应的注意力掩码（零填充）。
*   `--clip_l_dropout_rate`、`--clip_g_dropout_rate`、`--t5_dropout_rate`：
    *   对每个文本编码器的输出以指定的概率应用丢弃（输出置零）。有助于防止过拟合。默认值为`0.0`（无丢弃）。
*   `--pos_emb_random_crop_rate=<浮点数>` **[适用于SD3.5]**
    *   指定对MMDiT的位置嵌入应用随机裁剪的概率。这是SD3 5M（3.5）模型中训练的功能，对其他模型的效果有限。默认值为`0.0`。
*   `--enable_scaled_pos_embed` **[适用于SD3.5]**
    *   在多分辨率训练时，根据分辨率缩放位置嵌入。这是SD3 5M（3.5）模型中训练的功能，对其他模型的效果有限。
*   `--training_shift=<浮点数>`
    *   用于调整训练时时间步（噪声水平）分布的偏移值。与`weighting_scheme`结合应用。大于`1.0`的值倾向于强调噪声较大（结构较多）的区域，而小于`1.0`的值则强调噪声较小（细节较多）的区域。默认值为`1.0`。
*   `--weighting_scheme=<选项>`
    *   指定根据损失计算时的时间步（噪声水平）进行加权的方法。可选择`sigma_sqrt`、`logit_normal`、`mode`、`cosmap`、`uniform`（或`none`）。SD3论文中使用了`sigma_sqrt`。默认值为`uniform`。
*   `--logit_mean`、`--logit_std`、`--mode_scale`：
    *   当`weighting_scheme`选择`logit_normal`或`mode`时，用于控制其分布的参数。通常使用默认值即可。

#### 内存和速度相关参数

*   `--blocks_to_swap=<整数>` **[实验性功能]**
    *   为了减少VRAM使用量，设置在CPU和GPU之间交换模型部分（MMDiT的Transformer块）。以整数指定要交换的块数（例如：`32`）。增加该值会减少VRAM使用量，但会降低训练速度。请根据GPU的VRAM容量进行调整。可与`gradient_checkpointing`结合使用。
    *   不能与`--cpu_offload_checkpointing`同时使用。

#### 不兼容或已弃用的参数

*   `--v2`、`--v_parameterization`、`--clip_skip`：这些是针对Stable Diffusion v1/v2的参数，不用于SD3/3.5的训练。

### 4.2. 开始训练

设置必要的参数并运行命令后，训练将开始。基本流程和日志检查方法与[`train_network.py`指南](./train_network.md)相同。

## 5. 使用训练好的模型

训练完成后，LoRA模型文件（例如：`my_sd3_lora.safetensors`）将保存在指定的`output_dir`中。该文件可用于支持SD3/3.5模型的推理环境（例如：ComfyUI等）。

## 6. 其他功能

`sd3_train_network.py`包含许多与`train_network.py`相同的功能，例如生成示例图像（`--sample_prompts`等）和详细的优化器设置。有关这些功能的详细信息，请参阅[`train_network.py`指南](./train_network.md)和脚本的帮助（`python sd3_train_network.py --help`）。