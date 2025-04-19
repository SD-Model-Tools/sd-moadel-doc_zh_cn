flux_train.md 
# 使用 `flux_train_network.py` 训练 FLUX.1 模型的 LoRA 指南

本文档介绍了如何使用 `sd-scripts` 仓库中的 `flux_train_network.py` 脚本来训练 FLUX.1 模型的 LoRA（低秩自适应）模型。

## 1. 简介

`flux_train_network.py` 是一个用于训练 FLUX.1 模型的附加网络（如 LoRA）的脚本。FLUX.1 是一个图像生成模型，其架构与 Stable Diffusion 不同。使用此脚本，您可以创建一个 LoRA 模型，以再现特定的角色或绘画风格。

本指南面向已经了解 LoRA 训练基本步骤的用户，并假设您有使用 `train_network.py` 进行训练的经验。有关基本用法和通用选项，请参阅 [`train_network.py` 指南](./train_network.md)。

**前提条件：**

*   已克隆 `sd-scripts` 仓库并完成 Python 环境的设置。
*   已准备好用于训练的数据集。（有关数据集的准备，请参阅[数据集配置指南](./config.md)。）

## 2. 与 `train_network.py` 的区别

`flux_train_network.py` 基于 `train_network.py`，并针对 FLUX.1 模型进行了修改。主要区别如下：

*   **目标模型：** 支持 FLUX.1 模型（dev 或 schnell 版本）。
*   **模型架构：** 与 Stable Diffusion 不同，FLUX.1 采用基于 Transformer 的架构。它使用 CLIP-L 和 T5-XXL 两个文本编码器，并使用专用的自动编码器（AE）代替 VAE。
*   **必需参数：** 需要指定 FLUX.1 模型、CLIP-L、T5-XXL 和 AE 的模型文件路径。
*   **部分参数不兼容：** 一些针对 Stable Diffusion 的参数（例如 `--v2`, `--clip_skip`, `--max_token_length`）不适用于 FLUX.1 训练。
*   **FLUX.1 特有参数：** 添加了用于指定 FLUX.1 特有训练参数的参数，例如时间步采样方法和引导尺度。

## 3. 准备

开始训练前，您需要准备以下文件：

1.  **训练脚本：** `flux_train_network.py`
2.  **FLUX.1 模型文件：** 用作训练基础的 FLUX.1 模型的 `.safetensors` 文件（例如 `flux1-dev.safetensors`）。
3.  **文本编码器模型文件：**
    *   CLIP-L 模型的 `.safetensors` 文件。
    *   T5-XXL 模型的 `.safetensors` 文件。
4.  **自动编码器模型文件：** 与 FLUX.1 相对应的 AE 模型的 `.safetensors` 文件。
5.  **数据集定义文件 (.toml)：** 一个 TOML 格式的文件，用于描述训练数据集的配置。（详见[数据集配置指南](./config.md)。）

    *   这里以 `my_flux_dataset_config.toml` 为例。

## 4. 开始训练

通过在终端中运行 `flux_train_network.py` 来开始训练。基本命令行结构与 `train_network.py` 类似，但需要指定 FLUX.1 特有的参数。

下面是一个基本的命令行示例：

```bash
accelerate launch --num_cpu_threads_per_process 1 flux_train_network.py 
 --pretrained_model_name_or_path="<FLUX.1 模型路径>" 
 --clip_l="<CLIP-L 模型路径>" 
 --t5xxl="<T5-XXL 模型路径>" 
 --ae="<AE 模型路径>" 
 --dataset_config="my_flux_dataset_config.toml" 
 --output_dir="<训练结果输出目录>" 
 --output_name="my_flux_lora" 
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
 --apply_t5_attn_mask 
 --blocks_to_swap=18
```

※ 实际使用时，请确保命令写在一行，或使用适当的换行符（`\` 或 `^`）。

### 4.1. 主要命令行参数说明（与 `train_network.py` 相比新增或修改的部分）

除了 [`train_network.py` 指南](./train_network.md) 中介绍的参数外，您还需要指定以下 FLUX.1 特有的参数。有关通用参数（例如 `--output_dir`, `--output_name`, `--network_module`, `--network_dim`, `--network_alpha`, `--learning_rate` 等），请参考上述指南。

#### 模型相关 [必需]

*   `--pretrained_model_name_or_path="<FLUX.1 模型路径>"` **[必需]**
    *   指定用作训练基础的 FLUX.1 模型（dev 或 schnell 版本）的 `.safetensors` 文件路径。目前不支持 Diffusers 格式的目录。
*   `--clip_l="<CLIP-L 模型路径>"` **[必需]**
    *   指定 CLIP-L 文本编码器模型的 `.safetensors` 文件路径。
*   `--t5xxl="<T5-XXL 模型路径>"` **[必需]**
    *   指定 T5-XXL 文本编码器模型的 `.safetensors` 文件路径。
*   `--ae="<AE 模型路径>"` **[必需]**
    *   指定与 FLUX.1 对应的自动编码器模型的 `.safetensors` 文件路径。

#### FLUX.1 训练参数

*   `--t5xxl_max_token_length=<整数>`
    *   指定 T5-XXL 文本编码器使用的最大 token 长度。如果省略，对于 schnell 版本默认为 256，对于 dev 版本默认为 512。您可能需要根据数据集的标题长度进行调整。
*   `--apply_t5_attn_mask`
    *   在 T5-XXL 输出与 FLUX 模型内部（双块）注意力计算期间，应用与填充 token 对应的注意力掩码。这可能会提高精度，但会略微增加计算成本。
*   `--guidance_scale=<浮点数>`
    *   FLUX.1 dev 版本是在特定的引导尺度值下进行蒸馏的，因此在训练时也需要指定该值。默认值为 `3.5`。对于 schnell 版本，通常忽略此参数。
*   `--timestep_sampling=<选项>`
    *   指定训练期间使用的采样时间步（噪声水平）的方法。可选择 `sigma`, `uniform`, `sigmoid`, `shift`, `flux_shift`。默认值为 `sigma`。
*   `--sigmoid_scale=<浮点数>`
    *   当 `timestep_sampling` 指定为 `sigmoid`、`shift` 或 `flux_shift` 时的缩放系数。默认值为 `1.0`。
*   `--model_prediction_type=<选项>`
    *   指定模型预测的内容。可选择 `raw`（直接使用预测值）、`additive`（将预测值添加到噪声输入）、`sigma_scaled`（应用 sigma 缩放）。默认值为 `sigma_scaled`。
*   `--discrete_flow_shift=<浮点数>`
    *   指定流匹配中使用的调度器的偏移值。默认值为 `3.0`。

#### 内存和速度相关

*   `--blocks_to_swap=<整数>` **[实验性功能]**
    *   为了减少 VRAM 使用量，可以在 CPU 和 GPU 之间交换模型的部分（Transformer 块）。指定要交换的块数（例如 `18`）。增加此值会减少 VRAM 使用量，但会降低训练速度。请根据您的 GPU VRAM 容量进行调整。该参数可与 `gradient_checkpointing` 一起使用。
    *   不能与 `--cpu_offload_checkpointing` 同时使用。

#### 不兼容或已弃用的参数

*   `--v2`, `--v_parameterization`, `--clip_skip`：这些是 Stable Diffusion 特有的参数，不用于 FLUX.1 训练。
*   `--max_token_length`：这是针对 Stable Diffusion v1/v2 的参数。对于 FLUX.1，请使用 `--t5xxl_max_token_length`。
*   `--split_mode`：已弃用。请使用 `--blocks_to_swap` 代替。

### 4.2. 开始训练

设置好必要的参数并运行命令后，训练就会开始。基本流程和日志查看方法与 [`train_network.py` 指南](train_network.md) 相同。

## 5. 使用训练好的模型

训练完成后，LoRA 模型文件（例如 `my_flux_lora.safetensors`）将被保存到指定的 `output_dir`。该文件可在支持 FLUX.1 模型的推理环境（例如 ComfyUI + ComfyUI-FluxNodes）中使用。

## 6. 其他功能

`flux_train_network.py` 还包含许多与 `train_network.py` 相同的功能，例如生成示例图像（使用 `--sample_prompts` 等）和详细的优化器设置。有关这些功能的详细信息，请参阅 [`train_network.py` 指南](./train_network.md) 或运行脚本的帮助命令 (`python flux_train_network.py --help`)。