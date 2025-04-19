NovelAI提出的训练方法、自动标注、标签化、Windows + VRAM 12GB（对于SD v1.x）环境等所对应的fine tuning。在这里，fine tuning指的是用图像和标注训练模型（不包括LoRA、Textual Inversion和Hypernetworks）。

请同时参考[关于训练的共同文档](./train.md)。

# 概要

使用Diffusers对Stable Diffusion的U-Net进行fine tuning。对应了NovelAI的文章中的以下改进（虽然Aspect Ratio Bucketing参考了NovelAI的代码，但最终的代码都是原创的）。

* 使用CLIP（Text Encoder）的倒数第二层的输出，而不是最后一层的输出。
* 以非正方形分辨率进行训练（Aspect Ratio Bucketing）。
* 将标记长度从75扩展到225。
* 使用BLIP进行标注（自动创建标注）、使用DeepDanbooru或WD14Tagger进行自动标签化。
* 支持Hypernetwork的训练。
* 支持Stable Diffusion v2.0（base以及768/v）。
* 提前获取VAE的输出并保存到磁盘，以减少训练时的内存使用量并提高速度。

默认情况下，不训练Text Encoder。在整体模型的fine tuning中，通常只训练U-Net（NovelAI似乎也是如此）。可以通过选项指定来训练Text Encoder。

# 关于追加功能

## 更改CLIP的输出

为了将提示反映到图像中，CLIP（Text Encoder）将文本转换为特征量。Stable Diffusion使用CLIP的最后一层的输出，但可以更改为使用倒数第二层的输出。据NovelAI说，这样可以更准确地反映提示。
也可以保持原样，使用最后一层的输出。

*注意：Stable Diffusion 2.0默认使用倒数第二层。请不要指定clip_skip选项。

## 以非正方形分辨率进行训练

Stable Diffusion是在512*512的分辨率下训练的，但除了这个分辨率以外，还会在256*1024或384*640等分辨率下进行训练。这样可以减少裁剪的部分，更准确地学习提示和图像之间的关系。
训练分辨率会在不超过作为参数给出的分辨率的面积（=内存使用量）的范围内，以64像素为单位进行调整。

在机器学习中，通常将输入大小统一起来，但并不是特别有约束，实际上只要在同一个批次中统一即可。NovelAI所说的bucketing是指预先根据长宽比将训练数据分类到不同的训练分辨率中，然后在每个bucket内创建批次，以统一批次内的图像大小。

## 将标记长度从75扩展到225

Stable Diffusion的最大标记长度是75（包括开始和结束标记是77），这里将其扩展到225。
但是，由于CLIP的最大接受长度是75，因此在标记长度为225时，会简单地将其分成三部分，分别调用CLIP，然后将结果连接起来。

*注意：这种实现方式是否合适尚不明确。目前看来是可以正常工作的。特别是在2.0版本中，没有可以参考的实现方式，因此采用了独自的实现方式。

*注意：在Automatic1111的Web UI中，似乎会根据逗号进行分割，但这里只是进行了简单的分割。

# 训练步骤

请预先参考本仓库的README，进行环境设置。

## 数据准备

请参考[关于训练数据的准备](./train.md)。fine tuning仅支持使用元数据的fine tuning方式。

## 执行训练

例如，可以通过以下命令执行训练。以下是一个为了减少内存使用的设置示例。请根据需要修改每一行。

```
accelerate launch --num_cpu_threads_per_process 1 fine_tune.py 
    --pretrained_model_name_or_path=<.ckpt或.safetensors或Diffusers模型目录> 
    --output_dir=<训练好的模型的输出目录>  
    --output_name=<训练好的模型输出时的文件名> 
    --dataset_config=<数据准备时创建的.toml文件> 
    --save_model_as=safetensors 
    --learning_rate=5e-6 --max_train_steps=10000 
    --use_8bit_adam --xformers --gradient_checkpointing
    --mixed_precision=fp16
```

`num_cpu_threads_per_process`通常设置为1。

在`pretrained_model_name_or_path`中指定用于进行额外训练的原始模型。可以指定Stable Diffusion的checkpoint文件（.ckpt或.safetensors）、Diffusers的本地磁盘上的模型目录、Diffusers的模型ID（例如"stabilityai/stable-diffusion-2"）。

在`output_dir`中指定训练后的模型的保存目录。在`output_name`中指定模型的文件名（不包括扩展名）。通过`save_model_as`指定以safetensors格式保存模型。

在`dataset_config`中指定`.toml`文件。在文件内，初始的批次大小应设置为`1`以减少内存消耗。

将训练步数`max_train_steps`设置为10000。这里将学习率`learning_rate`设置为5e-6。

为了减少内存使用，指定`mixed_precision="fp16"`（在RTX30系列之后的显卡中，也可以指定`bf16`。请与在环境设置时对accelerate进行的设置保持一致）。同时指定`gradient_checkpointing`。

为了使用内存消耗较少的8bit AdamW作为优化器（用于将模型优化/训练到适应训练数据的类），指定`optimizer_type="AdamW8bit"`。

指定`xformers`选项，使用xformers的CrossAttention。如果没有安装xformers或出现错误（在某些环境下，当`mixed_precision="no"`时可能会出现这种情况），可以改为指定`mem_eff_attn`选项，使用节省内存版本的CrossAttention（速度会变慢）。

如果内存足够，可以编辑`.toml`文件，将批次大小增加到例如`4`（可能会提高速度和精度）。

### 常见选项

在以下情况下，请参考有关选项的文档。

- 训练Stable Diffusion 2.x或从中派生的模型
- 训练clip skip为2或以上的模型
- 使用超过75个标记的标注进行训练

### 关于批次大小

由于是对整个模型进行训练，与LoRA等相比，内存消耗会更大（与DreamBooth相同）。

### 关于学习率

通常使用1e-6到5e-6左右的学习率。可以参考其他fine tuning的例子。

### 以前的格式的数据集指定的命令行

通过选项指定分辨率和批次大小。命令行示例如下。

```
accelerate launch --num_cpu_threads_per_process 1 fine_tune.py 
    --pretrained_model_name_or_path=model.ckpt 
    --in_json meta_lat.json 
    --train_data_dir=train_data 
    --output_dir=fine_tuned 
    --shuffle_caption 
    --train_batch_size=1 --learning_rate=5e-6 --max_train_steps=10000 
    --use_8bit_adam --xformers --gradient_checkpointing
    --mixed_precision=bf16
    --save_every_n_epochs=4
```

<!-- 
### 使用fp16梯度的训练（实验性功能）
如果指定full_fp16选项，则会将梯度从通常的float32改为float16（fp16）进行训练（似乎不是混合精度，而是完全的fp16训练）。这样，在SD1.x的512*512尺寸下，VRAM使用量可以小于8GB，在SD2.x的512*512尺寸下，可以小于12GB。

请预先通过accelerate config指定fp16，并通过选项指定mixed_precision="fp16"（不能使用bf16）。

为了最小化内存使用量，请指定xformers、use_8bit_adam和gradient_checkpointing选项，并将train_batch_size设置为1。
（如果有余裕，可以逐步增加train_batch_size，应该可以稍微提高精度。）

这是通过对PyTorch的源代码打补丁强行实现的（在PyTorch 1.12.1和1.13.0中确认）。精度会明显下降，训练失败的概率也会增加。学习率和步数设置似乎也很敏感。请在了解这些风险的基础上，自行承担风险使用。
-->

# fine tuning特有的其他主要选项

关于所有选项，请参考其他文档。

## `train_text_encoder`
将Text Encoder也作为训练对象。内存使用量会略微增加。

在通常的fine tuning中，不将Text Encoder作为训练对象（可能是因为U-Net是根据Text Encoder的输出进行训练的），但在训练数据较少的情况下，像DreamBooth一样对Text Encoder进行训练似乎也是有效的。

## `diffusers_xformers`
使用Diffusers的xformers功能，而不是脚本自有的xformers替换功能。这样就无法进行Hypernetwork的训练。