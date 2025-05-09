# LoRA的训练

这是将[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)（arxiv）和[LoRA](https://github.com/microsoft/LoRA)（github）应用于Stable Diffusion的结果。

我大量参考了[cloneofsimo的仓库](https://github.com/cloneofsimo/lora)，在此表示感谢。

通常的LoRA仅适用于Linear和内核大小为1x1的Conv2d，但也可以扩展到内核大小为3x3的Conv2d。

将Conv2d 3x3扩展到LoRA是由[cloneofsimo](https://github.com/cloneofsimo/lora)首先发布的，[KohakuBlueleaf](https://github.com/KohakuBlueleaf/LoCon)揭示了其有效性。在此对KohakuBlueleaf表示深深的感谢。

即使在8GB VRAM下，似乎也能勉强运行。

请同时参阅[关于训练的共同文档](./train.md)。

# 可以训练的LoRA类型

支持以下两种类型。以下是在本仓库中的自定义名称。

1. __LoRA-LierLa__：（LoRA for __Li__n __e__a __r__  __La__ yers，读作LierLa）

    适用于Linear和内核大小为1x1的Conv2d的LoRA。

2. __LoRA-C3Lier__：（LoRA for __C__olutional layers with __3__ x3 Kernel and  __Li__n __e__a __r__ layers，读作C3Lier）

    在1的基础上，额外适用于内核大小为3x3的Conv2d的LoRA。

与LoRA-LierLa相比，LoRA-C3Lier由于适用的层增加，可能期待更高的精度。

在训练时，也可以使用 __DyLoRA__（后述）。

## 关于训练后的模型的注意事项

LoRA-LierLa可以被AUTOMATIC1111的Web UI的LoRA功能使用。

要使用LoRA-C3Lier在Web UI中生成图像，请使用这个[WebUI扩展](https://github.com/kohya-ss/sd-webui-additional-networks)。

两者都可以使用本仓库中的脚本将训练好的LoRA模型与Stable Diffusion模型合并。

目前与cloneofsimo的仓库和d8ahazard的[Dreambooth Extension for Stable-Diffusion-WebUI](https://github.com/d8ahazard/sd_dreambooth_extension)不兼容，因为进行了一些功能扩展（后述）。

# 训练步骤

请先参考本仓库的README，进行环境准备。

## 数据准备

请参考[关于训练数据准备](./train.md)。

## 执行训练

使用`train_network.py`。

在`train_network.py`中，通过`--network_module`选项指定要训练的模块名称。对于LoRA，指定`network.lora`。

学习率似乎比通常的DreamBooth或fine tuning要高，建议使用`1e-4`~`1e-3`左右的值。

以下是一个命令行示例。

```
accelerate launch --num_cpu_threads_per_process 1 train_network.py 
    --pretrained_model_name_or_path=<.ckpt或.safetensors或Diffusers版模型的目录> 
    --dataset_config=<数据准备时创建的.toml文件> 
    --output_dir=<训练好的模型的输出文件夹>  
    --output_name=<训练好的模型输出时的文件名> 
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=400 
    --learning_rate=1e-4 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
    --save_every_n_epochs=1 
    --network_module=networks.lora
```

这个命令行将训练LoRA-LierLa。

LoRA模型将被保存到`--output_dir`选项指定的文件夹中。其他选项和优化器等请参考[训练的共同文档](./train.md)的“常用选项”。

此外，还可以指定以下选项。

* `--network_dim`
  * 指定LoRA的RANK（例如`--network_dim=4`）。省略时默认为4。数值越大表示能力越强，但训练所需的内存和时间也会增加。盲目增加数值也不一定好。
* `--network_alpha`
  * 指定用于防止下溢并稳定训练的`alpha`值。默认为1。`
  * 当与`network_dim`的值相同指定时，行为与以前的版本相同。
* `--persistent_data_loader_workers`
  * 在Windows环境中指定时，epoch之间的等待时间将大大缩短。
* `--max_data_loader_n_workers`
  * 指定数据加载的进程数。进程数越多，数据加载越快，GPU利用率越高，但会消耗主内存。默认值为“`8`或`CPU并发线程数-1`中较小的一个”，因此如果主内存不充裕或GPU利用率达到90%左右，请根据这些数值，将其降低到`2`或`1`左右。
* `--network_weights`
  * 在训练前加载预训练的LoRA权重，并在此基础上继续训练。
* `--network_train_unet_only`
  * 只启用与U-Net相关的LoRA模块。在fine tuning式的训练中指定可能比较好。
* `--network_train_text_encoder_only`
  * 只启用与Text Encoder相关的LoRA模块。可能会期待Textual Inversion式的效果。
* `--unet_lr`
  * 在与U-Net相关的LoRA模块中使用与通常的学习率（通过`--learning_rate`选项指定）不同的学习率时指定。
* `--text_encoder_lr`
  * 在与Text Encoder相关的LoRA模块中使用与通常的学习率（通过`--learning_rate`选项指定）不同的学习率时指定。Text Encoder的学习率稍微低一些（5e-5等）可能比较好。
* `--network_args`
  * 可以指定多个参数，后述。
* `--alpha_mask`
  * 将图像的alpha值用作掩码。在训练透明图像时使用。[PR #1223](https://github.com/kohya-ss/sd-scripts/pull/1223)

当`--network_train_unet_only`和`--network_train_text_encoder_only`都没有指定（默认）时，Text Encoder和U-Net的LoRA模块都会被启用。

# 其他训练方法

## 训练LoRA-C3Lier

请按如下方式指定`--network_args`。用`conv_dim`指定Conv2d (3x3)的rank，用`conv_alpha`指定alpha。

```
--network_args "conv_dim=4" "conv_alpha=1"
```

在以下示例中，省略`alpha`时默认为1。

```
--network_args "conv_dim=4"
```

## DyLoRA

DyLoRA是[这篇论文](https://arxiv.org/abs/2210.07558)中提出的。官方实现在[这里](https://github.com/huawei-noah/KD-NLP/tree/main/DyLoRA)。

根据论文，LoRA的rank不一定是越高越好，需要根据目标模型、数据集、任务等寻找合适的rank。使用DyLoRA，可以同时训练多个rank的LoRA，从而省去分别训练和寻找最佳rank的麻烦。

本仓库的实现基于官方实现，并进行了一些扩展（因此可能存在bug等）。

### 本仓库中DyLoRA的特点

训练后的DyLoRA模型文件与LoRA兼容。此外，可以从模型文件中提取出多个dim的LoRA。

可以训练DyLoRA-LierLa和DyLoRA-C3Lier。

### 使用DyLoRA训练

指定`--network_module=networks.dylora`，像这样使用DyLoRA对应的`network.dylora`。

此外，通过`--network_args`指定`unit`，例如`--network_args "unit=4"`。`unit`是分割rank的单位。例如，指定`--network_dim=16 --network_args "unit=4"`。`unit`应该是`network_dim`的因数（`network_dim`是`unit`的倍数）。

如果不指定`unit`，则被视为`unit=1`。

示例如下。

```
--network_module=networks.dylora --network_dim=16 --network_args "unit=4"

--network_module=networks.dylora --network_dim=32 --network_alpha=16 --network_args "unit=4"
```

对于DyLoRA-C3Lier，通过`--network_args`指定`conv_dim`，例如`"conv_dim=4"`。与通常的LoRA不同，`conv_dim`需要与`network_dim`相同。示例如下。

```
--network_module=networks.dylora --network_dim=16 --network_args "conv_dim=16" "unit=4"

--network_module=networks.dylora --network_dim=32 --network_alpha=16 --network_args "conv_dim=32" "conv_alpha=16" "unit=8"
```

例如，当dim=16，unit=4时，可以训练和提取4、8、12、16四个rank的LoRA。通过比较不同rank的LoRA生成的图像，可以选择最佳的rank。

其他选项与通常的LoRA相同。

※ `unit`是本仓库的独有扩展。在DyLoRA中，与相同dim的通常LoRA相比，训练时间可能会更长，因此采用了更大的分割单位。

### 从DyLoRA模型中提取LoRA模型

使用`networks`文件夹中的`extract_lora_from_dylora.py`。按照指定的`unit`单位，从DyLoRA模型中提取LoRA模型。

命令行示例：

```powershell
python networks\extract_lora_from_dylora.py --model "foldername/dylora-model.safetensors" --save_to "foldername/dylora-model-split.safetensors" --unit 4
```

`--model`指定DyLoRA模型文件，`--save_to`指定提取的LoRA模型的保存文件名（将在文件名后附加rank数值），`--unit`指定DyLoRA训练时的`unit`。

## 层次别学习率

详情请参阅[PR #355](https://github.com/kohya-ss/sd-scripts/pull/355)。

可以为全部25个full模型的block指定权重。第一个block对应的LoRA不存在，但为了与层次别LoRA应用等保持兼容性，指定为25个。此外，即使不扩展到conv2d3x3，一部分block的LoRA也不存在，但为了统一描述，始终指定25个值。

在SDXL中，请指定down/up 9个，middle 3个数值。

通过`--network_args`指定以下参数。

- `down_lr_weight`：指定U-Net的down blocks的学习率权重。可以指定：
  - 每个block的权重：例如`"down_lr_weight=0,0,0,0,0,0,1,1,1,1,1,1"`，指定12个数值（SDXL中为9个）。
  - 从预设指定：例如`"down_lr_weight=sine"`，使用sine曲线指定权重。可以指定sine、cosine、linear、reverse_linear、zeros。此外，通过添加`+数值`（例如`"down_lr_weight=cosine+.25"`），可以加上指定的数值（范围将是0.25~1.25）。
- `mid_lr_weight`：指定U-Net的mid block的学习率权重。例如`"mid_lr_weight=0.5"`，指定一个数值（SDXL中为3个）。
- `up_lr_weight`：指定U-Net的up blocks的学习率权重。与down_lr_weight相同。
- 省略的部分将被视为1.0。如果权重为0，则不会创建该block的LoRA模块。
- `block_lr_zero_threshold`：如果权重小于或等于此值，则不会创建LoRA模块。默认为0。

### 层次别学习率命令行示例：

```powershell
--network_args "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5" "mid_lr_weight=2.0" "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5"

--network_args "block_lr_zero_threshold=0.1" "down_lr_weight=sine+.5" "mid_lr_weight=1.5" "up_lr_weight=cosine+.5"
```

### 层次别学习率toml文件示例：

```toml
network_args = [ "down_lr_weight=0.5,0.5,0.5,0.5,1.0,1.0,1.0,1.0,1.5,1.5,1.5,1.5", "mid_lr_weight=2.0", "up_lr_weight=1.5,1.5,1.5,1.5,1.0,1.0,1.0,1.0,0.5,0.5,0.5,0.5",]

network_args = [ "block_lr_zero_threshold=0.1", "down_lr_weight=sine+.5", "mid_lr_weight=1.5", "up_lr_weight=cosine+.5", ]
```

## 层次别dim (rank)

可以为全部25个full模型的block指定dim (rank)。与层次别学习率类似，一部分block的LoRA可能不存在，但始终指定25个值。

在SDXL中，请指定23个数值。由于与`sdxl_train.py`的[层次别学习率](./train_SDXL.md)兼容，因此一部分block的LoRA不存在。对应关系为`0: time/label embed, 1-9: input blocks 0-8, 10-12: mid blocks 0-2, 13-21: output blocks 0-8, 22: out`。

通过`--network_args`指定以下参数。

- `block_dims`：指定每个block的dim (rank)。例如`"block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"`，指定25个数值。
- `block_alphas`：指定每个block的alpha。与block_dims类似，指定25个数值。省略时使用`network_alpha`的值。
- `conv_block_dims`：将LoRA扩展到Conv2d 3x3时，指定每个block的dim (rank)。
- `conv_block_alphas`：将LoRA扩展到Conv2d 3x3时，指定每个block的alpha。省略时使用`conv_alpha`的值。

### 层次别dim (rank)命令行示例：

```powershell
--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "conv_block_dims=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"

--network_args "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2" "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2"
```

### 层次别dim (rank)toml文件示例：

```toml
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2",]
  
network_args = [ "block_dims=2,4,4,4,8,8,8,8,12,12,12,12,16,12,12,12,12,8,8,8,8,4,4,4,2", "block_alphas=2,2,2,2,4,4,4,4,6,6,6,6,8,6,6,6,6,4,4,4,4,2,2,2,2",]
```

# 其他脚本

与LoRA相关的脚本群，如合并等。

## 关于合并脚本

使用merge_lora.py可以将LoRA的训练结果与Stable Diffusion模型合并，或将多个LoRA模型合并。

对于SDXL，有sdxl_merge_lora.py可用。选项等相同，因此请将以下merge_lora.py替换为sdxl_merge_lora.py。

### 将LoRA模型与Stable Diffusion模型合并

合并后的模型可以像普通的Stable Diffusion的ckpt一样处理。例如，以下是命令行示例。

```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors --ratios 0.8
```

使用Stable Diffusion v2.x模型进行训练并合并时，请指定--v2选项。

通过--sd_model选项指定要合并的Stable Diffusion模型文件（目前仅支持.ckpt或.safetensors，不支持Diffusers）。

通过--save_to选项指定合并后的模型的保存路径（根据扩展名自动判断是.ckpt还是.safetensors）。

通过--models指定训练好的LoRA模型文件。可以指定多个，按顺序合并。

通过--ratios指定每个模型的适用率（即模型权重对原模型的反映程度），取值为0~1.0。例如，如果过拟合，可以尝试降低适用率来改善。模型的数量应与指定的模型数量相同。

多个模型时的示例：

```
python networks\merge_lora.py --sd_model ..\model\model.ckpt 
    --save_to ..\lora_train1\model-char1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors --ratios 0.8 0.5
```

### 合并多个LoRA模型

指定--concat选项，可以简单地将多个LoRA合并成一个新的LoRA模型。文件大小（以及dim/rank）将是指定LoRA的总大小（如果要在合并时改变dim (rank)，请使用`svd_merge_lora.py`）。

例如，以下是命令行示例。

```
python networks\merge_lora.py --save_precision bf16 
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors 
    --ratios 1.0 -1.0 --concat --shuffle
```

指定--concat选项。

此外，添加--shuffle选项，权重将被打乱。如果不打乱，合并后的LoRA可以被分解回原来的LoRA，因此在拷贝学习等情况下，可能会导致原始训练数据被暴露。请注意。

通过--save_to选项指定合并后的LoRA模型的保存路径（根据扩展名自动判断是.ckpt还是.safetensors）。

通过--models指定训练好的LoRA模型文件。可以指定三个以上。

通过--ratios指定每个模型的比率（即模型权重对原模型的反映程度），取值为0~1.0。将两个模型一对一合并时，比率应为“0.5 0.5”。如果使用“1.0 1.0”，则总权重将过大，结果可能不理想。

使用v1训练的LoRA和v2训练的LoRA、或者rank（维度数）不同的LoRA不能合并。U-Net alone的LoRA和U-Net+Text Encoder的LoRA理论上可以合并，但结果未知。

### 其他选项

* `precision`
  * 可以指定合并计算时的精度，有float、fp16、bf16可选。省略时为了确保精度默认为float。如果要减少内存使用量，可以指定fp16/bf16。
* `save_precision`
  * 可以指定模型保存时的精度，有float、fp16、bf16可选。省略时与precision相同。

还有其他一些选项，请使用--help查看。

## 合并多个rank不同的LoRA模型

使用`svd_merge_lora.py`将多个LoRA近似为一个LoRA（无法完全再现）。例如，以下是命令行示例。

```
python networks\svd_merge_lora.py 
    --save_to ..\lora_train1\model-char1-style1-merged.safetensors 
    --models ..\lora_train1\last.safetensors ..\lora_train2\last.safetensors 
    --ratios 0.6 0.4 --new_rank 32 --device cuda
```

`merge_lora.py`的主要选项相同。以下选项是新增的。

- `--new_rank`
  * 指定创建的LoRA的rank。
- `--new_conv_rank`
  * 指定创建的 Conv2d 3x3 LoRA 的 rank。省略时与 `new_rank` 相同。
- `--device`
  * 指定 `--device cuda` 以在GPU上进行计算。处理速度更快。

## 使用本仓库中的图像生成脚本生成

在gen_img_diffusers.py中添加`--network_module`和`--network_weights`选项。含义与训练时相同。

通过`--network_mul`选项指定0~1.0的数值，可以改变LoRA的应用率。

## 在Diffusers的pipeline中生成

请参考以下示例。只需要networks/lora.py文件。Diffusers的版本可能需要是0.10.2。

```python
import torch
from diffusers import StableDiffusionPipeline
from networks.lora import LoRAModule, create_network_from_weights
from safetensors.torch import load_file

# 如果ckpt是基于CompVis的，请事先使用tools/convert_diffusers20_original_sd.py转换为Diffusers。有关详细信息，请参阅--help。

model_id_or_dir = r"model_id_on_hugging_face_or_dir"
device = "cuda"

# 创建管道
print(f"从{model_id_or_dir}创建管道...")
pipe = StableDiffusionPipeline.from_pretrained(model_id_or_dir, revision="fp16", torch_dtype=torch.float16)
pipe = pipe.to(device)
vae = pipe.vae
text_encoder = pipe.text_encoder
unet = pipe.unet

# 加载LoRA网络
print(f"加载LoRA网络...")

lora_path1 = r"lora1.safetensors"
sd = load_file(lora_path1)   # 如果文件是.ckpt，请使用torch.load。
network1, sd = create_network_from_weights(0.5, None, vae, text_encoder,unet, sd)
network1.apply_to(text_encoder, unet)
network1.load_state_dict(sd)
network1.to(device, dtype=torch.float16)

# # 您也可以合并权重，而不是apply_to+load_state_dict。network.set_multiplier不起作用
# network.merge_to(text_encoder, unet, sd)

lora_path2 = r"lora2.safetensors"
sd = load_file(lora_path2) 
network2, sd = create_network_from_weights(0.7, None, vae, text_encoder,unet, sd)
network2.apply_to(text_encoder, unet)
network2.load_state_dict(sd)
network2.to(device, dtype=torch.float16)

lora_path3 = r"lora3.safetensors"
sd = load_file(lora_path3)
network3, sd = create_network_from_weights(0.5, None, vae, text_encoder,unet, sd)
network3.apply_to(text_encoder, unet)
network3.load_state_dict(sd)
network3.to(device, dtype=torch.float16)

# 提示
prompt = "杰作，最佳质量，1个女孩，穿着白衬衫，看着观众"
negative_prompt = "糟糕的质量，最差的质量，糟糕的解剖结构，糟糕的手"

# 执行管道
print("生成图像...")
with torch.autocast("cuda"):
    image = pipe(prompt, guidance_scale=7.5, negative_prompt=negative_prompt).images[0]

# 如果未合并，您可以使用set_multiplier
# network1.set_multiplier(0.8)
# 并再次生成图像...

# 保存图像
image.save(r"由Diffusers生成的图像.png")
```

## 从两个模型差异创建LoRA模型

这是参考[这个讨论](https://github.com/cloneofsimo/lora/discussions/56)实现的。公式直接使用（似乎使用奇异值分解进行近似）。

将两个模型（例如fine tuning前的模型和fine tuning后的模型）的差异近似为LoRA。

### 脚本执行方法

请按如下方式指定。

```
python networks\extract_lora_from_models.py --model_org base-model.ckpt
    --model_tuned fine-tuned-model.ckpt 
    --save_to lora-weights.safetensors --dim 4
```

通过`--model_org`选项指定原始的Stable Diffusion模型。应用创建的LoRA模型时，将使用此模型。.ckpt或.safetensors可以指定。

通过`--model_tuned`选项指定要提取差异的目标Stable Diffusion模型。例如，fine tuning或DreamBooth后的模型。.ckpt或.safetensors可以指定。

通过`--save_to`指定LoRA模型的保存路径。`--dim`指定LoRA的维度数。

生成的LoRA模型可以像训练好的LoRA模型一样使用。

如果两个模型的Text Encoder相同，则LoRA将仅为U-Net的LoRA。

### 其他选项

- `--v2`
  * 使用v2.x的Stable Diffusion模型时指定。
- `--device`
  * 指定`--device cuda`以在GPU上进行计算。处理速度更快（CPU下也不太慢，大约是2~数倍）。
- `--save_precision`
  * 指定LoRA的保存格式，有"float"、"fp16"、"bf16"可选。省略时默认为float。
- `--conv_dim`
  * 指定时将LoRA的应用范围扩展到Conv2d 3x3。指定Conv2d 3x3的rank。

## 图像大小调整脚本

（稍后将整理文档，目前先在这里写说明。）

作为Aspect Ratio Bucketing的功能扩展，可以将较小的图像保持原样作为训练数据。此外，收到将原始训练图像缩小后的图像添加到训练数据的报告和预处理脚本，因此进行了整理和添加。感谢bmaltais。

### 脚本执行方法

请按如下方式指定。原始图像和调整大小后的图像将被保存到转换后的文件夹中。调整大小后的图像的文件名将附加调整大小后的分辨率（例如`+512x512`）。如果调整大小后的分辨率小于原始图像，则不会被放大。

```
python tools\resize_images_to_resolution.py --max_resolution 512x512,384x384,256x256 --save_as_png 
    原始图像文件夹 转换后的文件夹
```

原始图像文件夹中的图像文件将被调整大小，以使其面积与指定的分辨率（可指定多个）相同，并保存到转换后的文件夹中。非图像文件将保持不变。

通过`--max_resolution`选项指定调整大小后的大小，如示例所示。面积将被调整为该大小。指定多个时，将分别调整大小为每个分辨率。例如，`512x512,384x384,256x256`将在转换后的文件夹中生成原始大小和调整大小后的大小×3，共4张图像。

通过`--save_as_png`选项指定以png格式保存。省略时将以jpeg格式（quality=100）保存。

通过`--copy_associated_files`选项指定时，将与图像文件名（除扩展名外）相同的其他文件（例如caption）以调整大小后的图像的文件名复制。

### 其他选项

- `divisible_by`
  * 调整大小后的图像大小（高度和宽度）将通过裁剪图像中心，使其可以被该值整除。
- `interpolation`
  * 指定缩小时的插值方法。可选`area`、`cubic`、`lanczos4`，默认为`area`。

# 附加信息

## 与cloneofsimo仓库的区别

截至2022/12/25，本仓库将LoRA的应用范围扩展到Text Encoder的MLP、U-Net的FFN和Transformer的in/out projection，从而增强了表示能力。但代价是内存使用量增加，8GB VRAM勉强够用。

此外，模块替换机制完全不同。

## 关于未来扩展

不仅限于LoRA，还可以支持其他扩展。未来将考虑添加这些扩展。