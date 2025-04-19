由于文档正在更新中，可能存在描述错误。

# 关于学习，通用篇

本仓库支持对模型的 fine tuning、DreamBooth，以及 LoRA 和 Textual Inversion（包括 [XTI:P+](https://github.com/kohya-ss/sd-scripts/pull/327)）的学习。本文档将介绍它们共通的学习数据准备方法和选项等。

# 概要

在开始之前，请参考本仓库的 README 进行环境设置。

本文档将介绍以下内容：

1. 学习数据的准备方法（使用设置文件的新格式）
2. 学习中使用的术语的简要说明
3. 之前的指定格式（不使用设置文件，直接从命令行指定）
4. 在学习过程中生成示例图像
5. 各个脚本中共通的、常用的选项
6. fine tuning 方式的元数据准备：标注等

1.只需执行步骤1就可以开始学习了（学习详情请参考各脚本的文档）。2.步骤2及之后的内容请根据需要参考。

# 关于学习数据的准备

请在任意文件夹（可以是多个文件夹）中准备学习数据的图像文件。支持 `.png`、`.jpg`、`.jpeg`、`.webp`、`.bmp` 格式。基本上不需要进行预处理，如调整大小等。

但是，建议不要使用比学习分辨率（后述）明显小的图像，或者事先使用超分辨率 AI 等进行放大。另外，极大的图像（约3000x3000像素或更大）可能会导致错误，因此请事先缩小。

在学习时，需要整理要让模型学习的图片数据，并将其指定给脚本。根据图片数据的数量、学习对象、是否有标题（图片的说明）等，可以通过几种方法指定学习数据。以下是可用的方法（每个名称并不是通用的，而是本仓库的定义）。关于正则化图片将在后面描述。

1. DreamBooth、class+identifier 方式（可使用正则化图像）

    通过特定单词（identifier）与学习对象关联来进行学习。不需要准备标题。例如，用于学习特定角色时，由于不需要准备标题，因此比较方便。但是，由于头发样式、服装、背景等学习数据中的所有元素都与 identifier 关联，因此在生成图像时可能无法更改服装等。

2. DreamBooth、标题方式（可使用正则化图像）

    为每张图像准备记录标题的文本文件来进行学习。例如，用于学习特定角色时，通过在标题中描述图像的细节（穿白衣服的角色A、穿红衣服的角色A等），可以将角色与其他元素分离，从而更严格地学习角色。

3. fine tuning 方式（不可使用正则化图像）

    事先将标题汇总到元数据文件中。支持将标签和标题分开管理、预先缓存latents以加快学习速度等功能（均在其他文档中说明）。（虽然称为fine tuning方式，但不仅限于fine tuning）。

可用于学习的对象或方法与可用的指定方法的组合如下：

| 学习对象或方法         | 脚本                         | DB / class+identifier | DB / 标题 | fine tuning |
| ---------------------- | ---------------------------- | --------------------- | --------- | ----------- |
| 对模型进行 fine tuning | `fine_tune.py`               | x                     | x         | o           |
| 对模型进行 DreamBooth  | `train_db.py`                | o                     | o         | x           |
| LoRA                   | `train_network.py`           | o                     | o         | o           |
| Textual Inversion      | `train_textual_inversion.py` | o                     | o         | o           |

## 选择哪种方式

对于 LoRA 和 Textual Inversion，若想简单地进行学习而不准备标题文件，则可以使用 DreamBooth class+identifier 方式；若能准备标题文件，则可以使用 DreamBooth 标题方式。若学习数据量大且不使用正则化图像，则也可以考虑 fine tuning 方式。

对于 DreamBooth 也是同样，但不能使用 fine tuning 方式。fine tuning 的情况下，只能使用 fine tuning 方式。

# 关于各种指定方法的说明

这里只介绍每种指定方法的典型模式。更详细的指定方法请参考 [数据集设置](./config.md)。

# DreamBooth、class+identifier 方式（可使用正则化图像）

在这种方式中，每张图像都与 `class identifier` 这样的标题一起学习（例如 `shs dog`）。

## 步骤1：决定 identifier 和 class

确定要与学习对象关联的单词 identifier 和对象所属的 class。

（虽然有各种不同的称呼，但这里按照原始论文的说法）。

下面简要说明（详情请自行查找）。

class 是学习对象的一般类别。例如，若要学习特定的犬种，则 class 为 dog。若是动漫角色，则 class 可能是 boy 或 girl、1boy 或 1girl，取决于模型。

identifier 是用于区分学习对象的任意单词，但根据原始论文，建议使用“tokenizer中为1个token的3个字符以下、且较少见的单词”。

使用 identifier 和 class，例如通过“shs dog”来学习模型，可以将学习对象与 class 区分开来。

在生成图像时，使用“shs dog”即可生成所学犬种的图像。

（我最近使用的 identifier 例子有 ``shs sts scs cpc coc cic msm usu ici lvl cic dii muk ori hru rik koo yos wny`` 等。实际上，最好使用 Danbooru Tag 中不包含的单词）。

## 步骤2：决定是否使用正则化图像，若使用，则生成正则化图像

正则化图像是防止上述 class 被学习对象影响的图像（language drift）。若不使用正则化图像，则例如使用 `shs 1girl` 学习特定角色时，即使仅使用 `1girl` 作为提示，也会生成与该角色相似的图像。这是因为 `1girl` 包含在学习的标题中。

通过同时学习学习对象的图像和正则化图像，可以保持 class 的原样，仅在提示中使用 identifier 时生成学习对象。

若仅需在 LoRA 或 DreamBooth 中生成特定角色，则可以不使用正则化图像。

对于 Textual Inversion，可能也不需要使用正则化图像（因为要学习的 token string 不包含在标题中，因此不会被学习）。

作为正则化图像，通常使用学习对象模型通过 class 名称生成的图像（例如 `1girl`）。但若生成的图像质量较差，也可以通过改进提示或从网络下载其他图像来使用。

（由于正则化图像也会被学习，因此其质量会影响模型）。

通常，建议准备数百张左右的正则化图像（若数量较少，则 class 图像不会被一般化，从而会学习到它们的特征）。

使用生成的图像时，通常需要将其大小调整为学习分辨率（更准确地说，是 bucket 的分辨率，后述）。

## 步骤 2. 编写配置文件

创建一个文本文件，将其扩展名改为 `.toml`。例如，可以按照以下方式编写：

（以 `#` 开头的部分是注释，因此可以直接复制粘贴，也可以删除，不影响使用。）

```toml
[general]
enable_bucket = true                        # 是否使用宽高比桶

[[datasets]]
resolution = 512                            # 训练分辨率
batch_size = 4                              # 批次大小

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 指定包含训练图像的文件夹
  class_tokens = 'hoge girl'                # 指定标识符类
  num_repeats = 10                          # 训练图像的重复次数

  # 以下仅在需要正则化图像时描述。如果不使用，请删除
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # 指定包含正则化图像的文件夹
  class_tokens = 'girl'                     # 指定类
  num_repeats = 1                           # 正则化图像的重复次数，基本上为1即可
```

基本上，只需更改以下位置即可进行训练：

1. 训练分辨率

    指定一个数值时，为正方形（例如 `512` 即为 512x512），用方括号和逗号分隔两个数值时，为宽 x 高（例如 `[512,768]` 即为 512x768）。在 SD1.x 系列中，原始训练分辨率为 512。指定较大的分辨率（如 `[512,768]`）可能有助于减少生成纵向或横向图像时的崩溃。SD2.x 768 系列中，分辨率为 `768`。

2. 批次大小

    指定同时训练多少数据。根据 GPU 的显存大小和训练分辨率而有所不同。详情请参见后文。此外，根据微调/DreamBooth/LoRA 等的不同，也会有所不同，请参阅各脚本的说明。

3. 文件夹指定

    指定训练图像和正则化图像（如果使用）所在的文件夹。直接指定包含图像数据的文件夹。

4. 标识符和类的指定

    如前面的示例所示。

5. 重复次数

    在后文中描述。

### 关于重复次数

重复次数用于调整正则化图像的数量和训练图像的数量。由于正则化图像的数量通常多于训练图像，因此通过重复训练图像来匹配数量，使其以 1:1 的比例进行训练。

请指定重复次数，使得 **训练图像的重复次数 × 训练图像的数量 ≥ 正则化图像的重复次数 × 正则化图像的数量**。

（1 个 epoch（数据循环一次为 1 个 epoch）的数据数量为「训练图像的重复次数 × 训练图像的数量」。如果正则化图像的数量超过此数量，多余的正则化图像将不被使用。）

## 步骤 3. 进行训练

请参考各自的文档进行训练。

# DreamBooth、字幕方式（可使用正则化图像）

在这种方式中，每张图像都通过字幕进行训练。

## 步骤 1. 准备字幕文件

在包含训练图像的文件夹中，为每张图像创建一个具有相同文件名的文件，扩展名为 `.caption`（可通过设置更改）。每个文件应仅包含一行文本。编码为 `UTF-8`。

## 步骤 2. 确定是否使用正则化图像，并生成正则化图像（如果使用）

与类+标识符形式相同。同样，可以为正则化图像添加字幕，但通常不需要。

## 步骤 2. 编写配置文件

创建一个文本文件，将其扩展名改为 `.toml`。例如，可以按照以下方式编写：

```toml
[general]
enable_bucket = true                        # 是否使用宽高比桶

[[datasets]]
resolution = 512                            # 训练分辨率
batch_size = 4                              # 批次大小

  [[datasets.subsets]]
  image_dir = 'C:\hoge'                     # 指定包含训练图像的文件夹
  caption_extension = '.caption'            # 字幕文件的扩展名。如果使用 .txt，请更改此处
  num_repeats = 10                          # 训练图像的重复次数

  # 以下仅在需要正则化图像时描述。如果不使用，请删除
  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'                      # 指定包含正则化图像的文件夹
  class_tokens = 'girl'                     # 指定类
  num_repeats = 1                           # 正则化图像的重复次数，基本上为1即可
```

基本上，只需更改以下位置即可进行训练。没有特别说明的部分与类+标识符方式相同。

1. 训练分辨率
2. 批次大小
3. 文件夹指定
4. 字幕文件的扩展名

    可以指定任意扩展名。
5. 重复次数

## 步骤 3. 进行训练

请参考各自的文档进行训练。

# 微调方式

## 步骤 1. 准备元数据

将字幕和标签汇总的管理文件称为元数据。采用 json 格式，扩展名为 `.json`。创建方法较长，已在本文档末尾说明。
## 步骤2. 编写设定文件

创建一个文本文件，并将其扩展名改为 `.toml`。例如，可以这样编写：

```toml
[general]
shuffle_caption = true
keep_tokens = 1

[[datasets]]
resolution = 512                                    # 学习分辨率
batch_size = 4                                      # 批次大小

  [[datasets.subsets]]
  image_dir = 'C:\piyo'                             # 指定包含学习用图像的文件夹
  metadata_file = 'C:\piyo\piyo_md.json'            # 元数据文件名
```

基本上，只需更改以下位置即可进行学习。没有特别说明的部分与DreamBooth、class+identifier方式相同。

1. 学习分辨率
2. 批次大小
3. 文件夹指定
4. 元数据文件名

    指定通过后述方法创建的元数据文件。


## 步骤3. 开始学习

请参考各自的文档进行学习。

# 学习中使用的术语简要说明

这里省略了细节，我也不是完全理解，请自行查阅相关资料。

## 微调（fine tuning）

指对模型进行学习和微调。在Stable Diffusion中，狭义的fine tuning是指用图像和标题对模型进行学习。DreamBooth可以说是狭义的fine tuning的一种特殊方式。广义的fine tuning包括LoRA、Textual Inversion、Hypernetworks等，涵盖了所有模型学习的范畴。

## 步骤（step）

粗略地说，用学习数据计算一次就是一个步骤。“将学习数据的标题输入到当前模型中，生成图像并将其与学习数据的图像进行比较，然后对模型进行微小的调整，使其更接近学习数据”，这就是一个步骤。

## 批次大小（batch size）

批次大小是指在一个步骤中同时计算的数据数量。批量计算可以相对提高速度，而且一般认为精度也会提高。

`批次大小 × 步骤数` 等于用于学习的数据数量。因此，如果增加批次大小，最好相应地减少步骤数。

（但是，例如“批次大小为1，1600步”和“批次大小为4，400步”不会得到相同的结果。如果学习率相同，一般来说后者会导致学习不足。可以稍微提高学习率（例如 `2e-6`），或者将步骤数增加到500步等进行调整。）

增加批次大小会相应地消耗更多的GPU内存。如果内存不足，会导致错误，而且即使没有错误，也会降低学习速度。可以通过任务管理器或 `nvidia-smi` 命令查看内存使用情况进行调整。

顺便说一下，“batch”大致意味着“一批数据”。

## 学习率（learning rate）

粗略地说，学习率表示每个步骤模型变化的程度。指定较大的值可以加快学习速度，但也可能导致模型过度变化而损坏，或者无法达到最佳状态。指定较小的值会减慢学习速度，也可能无法达到最佳状态。

fine tuning、DreamBooth、LoRA的学习率各不相同，而且还会受到学习数据、模型、批次大小和步骤数的影响。可以从一般值开始，根据学习状态进行调整。

默认情况下，整个学习过程中的学习率是固定的。可以通过指定调度器来决定学习率如何变化，这也会影响结果。

## 周期（epoch）

当学习数据被完整地学习一遍（数据循环一遍）时，就称为一个epoch。如果指定了重复次数，那么在重复后的数据循环一遍也称为一个epoch。

一个epoch的步骤数基本上是 `数据数量 ÷ 批次大小`，但在使用Aspect Ratio Bucketing时，步骤数会略微增加（因为不同bucket的数据不能放在同一个批次中）。

## 宽高比分桶（Aspect Ratio Bucketing）

Stable Diffusion v1是在512×512分辨率下训练的，但除了这个分辨率，还会在256×1024、384×640等其他分辨率下进行训练。这样可以减少图像的裁剪，更准确地学习标题和图像之间的关系。

此外，由于可以在任意分辨率下进行训练，因此无需事先统一图像数据的宽高比。

可以通过设置启用或禁用该功能，在上述设定文件的例子中是启用的（设置为 `true`）。

学习分辨率会在不超过给定参数的面积（即内存使用量）的范围内，以64像素为单位（默认，可更改）进行调整和生成。

在机器学习中，通常需要统一输入数据的大小，但实际上，只要在同一个批次内保持一致即可。NovelAI所说的bucketing似乎是指事先根据学习数据的宽高比将其分类到不同的学习分辨率中，然后在每个bucket内创建批次，以保持批次内图像大小的一致。

# 旧的指定形式（不使用设定文件，直接通过命令行指定）

这里介绍不使用 `.toml` 文件，而是通过命令行选项进行指定的方法。有DreamBooth class+identifier方式、DreamBooth标题方式和fine tuning方式。

## DreamBooth、class+identifier方式

通过文件夹名称指定重复次数，并使用 `train_data_dir` 选项和 `reg_data_dir` 选项。

### 步骤1. 准备学习用图像

创建一个文件夹用于存放学习用图像。 __然后在该文件夹内__，按照以下格式创建子文件夹：

```
<重复次数>_<identifier> <class>
```

注意不要忘记中间的 `_`。

例如，如果要使用“sls frog”作为提示词，并重复数据20次，则文件夹名为“20_sls frog”。如下所示：

![image](https://user-images.githubusercontent.com/52813779/210770636-1c851377-5936-4c15-90b7-8ac8ad6c2074.png)
### 多类别、多对象（identifier）学习

方法很简单，只需在训练图像文件夹内创建多个名为 ``重复次数_<identifier> <class>`` 的文件夹，并在正则化图像文件夹中创建多个名为 ``重复次数_<class>`` 的文件夹。

例如，要同时学习“sls frog”和“cpc rabbit”，可以按如下方式设置：

![image](https://user-images.githubusercontent.com/52813779/210777933-a22229db-b219-4cd8-83ca-e87320fc4192.png)

如果类别只有一个，但对象有多个，则正则化图像文件夹可以只有一个。例如，如果1girl有两个角色A和B，可以按如下方式设置：

- train_girls
  - 10_sls 1girl
  - 10_cpc 1girl
- reg_girls
  - 1_1girl

### 步骤2：准备正则化图像

使用正则化图像的步骤如下：

创建一个存储正则化图像的文件夹，并在其中创建一个名为 ``<重复次数>_<class>`` 的目录。

例如，如果使用“frog”作为提示词，并且数据不重复（只重复一次），则可以按如下方式设置：

![image](https://user-images.githubusercontent.com/52813779/210770897-329758e5-3675-49f1-b345-c135f1725832.png)

### 步骤3：执行训练

运行各个训练脚本。使用 `--train_data_dir` 选项指定上述训练数据的文件夹（__不是包含图像的文件夹，而是其父文件夹__），使用 `--reg_data_dir` 选项指定正则化图像的文件夹（__不是包含图像的文件夹，而是其父文件夹__）。

## DreamBooth、标题方式

在训练图像和正则化图像的文件夹中，如果存在与图像文件同名、扩展名为.caption（可通过选项更改）的文件，则会从该文件中读取标题并作为提示词进行训练。

※这些图像的训练将不再使用文件夹名（identifier class）。

标题文件的扩展名默认为.caption。可以使用训练脚本的 `--caption_extension` 选项进行更改。使用 `--shuffle_caption` 选项可以在训练时对以逗号分隔的各个部分进行随机排序。

## 微调方式

直到创建元数据的过程都与使用配置文件时相同。使用 `in_json` 选项指定元数据文件。

# 训练过程中的示例输出

可以使用训练中的模型尝试生成图像，以确认训练进度。在训练脚本中指定以下选项：

- `--sample_every_n_steps` / `--sample_every_n_epochs`
    
    指定每隔多少步骤或多少epoch进行一次示例输出。如果同时指定，则优先使用epoch。

- `--sample_at_first`
    
    在训练开始前进行示例输出。可以与训练前的结果进行比较。

- `--sample_prompts`

    指定示例输出提示词的文件。

- `--sample_sampler`

    指定用于示例输出的采样器。
    可选值为 `'ddim', 'pndm', 'heun', 'dpmsolver', 'dpmsolver++', 'dpmsingle', 'k_lms', 'k_euler', 'k_euler_a', 'k_dpm_2', 'k_dpm_2_a'`。

要进行示例输出，需要事先准备一个包含提示词的文本文件。每行写一个提示词。

例如，可以按如下方式设置：

```txt
# 提示词1
masterpiece, best quality, 1girl, in white shirts, upper body, looking at viewer, simple background --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 768 --h 768 --d 1 --l 7.5 --s 28

# 提示词2
masterpiece, best quality, 1boy, in business suit, standing at street, looking back --n low quality, worst quality, bad anatomy,bad composition, poor, low effort --w 576 --h 832 --d 2 --l 5.5 --s 40
```

以 `#` 开头的行是注释。以 `--` 开头的部分（如 `--n`）可以指定生成图像的选项。以下选项可用：

- `--n` 将之后的选项视为负面提示词，直到下一个 `--` 选项。
- `--w` 指定生成图像的宽度。
- `--h` 指定生成图像的高度。
- `--d` 指定生成图像的种子。
- `--l` 指定生成图像的CFG scale。
- `--s` 指定生成图像的步数。

# 各脚本中常见的常用选项

脚本更新后，文档可能没有及时更新。如果需要查看可用选项，请使用 `--help` 选项。

## 指定训练使用的模型

- `--v2` / `--v_parameterization`
    
    如果要使用Hugging Face的stable-diffusion-2-base或从它派生的微调模型（在推理时需要使用 `v2-inference.yaml` 的模型），请使用 `--v2` 选项。如果要使用stable-diffusion-2或768-v-ema.ckpt及其微调模型（在推理时需要使用 `v2-inference-v.yaml` 的模型），请同时使用 `--v2` 和 `--v_parameterization` 选项。

    Stable Diffusion 2.0的主要变化如下：

    1. 使用的Tokenizer
    2. 使用的Text Encoder以及使用的输出层（2.0使用倒数第二层）
    3. Text Encoder的输出维度（768->1024）
    4. U-Net的结构（CrossAttention的头数等）
    5. v-parameterization（采样方法似乎已更改）

    其中，base版本采用了1～4的变化，而非base版本（768-v）采用了1～5的变化。`--v2` 选项启用1～4的变化，`--v_parameterization` 选项启用5的变化。

- `--pretrained_model_name_or_path` 
    
    指定用于进行额外训练的基础模型。可以指定Stable Diffusion的checkpoint文件（.ckpt或.safetensors）、Diffusers的本地磁盘上的模型目录或Diffusers的模型ID（例如"stabilityai/stable-diffusion-2"）。
## 训练设定

- `--output_dir`

    指定训练后的模型保存的文件夹。

- `--output_name`

    指定模型的文件名，不包含扩展名。

- `--dataset_config`

    指定包含数据集配置的 `.toml` 文件。

- `--max_train_steps` / `--max_train_epochs`

    指定训练的步数或 epoch 数。如果同时指定，则 epoch 数的优先级更高。

- `--mixed_precision`

    为了减少内存使用，以混合精度（mixed precision）进行训练。例如 `--mixed_precision="fp16"`。与不使用混合精度（默认）相比，精度可能会降低，但可以大幅减少训练所需的 GPU 内存。

    （对于 RTX30 系列及之后的显卡，也可以指定 `bf16`。请确保与使用 accelerate 进行的环境设置相一致）。

- `--gradient_checkpointing`

    不是一次性进行训练时的权重计算，而是分步进行，以减少训练所需的 GPU 内存量。开启或关闭对精度没有影响，但开启后可以增加批大小，从而对训练结果产生影响。

    通常，开启后会降低训练速度，但由于可以增加批大小，总体的训练时间可能会更快。

- `--xformers` / `--mem_eff_attn`

    指定 xformers 选项时，使用 xformers 的 CrossAttention。如果未安装 xformers 或出现错误（在某些环境下，如 `mixed_precision="no"` 时），可以指定 `mem_eff_attn` 选项以使用节省内存版本的 CrossAttention（比 xformers 慢）。

- `--clip_skip`

    指定 `2` 时，使用 Text Encoder (CLIP) 的倒数第二层的输出。指定 1 或省略该选项时，使用最后一层的输出。

    ※SD2.0 默认使用倒数第二层的输出，因此在 SD2.0 的训练中不要指定此选项。

    如果训练的模型原本就是使用倒数第二层进行训练的，那么指定 2 是合适的。

    如果原本使用最后一层进行训练，由于整个模型都是基于此进行训练的，因此如果改用倒数第二层进行训练，可能需要一定数量的训练数据和较长的训练时间才能获得理想的训练结果。

- `--max_token_length`

    默认值为 75。指定 `150` 或 `225` 可以扩展 token 长度进行训练。如果使用长标题进行训练，请指定此选项。

    但是，训练时的 token 扩展规范与 Automatic1111 的 Web UI 略有不同（例如，分词的规范），因此如果没有必要，建议使用默认的 75 进行训练。

    与 clip_skip 类似，如果训练长度与模型的原始训练状态不同，可能需要一定数量的训练数据和较长的训练时间。

- `--weighted_captions`

    指定此选项后，可以使用与 Automatic1111 的 Web UI 类似的加权标题。除“Textual Inversion 和 XTI”外的训练均可使用。不仅可以用于标题，也可以用于 DreamBooth 方法的 token 字符串。

    加权标题的语法与 Web UI 几乎相同，可以使用 (abc)、[abc]、(abc:1.23) 等。也可以嵌套使用。但请注意，括号内不能包含逗号，否则在 prompt 的 shuffle/dropout 时会导致括号匹配错误。

- `--persistent_data_loader_workers`

    在 Windows 环境中指定此选项可以大幅减少 epoch 之间的等待时间。

- `--max_data_loader_n_workers`

    指定数据加载的进程数。进程数越多，数据加载越快，GPU 利用率越高，但会消耗更多的主内存。默认值为 `8` 或 `CPU 同时执行线程数-1` 中的较小者。如果主内存不足或 GPU 使用率已经达到 90% 左右，可以根据实际情况降低此数值，例如设置为 `2` 或 `1`。

- `--logging_dir` / `--log_prefix`

    与训练日志保存相关的选项。`logging_dir` 选项指定日志保存的文件夹，将保存 TensorBoard 格式的日志。

    例如，指定 `--logging_dir=logs`，将在工作文件夹中创建一个 logs 文件夹，并在其下的日期文件夹中保存日志。

    如果指定 `--log_prefix` 选项，将在日期前添加指定的字符串。例如，可以使用 `--logging_dir=logs --log_prefix=db_style1_` 来标识日志。

    要在 TensorBoard 中查看日志，需要打开另一个命令提示符，在工作文件夹中输入以下命令：

    ```
    tensorboard --logdir=logs
    ```

    （tensorboard 应该在环境设置时已经安装。如果没有，可以使用 `pip install tensorboard` 安装。）

    然后打开浏览器，访问 http://localhost:6006/ 即可查看日志。

- `--log_with` / `--log_tracker_name`

    与训练日志保存相关的选项。除了 `tensorboard` 外，还支持保存到 `wandb`。详细信息请参见 [PR#428](https://github.com/kohya-ss/sd-scripts/pull/428)。

- `--noise_offset`

    实现参考：https://www.crosslabs.org//blog/diffusion-with-offset-noise

    似乎可以改善整体偏暗或偏亮的图像生成结果。在 LoRA 训练中似乎也有效。建议指定 `0.1` 左右的值。

- `--adaptive_noise_scale`（实验性选项）

    根据 latents 的各通道的平均值的绝对值自动调整 Noise offset 的值。与 `--noise_offset` 同时指定时生效。Noise offset 的值通过 `noise_offset + abs(mean(latents, dim=(2,3))) * adaptive_noise_scale` 计算。由于 latent 接近正态分布，因此建议指定 `noise_offset` 的 1/10 到相同量级的值。

    也可以指定负值，此时 Noise offset 将被限制在 0 以上。

- `--multires_noise_iterations` / `--multires_noise_discount`

    Multi resolution noise（pyramid noise）的设置。详细信息请参见 [PR#471](https://github.com/kohya-ss/sd-scripts/pull/471) 和 [Multi-Resolution Noise for Diffusion Model Training](https://wandb.ai/johnowhitaker/multires_noise/reports/Multi-Resolution-Noise-for-Diffusion-Model-Training--VmlldzozNjYyOTU2)。

    指定 `--multires_noise_iterations` 的数值即可启用。建议值为 6~10。`--multires_noise_discount` 的值可以设为 0.1~0.3（LoRA 训练等数据集较小的情况下的作者推荐值），或 0.8（原始文章的推荐值）（默认值为 0.3）。

- `--debug_dataset`

    添加此选项后，可以在开始训练前预览将要训练的图像数据和标题。按 Esc 键退出预览，按 `S` 键进入下一步（批次），按 `E` 键进入下一个 epoch。

    ※在 Linux 环境（包括 Colab）中，图像不会显示。

- `--vae`

    指定 Stable Diffusion 的 checkpoint、VAE 的 checkpoint 文件、Diffusers 的模型或 VAE（均可为本地文件或 Hugging Face 模型 ID），将使用指定的 VAE 进行训练（在 latents 缓存时或训练中的 latents 获取时）。

    在 DreamBooth 和 fine tuning 中，保存的模型将嵌入指定的 VAE。

- `--cache_latents` / `--cache_latents_to_disk`

    将 VAE 的输出缓存到主内存，以减少使用的 VRAM。启用后，除 `flip_aug` 外的其他数据增强方法将不可用。同时，整体训练速度会略微加快。
指定`cache_latents_to_disk`后，缓存将被保存到磁盘。即使脚本结束并再次启动，缓存仍然有效。

- `--min_snr_gamma`

    指定Min-SNR加权策略。详情请参考[此处](https://github.com/kohya-ss/sd-scripts/pull/308)。论文中推荐值为`5`。

## 模型保存设置

- `--save_precision`

    指定保存时的数据精度。如果指定`save_precision`选项为`float`、`fp16`或`bf16`，则以相应格式保存模型（在DreamBooth和fine tuning中以Diffusers格式保存模型时无效）。如果您想减小模型的大小，可以使用此选项。

- `--save_every_n_epochs` / `--save_state` / `--resume`

    如果为`save_every_n_epochs`选项指定一个数值，则每隔指定epoch数保存一次学习过程中的模型。

    如果同时指定`save_state`选项，则还会保存包括优化器状态在内的学习状态（与从保存的模型中恢复学习相比，可以预期精度提高和学习时间缩短）。保存目标是一个文件夹。
    
    学习状态将以`<output_name>-??????-state`（??????是epoch数）为名的文件夹保存在输出目录中。在长时间学习时请使用此功能。

    要从保存的学习状态恢复学习，请使用`resume`选项。需指定保存的学习状态文件夹（不是`output_dir`，而是其中的state文件夹）。

    请注意，由于Accelerator的限制，epoch数和global step不会被保存，恢复时也会从1开始。

- `--save_every_n_steps`

    如果为`save_every_n_steps`选项指定一个数值，则每隔指定step数保存一次学习过程中的模型。可以与`save_every_n_epochs`同时指定。

- `--save_model_as` （仅限DreamBooth和fine tuning）

    可以从`ckpt`、`safetensors`、`diffusers`、`diffusers_safetensors`中选择模型的保存格式。
    
    指定方式为`--save_model_as=safetensors`。如果以Stable Diffusion格式（ckpt或safetensors）读取并以Diffusers格式保存，则会从Hugging Face下载v1.5或v2.1的信息来补充缺失的信息。

- `--huggingface_repo_id` 等

    如果指定了`huggingface_repo_id`，则在保存模型时同时上传到HuggingFace。请注意访问令牌的处理（请参考HuggingFace的文档）。

    其他参数请按如下方式指定。

    -   `--huggingface_repo_id "your-hf-name/your-model" --huggingface_path_in_repo "path" --huggingface_repo_type model --huggingface_repo_visibility private --huggingface_token hf_YourAccessTokenHere`

    如果`huggingface_repo_visibility`指定为`public`，则仓库将公开。如果省略或指定为`private`（或其他非public值），则仓库将为私有。

    如果在指定`--save_state`时添加`--save_state_to_huggingface`，则还会上传state。

    如果在指定`--resume`时添加`--resume_from_huggingface`，则会从HuggingFace下载state并恢复学习。此时的`--resume`选项应为 `--resume {repo_id}/{path_in_repo}:{revision}:{repo_type}`。
    
    例: `--resume_from_huggingface --resume your-hf-name/your-model/path/test-000002-state:main:model`

    如果指定`--async_upload`选项，则会异步上传。

## 优化器相关设置

- `--optimizer_type`
    指定优化器类型。可选以下类型。
    - AdamW : [torch.optim.AdamW](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html)
    - 与过去版本的未指定时相同
    - AdamW8bit : 参数同上
    - PagedAdamW8bit : 参数同上
    - 与过去版本的`--use_8bit_adam`指定时相同
    - Lion : https://github.com/lucidrains/lion-pytorch
    - 与过去版本的`--use_lion_optimizer`指定时相同
    - Lion8bit : 参数同上
    - PagedLion8bit : 参数同上
    - SGDNesterov : [torch.optim.SGD](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html), nesterov=True
    - SGDNesterov8bit : 参数同上
    - DAdaptation(DAdaptAdamPreprint) : https://github.com/facebookresearch/dadaptation
    - DAdaptAdam : 参数同上
    - DAdaptAdaGrad : 参数同上
    - DAdaptAdan : 参数同上
    - DAdaptAdanIP : 参数同上
    - DAdaptLion : 参数同上
    - DAdaptSGD : 参数同上
    - Prodigy : https://github.com/konstmish/prodigy
    - AdaFactor : [Transformers AdaFactor](https://huggingface.co/docs/transformers/main_classes/optimizer_schedules)
    - 自定义优化器

- `--learning_rate`

    指定学习率。适当的学习率因学习脚本而异，请参考各自的说明。

- `--lr_scheduler` / `--lr_warmup_steps` / `--lr_scheduler_num_cycles` / `--lr_scheduler_power`
  
    学习率调度器相关的指定。

    `lr_scheduler`选项可从linear、cosine、cosine_with_restarts、polynomial、constant、constant_with_warmup、自定义调度器中选择。默认值为constant。
    
    `lr_warmup_steps`可指定调度器的预热步数。
    
    `lr_scheduler_num_cycles`是cosine with restarts调度器的重启次数，`lr_scheduler_power`是polynomial调度器的多项式幂。

    详情请自行查阅。

    使用自定义调度器时，与自定义优化器一样，通过`--lr_scheduler_args`指定选项参数。

### 优化器的指定

优化器的选项参数通过`--optimizer_args`选项指定。采用key=value的形式，可以指定多个值。此外，value可以通过逗号分隔指定多个值。例如，为AdamW优化器指定参数时，应为`--optimizer_args weight_decay=0.01 betas=.9,.999`。

指定选项参数时，请确认各自优化器的规格。

部分优化器有必需参数，如果省略则会自动添加（例如SGDNesterov的momentum）。请查看控制台输出。

D-Adaptation优化器会自动调整学习率。学习率选项中指定的值不是学习率本身，而是D-Adaptation决定的学习率的应用率，因此通常应指定为1.0。如果想为Text Encoder指定U-Net一半的学习率，应指定`--text_encoder_lr=0.5 --unet_lr=1.0`。

AdaFactor优化器如果指定`relative_step=True`，则可以自动调整学习率（省略时默认添加）。自动调整时，学习率调度器强制使用adafactor_scheduler。此外，似乎指定`scale_parameter`和`warmup_init`是有效的。

自动调整时的选项指定例如为 `--optimizer_args "relative_step=True" "scale_parameter=True" "warmup_init=True"`。

如果不自动调整学习率，则需添加选项参数 `relative_step=False`。这种情况下，学习率调度器建议使用constant_with_warmup，并且似乎不建议进行梯度clip norm。因此，参数应为 `--optimizer_type=adafactor --optimizer_args "relative_step=False" --lr_scheduler="constant_with_warmup" --max_grad_norm=0.0`。
### 使用任意的优化器

如果使用``torch.optim``中的优化器，请仅指定类名（例如``--optimizer_type=RMSprop``）；如果使用其他模块的优化器，请以“模块名.类名”的形式指定（例如``--optimizer_type=bitsandbytes.optim.lamb.LAMB``）。

（内部通过importlib实现，实际效果未验证。如有需要，请自行安装相应包。）

<!-- 
## 使用任意大小的图像进行训练 --resolution
可以训练非正方形的图像。在resolution参数中以“宽度,高度”的形式指定（例如“448,640”）。宽度和高度必须是64的倍数。请确保训练图像和正则化图像的大小一致。

个人经验表明，使用“448,640”等尺寸进行训练，有助于生成更丰富的纵向图像。

## 宽高比分桶（Aspect Ratio Bucketing）--enable_bucket / --min_bucket_reso / --max_bucket_reso
启用宽高比分桶功能，需要指定enable_bucket选项。Stable Diffusion 默认以 512x512 进行训练，但同时也支持其他分辨率，如 256x768 和 384x640。

使用此选项时，不需要将训练图像和正则化图像统一到特定的分辨率。程序会自动选择最合适的分辨率进行训练。
由于分辨率是64像素为单位，因此原始图像的宽高比可能与训练时的宽高比不完全一致，可能会有轻微的裁剪。

可以使用min_bucket_reso选项指定最小分辨率，使用max_bucket_reso选项指定最大分辨率。默认最小分辨率为256，最大分辨率为1024。
例如，如果将最小分辨率设置为384，则不会使用256x1024或320x768等分辨率。
如果将分辨率提高到768x768，可能需要将最大分辨率设置为1280等。

启用宽高比分桶时，建议准备与训练图像具有相似宽高比分布的正则化图像。

（因为这样可以避免一个batch内的图像过于偏向训练图像或正则化图像。虽然影响可能不是很大……）

## 数据增强（augmentation）--color_aug / --flip_aug
数据增强是一种通过在训练时动态改变数据来提高模型性能的技术。color_aug会随机改变图像的颜色，而flip_aug会对图像进行随机翻转。

由于数据增强会动态改变数据，因此不能与cache_latents选项同时使用。


## 使用fp16梯度的训练（实验性功能） --full_fp16
启用full_fp16选项后，梯度将从默认的float32转换为float16（fp16），从而进行完全的fp16训练（而非混合精度训练）。

这可以减少SD1.x在512x512分辨率下的VRAM使用量到8GB以下，SD2.x在512x512分辨率下的VRAM使用量到12GB以下。

需要预先通过accelerate config启用fp16支持，并添加``mixed_precision="fp16"``选项（不支持bf16）。

为了最小化VRAM使用量，建议同时启用xformers、use_8bit_adam、cache_latents和gradient_checkpointing，并将train_batch_size设置为1。

（如果有足够的显存，可以尝试逐渐增加train_batch_size，这可能会略微提高精度。）

该功能通过对PyTorch源码打补丁实现（已在PyTorch 1.12.1和1.13.0上验证）。请注意，使用该功能可能会导致精度下降和训练失败的风险增加。
同时，学习率和训练步数的设置也需要谨慎调整。在了解相关风险的前提下，自行承担责任使用该功能。

-->

# 创建元数据文件

## 准备训练数据

如前所述，请准备好想要训练的图像数据，并将其放入任意文件夹中。

例如，可以按照以下方式组织图像数据：

![训练数据文件夹截图](https://user-images.githubusercontent.com/52813779/208907739-8e89d5fa-6ca8-4b60-8927-f484d2a9ae04.png)

## 自动标注（可选）

如果仅使用标签进行训练而不需要使用标题（caption），请跳过此步骤。

如果手动准备标题（caption），请将标题文件与训练图像放在同一目录下，文件名相同，扩展名为.caption等。每个标题文件应为单行文本文件。

### 使用BLIP进行图像标注

最新版的程序已经包含了BLIP模型及其权重，无需额外下载。

运行finetune目录下的make_captions.py脚本：

```
python finetune\make_captions.py --batch_size <批次大小> <训练数据目录>
```

例如，如果批次大小为8，训练数据存放在上级目录的train_data文件夹中，可以运行：

```
python finetune\make_captions.py --batch_size 8 ..\train_data
```

标注文件将以.caption为扩展名，与训练图像存放在同一目录下，文件名相同。

批次大小（batch_size）可以根据GPU的VRAM容量进行调整。较大的批次大小可以提高处理速度（即使VRAM为12GB，也可能可以进一步增加批次大小）。
可以通过max_length选项指定标题的最大长度，默认为75。如果模型以225的token长度进行训练，可以考虑增加此值。
通过caption_extension选项可以更改标注文件的扩展名，默认为.caption（如果更改为.txt，可能会与DeepDanbooru的输出文件冲突）。

如果有多个训练数据目录，需要分别对每个目录运行该脚本。

由于BLIP模型的推断结果具有一定的随机性，多次运行的结果可能会有所不同。如果需要固定结果，可以通过--seed选项指定随机种子，例如``--seed 42``。

更多选项的说明可以通过``--help``参数查看（由于相关文档不完善，可能需要直接查看源码了解参数含义）。

默认情况下，标注文件的扩展名为.caption。

![包含标注文件的文件夹](https://user-images.githubusercontent.com/52813779/208908845-48a9d36c-f6ee-4dae-af71-9ab462d1459e.png)

例如，生成的标注文件可能如下所示：

![标注文件与图像](https://user-images.githubusercontent.com/52813779/208908947-af936957-5d73-4339-b6c8-945a52857373.png)

## 使用DeepDanbooru进行图像标签提取（可选）

如果不需要提取danbooru标签，可以直接跳到“标题和标签信息预处理”章节。

可以使用DeepDanbooru或WD14Tagger进行图像标签提取。据称WD14Tagger的准确率更高。如果要使用WD14Tagger，请直接跳转到下一章节。
## 环境准备

将 DeepDanbooru https://github.com/KichangKim/DeepDanbooru 克隆到工作文件夹，或者下载 zip 文件并解压。我是通过 zip 文件解压的。
同时，从 DeepDanbooru 的 Releases 页面 https://github.com/KichangKim/DeepDanbooru/releases 的 "DeepDanbooru Pretrained Model v3-20211112-sgd-e28" 的 Assets 中下载 deepdanbooru-v3-20211112-sgd-e28.zip，并将其解压到 DeepDanbooru 文件夹中。

从以下位置下载。点击 Assets 打开，然后从那里下载。

![DeepDanbooru 下载页面](https://user-images.githubusercontent.com/52813779/208909417-10e597df-7085-41ee-bd06-3e856a1339df.png)

请创建如下所示的目录结构。

![DeepDanbooru 目录结构](https://user-images.githubusercontent.com/52813779/208909486-38935d8b-8dc6-43f1-84d3-fef99bc471aa.png)

安装 Diffusers 环境所需的库。移动到 DeepDanbooru 文件夹并安装（实际上只是添加了 tensorflow-io）。

```bash
pip install -r requirements.txt
```

接下来，安装 DeepDanbooru 本身。

```bash
pip install .
```

至此，标签环境的准备工作已经完成。

## 执行标签

移动到 DeepDanbooru 文件夹，执行 deepdanbooru 进行标签。

```bash
deepdanbooru evaluate <训练数据文件夹> --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

如果将训练数据放在父文件夹的 train_data 中，则如下所示。

```bash
deepdanbooru evaluate ../train_data --project-path deepdanbooru-v3-20211112-sgd-e28 --allow-folder --save-txt
```

标签文件将在与训练数据图像相同的目录中创建，文件名相同，扩展名为 .txt。由于是一件一件处理的，所以比较慢。

如果有多个训练数据文件夹，请对每个文件夹执行此操作。

生成的文件如下所示。

![DeepDanbooru 生成的文件](https://user-images.githubusercontent.com/52813779/208909855-d21b9c98-f2d3-4283-8238-5b0e5aad6691.png)

标签如下所示（信息量很大……）。

![DeepDanbooru 标签和图像](https://user-images.githubusercontent.com/52813779/208909908-a7920174-266e-48d5-aaef-940aba709519.png)

## 使用 WD14Tagger 进行标签

这是使用 WD14Tagger 代替 DeepDanbooru 的步骤。

使用 Automatic1111 先生的 WebUI 中的 tagger。参考了github页面（https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger ）的信息。

初始环境准备时所需的模块已经安装完毕。同时，权重将从 Hugging Face 自动下载。

## 执行标签

执行脚本进行标签。
```bash
python tag_images_by_wd14_tagger.py --batch_size <批处理大小> <训练数据文件夹>
```

如果将训练数据放在父文件夹的 train_data 中，则如下所示。
```bash
python tag_images_by_wd14_tagger.py --batch_size 4 ..\train_data
```

首次运行时，模型文件将自动下载到 wd14_tagger_model 文件夹中（文件夹可以通过选项更改）。如下所示。

![下载的文件](https://user-images.githubusercontent.com/52813779/208910447-f7eb0582-90d6-49d3-a666-2b508c7d1842.png)

标签文件将在与训练数据图像相同的目录中创建，文件名相同，扩展名为 .txt。

![生成的标签文件](https://user-images.githubusercontent.com/52813779/208910534-ea514373-1185-4b7d-9ae3-61eb50bc294e.png)

![标签和图像](https://user-images.githubusercontent.com/52813779/208910599-29070c15-7639-474f-b3e4-06bd5a3df29e.png)

通过 thresh 选项，可以指定被判定为标签的 confidence（置信度）达到多少以上。默认值与 WD14Tagger 示例相同，为 0.35。降低该值会添加更多标签，但准确率会降低。

batch_size 请根据 GPU 的 VRAM 容量进行调整。较大的值会更快（即使 VRAM 为 12GB，也可以再大一些）。通过 caption_extension 选项，可以更改标签文件的扩展名。默认值为 .txt。

通过 model_dir 选项，可以指定模型的保存文件夹。

如果指定 force_download 选项，即使保存文件夹存在，也会重新下载模型。

如果有多个训练数据文件夹，请对每个文件夹执行此操作。

## 预处理标题和标签信息

为了方便从脚本中进行处理，将标题和标签合并到一个文件中作为元数据。
### 字幕预处理

要将字幕放入元数据中，请在工作文件夹内执行以下操作（如果不使用字幕进行训练，则无需执行）：（实际上是在一行中描述的，依此类推）。指定 `--full_path` 选项以将图像文件的位置以完整路径存储在元数据中。如果省略此选项，则会以相对路径记录，但需要在 `.toml` 文件中另外指定文件夹。

```
python merge_captions_to_metadata.py --full_path <训练数据文件夹>
    --in_json <要读取的元数据文件名> <元数据文件名>
```

元数据文件名可以是任意名称。
如果训练数据是 `train_data`，没有要读取的元数据文件，元数据文件是 `meta_cap.json`，则如下所示。

```
python merge_captions_to_metadata.py --full_path train_data meta_cap.json
```

可以使用 `caption_extension` 选项指定字幕的扩展名。

如果有多个训练数据文件夹，则需要指定 `full_path` 参数，并对每个文件夹执行。

```
python merge_captions_to_metadata.py --full_path 
    train_data1 meta_cap1.json
python merge_captions_to_metadata.py --full_path --in_json meta_cap1.json 
    train_data2 meta_cap2.json
```

如果省略 `in_json`，则如果存在写入目标元数据文件，则从那里读取并覆盖。

__※每次更改 `in_json` 选项和写入目标以输出到不同的元数据文件会更安全。__

### 标签预处理

同样，将标签汇总到元数据中（如果不使用标签进行训练，则无需执行）。
```
python merge_dd_tags_to_metadata.py --full_path <训练数据文件夹> 
    --in_json <要读取的元数据文件名> <要写入的元数据文件名>
```

在与之前相同的目录结构中，读取 `meta_cap.json` 并写入 `meta_cap_dd.json` 的情况下，如下所示。
```
python merge_dd_tags_to_metadata.py --full_path train_data --in_json meta_cap.json meta_cap_dd.json
```

如果有多个训练数据文件夹，则需要指定 `full_path` 参数，并对每个文件夹执行。

```
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap2.json
    train_data1 meta_cap_dd1.json
python merge_dd_tags_to_metadata.py --full_path --in_json meta_cap_dd1.json 
    train_data2 meta_cap_dd2.json
```

如果省略 `in_json`，则如果存在写入目标元数据文件，则从那里读取并覆盖。

__※每次更改 `in_json` 选项和写入目标以输出到不同的元数据文件会更安全。__

### 字幕和标签清理

到目前为止，元数据文件中包含了字幕和 DeepDanbooru 的标签。但是自动字幕可能会有一些表述上的差异（※），标签中可能包含下划线或评级（如果是 DeepDanbooru），因此最好使用编辑器的替换功能清理字幕和标签。

※例如，如果要训练动漫风格的少女图像，字幕中可能会有 `girl`/`girls`/`woman`/`women` 等不同的表述。将 “anime girl” 简化为 “girl” 可能更合适。

我们准备了一个用于清理的脚本，请根据情况编辑脚本内容后使用。

（不再需要指定训练数据文件夹。元数据中的所有数据都将被清理。）

```
python clean_captions_and_tags.py <要读取的元数据文件名> <要写入的元数据文件名>
```

请注意，这里没有 `--in_json`。例如：
```
python clean_captions_and_tags.py meta_cap_dd.json meta_clean.json
```

至此，字幕和标签的预处理已经完成。
## latents的预先获取

*此步骤不是必须的。即使省略，也可以在训练时获取latents并进行学习。*
另外，如果在训练时进行 `random_crop` 或 `color_aug` 等操作，则无法预先获取latents（因为每次都会改变图像）。如果不进行预先获取，可以使用到目前为止的元数据进行学习。

预先获取图像的潜在表示并将其保存到磁盘。这样可以加快学习速度。同时，还会进行bucketing（根据长宽比对训练数据进行分类）。

请在工作文件夹中输入以下内容：
```
python prepare_buckets_latents.py --full_path <训练数据文件夹>  
    <读取的元数据文件名> <写入的元数据文件名> 
    <fine tuning的模型名或checkpoint> 
    --batch_size <批次大小> 
    --max_resolution <分辨率 宽度,高度> 
    --mixed_precision <精度>
```

如果模型是model.ckpt，批次大小为4，学习分辨率为512*512，精度为no（float32），从meta_clean.json读取元数据并写入meta_lat.json，则如下所示：
```
python prepare_buckets_latents.py --full_path 
    train_data meta_clean.json meta_lat.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no
```

训练数据文件夹中会以numpy的npz格式保存latents。

可以使用--min_bucket_reso选项指定分辨率的最小大小，使用--max_bucket_reso指定最大大小。默认分别为256和1024。例如，如果指定最小大小为384，则不会使用256*1024或320*768等分辨率。
如果将分辨率提高到768*768等较大的值，可以将最大大小指定为1280等。

如果指定--flip_aug选项，则会进行左右翻转的数据增强（Data Augmentation）。这可以使数据量伪装成两倍，但如果数据不是左右对称的（例如角色的外观、发型等），则可能会导致学习失败。

（对于翻转的图像，也会获取latents并保存*_flip.npz文件。这是一个简单的实现，不需要在fine_tune.py中指定任何选项。如果存在带有_flip的文件，则会随机读取带有flip和不带flip的文件。）

批次大小即使在VRAM 12GB的情况下也可以尝试增加一些。
分辨率需要是64的倍数，以"宽度,高度"的格式指定。分辨率直接影响fine tuning时的内存大小。VRAM 12GB的情况下，512,512似乎是极限（*）。16GB的VRAM可能可以提高到512,704或512,768。但是，即使是256,256等，在VRAM 8GB的情况下似乎也比较吃力（因为参数和优化器等需要一定的内存，与分辨率无关）。

*有报告称，在batch size为1的情况下，12GB VRAM，640,640也可以运行。

如下所示，会显示bucketing的结果：

![bucketing的结果](https://user-images.githubusercontent.com/52813779/208911419-71c00fbb-2ce6-49d5-89b5-b78d7715e441.png)

如果有多个训练数据文件夹，可以通过指定full_path参数，对每个文件夹分别执行：
```
python prepare_buckets_latents.py --full_path  
    train_data1 meta_clean.json meta_lat1.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

python prepare_buckets_latents.py --full_path 
    train_data2 meta_lat1.json meta_lat2.json model.ckpt 
    --batch_size 4 --max_resolution 512,512 --mixed_precision no

```
可以将读取源和写入目标指定为相同的文件，但为了安全起见，最好分别指定不同的文件。

*为了安全起见，每次更改参数并写入不同的元数据文件。*