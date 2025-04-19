本文档介绍了可以通过 `--dataset_config` 传递的配置文件。

## 概要

通过传递配置文件，用户可以进行更细致的设置。

* 可以配置多个数据集
    * 例如，可以为每个数据集设置不同的 `resolution`，并混合使用它们进行训练。
    * 对于支持DreamBooth和fine-tuning两种训练方法的方法，可以混合使用这两种方式的数据集。
* 可以为每个子集设置不同的选项
    * 子集是指将数据集按图像目录或元数据分割后的部分。多个子集组成一个数据集。
    * 像 `keep_tokens` 和 `flip_aug` 这样的选项可以为每个子集单独设置。而像 `resolution` 和 `batch_size` 这样的选项则是针对整个数据集的，对于属于同一数据集的子集，这些选项的值是相同的。具体细节将在后文中说明。

配置文件可以使用JSON或TOML格式。考虑到易读性，推荐使用[TOML](https://toml.io/en/v1.0.0-rc.2)。以下说明将基于TOML格式。

下面是一个TOML配置文件的例子：

```toml
[general]
shuffle_caption = true
caption_extension = '.txt'
keep_tokens = 1

# 这是一个DreamBooth方式的数据集
[[datasets]]
resolution = 512
batch_size = 4
keep_tokens = 2

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
  class_tokens = 'hoge girl'
  # 这个子集的keep_tokens = 2（使用所属datasets的值）

  [[datasets.subsets]]
  image_dir = 'C:\fuga'
  class_tokens = 'fuga boy'
  keep_tokens = 3

  [[datasets.subsets]]
  is_reg = true
  image_dir = 'C:\reg'
  class_tokens = 'human'
  keep_tokens = 1

# 这是一个fine-tuning方式的数据集
[[datasets]]
resolution = [768, 768]
batch_size = 2

  [[datasets.subsets]]
  image_dir = 'C:\piyo'
  metadata_file = 'C:\piyo\piyo_md.json'
  # 这个子集的keep_tokens = 1（使用general的值）
```

在这个例子中，三个目录将以512x512（batch size 4）的分辨率使用DreamBooth方式进行训练，而一个目录将以768x768（batch size 2）的分辨率使用fine-tuning方式进行训练。

## 数据集和子集的设置

数据集和子集的设置可以在多个地方进行配置。

* `[general]`
    * 这里指定适用于所有数据集或所有子集的选项。
    * 如果在数据集或子集的设置中存在同名的选项，则数据集或子集的设置将优先。
* `[[datasets]]`
    * `datasets` 是用于注册数据集设置的部分。这里指定适用于各个数据集的选项。
    * 如果子集中存在同名的选项，则子集的设置将优先。
* `[[datasets.subsets]]`
    * `datasets.subsets` 是用于注册子集设置的部分。这里指定适用于各个子集的选项。

下面是一个关于图像目录和注册位置对应关系的示意图，基于前面的例子：

```
C:\
├─ hoge  ->  [[datasets.subsets]] No.1  ┐                        ┐
├─ fuga  ->  [[datasets.subsets]] No.2  |->  [[datasets]] No.1   |->  [general]
├─ reg   ->  [[datasets.subsets]] No.3  ┘                        |
└─ piyo  ->  [[datasets.subsets]] No.4  -->  [[datasets]] No.2   ┘
```

每个图像目录对应一个 `[[datasets.subsets]]`。多个 `[[datasets.subsets]]` 组合成一个 `[[datasets]]`。所有 `[[datasets]]` 和 `[[datasets.subsets]]` 都属于 `[general]`。

不同注册位置可以指定的选项不同。如果存在同名的选项，则较低级别的注册位置的值将优先。查看前面的例子中 `keep_tokens` 选项的处理方式可以帮助理解这一点。

此外，可用的选项还会根据训练方法支持的选项而有所不同。

* 仅适用于DreamBooth方式的选项
* 仅适用于fine-tuning方式的选项
* 如果支持caption dropout，则可用的选项

对于同时支持DreamBooth和fine-tuning的训练方法，可以混合使用这两种方式。但是，需要注意的是，在同一个数据集中，不能混合使用DreamBooth和fine-tuning方式的子集。也就是说，如果要混合使用这两种方式，需要将它们设置为不同的数据集。

程序的行为是，如果存在 `metadata_file` 选项，则判断为fine-tuning方式的子集。因此，对于属于同一数据集的子集，要么全部具有 `metadata_file` 选项，要么全部不具有该选项。

下面介绍可用的选项。对于与命令行参数名称相同的选项，通常不再重复说明。请参考其他README文件。

### 所有训练方法共通的选项

这些选项与训练方法无关。

#### 数据集选项

这些选项与数据集的设置相关，不能在 `datasets.subsets` 中指定。

| 选项名 | 设置示例 | `[general]` | `[[datasets]]` |
| ---- | ---- | ---- | ---- |
| `batch_size` | `1` | o | o |
| `bucket_no_upscale` | `true` | o | o |
| `bucket_reso_steps` | `64` | o | o |
| `enable_bucket` | `true` | o | o |
| `max_bucket_reso` | `1024` | o | o |
| `min_bucket_reso` | `128` | o | o |
| `resolution` | `256`, `[512, 512]` | o | o |

* `batch_size`
    * 与命令行参数 `--train_batch_size` 等效。
* `max_bucket_reso`, `min_bucket_reso`
    * 指定bucket的最大和最小分辨率。必须能被 `bucket_reso_steps` 整除。

这些设置对于数据集是固定的。因此，属于同一数据集的子集将共享这些设置。例如，如果要使用不同分辨率的数据集，可以像上面的例子一样，将它们定义为不同的数据集。

#### 子集选项

这些选项与子集的设置相关。

| 选项名 | 设置示例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `color_aug` | `false` | o | o | o |
| `face_crop_aug_range` | `[1.0, 3.0]` | o | o | o |
| `flip_aug` | `true` | o | o | o |
| `keep_tokens` | `2` | o | o | o |
| `num_repeats` | `10` | o | o | o |
| `random_crop` | `false` | o | o | o |
| `shuffle_caption` | `true` | o | o | o |
| `caption_prefix` | `“masterpiece, best quality, ”` | o | o | o |
| `caption_suffix` | `“, from side”` | o | o | o |
| `caption_separator` | （通常不设置） | o | o | o |
| `keep_tokens_separator` | `“|||”` | o | o | o |
| `secondary_separator` | `“;;;”` | o | o | o |
| `enable_wildcard` | `true` | o | o | o |
| `resize_interpolation` |（通常不设置） | o | o | o |

* `num_repeats`
    * 指定子集中图像的重复次数。相当于fine-tuning中的 `--dataset_repeats`，但 `num_repeats` 可用于任何训练方法。
* `caption_prefix`, `caption_suffix`
    * 指定在标题前后添加的字符串。shuffle操作将在添加这些字符串后进行。如果设置了 `keep_tokens`，请注意这一点。

* `caption_separator`
    * 指定用于分隔标签的字符串。默认为 `,`。通常不需要设置此选项。

* `keep_tokens_separator`
    * 指定用于分隔标题中要保留部分的字符。例如，`aaa, bbb ||| ccc, ddd, eee, fff ||| ggg, hhh` 将保留 `aaa, bbb` 和 `ggg, hhh` 部分，不会被shuffle或drop。中间的逗号是可选的。结果可能是 `aaa, bbb, eee, ccc, fff, ggg, hhh` 或 `aaa, bbb, fff, ccc, eee, ggg, hhh` 等。

* `secondary_separator`
    * 指定额外的分隔符。由此分隔的部分将被视为一个标签，进行shuffle和drop，然后替换为 `caption_separator`。例如，`aaa;;;bbb;;;ccc` 将被替换为 `aaa,bbb,ccc`，或者一起被drop。

* `enable_wildcard`
    * 启用通配符表示法和多行标题。通配符表示法和多行标题将在后面说明。

* `resize_interpolation`
    * 指定图像缩放时使用的插值方法。通常不需要设置。如果指定了，则将使用相同的插值方法进行放大和缩小。可以指定的值有 `lanczos`, `nearest`, `bilinear`, `linear`, `bicubic`, `cubic`, `area`, `box`。默认（未指定时）为缩小时使用 `area`，放大时使用 `lanczos`。如果指定了此选项，则将使用相同的插值方法进行放大和缩小。指定 `lanczos` 或 `box` 时将使用PIL，其他值将使用OpenCV。

### 仅适用于DreamBooth方式的选项

DreamBooth方式的选项仅存在于子集选项中。

#### 子集选项

这些选项与DreamBooth方式的子集设置相关。

| 选项名 | 设置示例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `‘C:\hoge’` | - | - | o（必须） |
| `caption_extension` | `".txt"` | o | o | o |
| `class_tokens` | `“sks girl”` | - | - | o |
| `cache_info` | `false` | o | o | o | 
| `is_reg` | `false` | - | - | o |

首先需要注意的是，`image_dir` 必须指定图像文件直接位于该路径下。在传统的DreamBooth方法中，图像文件需要放在子目录下，但这里的规范与之不兼容。此外，即使文件夹名为 `5_cat`，也不会反映图像的重复次数和类别名称。如果需要单独设置这些，请使用 `num_repeats` 和 `class_tokens` 显式指定。

* `image_dir`
    * 指定图像目录的路径。这是必须的选项。
    * 图像必须直接位于目录下。
* `class_tokens`
    * 设置类别标记。
    * 仅当图像对应的标题文件不存在时，才会在训练时使用。是否使用的判断是针对每个图像进行的。如果未指定 `class_tokens` 且标题文件也不存在，则会出错。
* `cache_info`
    * 指定是否缓存图像大小和标题。默认值为 `false`。缓存将保存在 `image_dir` 中的 `metadata_cache.json` 文件中。
    * 缓存可以加快第二次及以后读取数据集的速度。对于数千张以上的图像，启用此选项是有效的。
* `is_reg`
    * 指定子集中的图像是否为正则化图像。默认值为 `false`，即不是正则化图像。

### 仅适用于fine-tuning方式的选项

fine-tuning方式的选项仅存在于子集选项中。

#### 子集选项

这些选项与fine-tuning方式的子集设置相关。

| 选项名 | 设置示例 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- | ---- |
| `image_dir` | `‘C:\hoge’` | - | - | o |
| `metadata_file` | `'C:\piyo\piyo_md.json'` | - | - | o（必须） |

* `image_dir`
    * 指定图像目录的路径。与DreamBooth方式不同，这里不是必须的，但建议指定。
        * 不需要指定的情况是，在生成元数据文件时，使用了 image_dir 以外的路径。

    * 图像必须直接位于目录下。
* `metadata_file`
    * 指定子集中使用的元数据文件的路径。这是必须的选项。
        * 与命令行参数 `--in_json` 等效。
    * 由于子集需要单独指定元数据文件，因此不建议创建跨目录的元数据文件作为一个元数据文件。强烈建议为每个图像目录创建元数据文件，并将它们注册为不同的子集。

### 如果支持caption dropout，则可用的选项

如果训练方法支持caption dropout，则以下选项可用。这些选项仅存在于子集选项中。

#### 子集选项

这些选项与支持caption dropout的子集设置相关。

| 选项名 | `[general]` | `[[datasets]]` | `[[dataset.subsets]]` |
| ---- | ---- | ---- | ---- |
| `caption_dropout_every_n_epochs` | o | o | o |
| `caption_dropout_rate` | o | o | o |
| `caption_tag_dropout_rate` | o | o | o |

## 子集重复时的行为

对于DreamBooth方式的数据集，如果其中存在 `image_dir` 相同的子集，则被视为重复，第二个及以后的子集将被忽略。
对于fine-tuning方式的数据集，如果其中存在 `metadata_file` 相同的子集，则被视为重复，第二个及以后的子集将被忽略。

然而，如果子集属于不同的数据集，则不被视为重复。
例如，如果将具有相同 `image_dir` 的子集放在不同的数据集中，如下所示，则不会被视为重复。
这在以不同分辨率训练同一图像时很有用。

```toml
# 如果存在于不同的数据集中，则不被视为重复，都会被用于训练

[[datasets]]
resolution = 512

  [[datasets.subsets]]
  image_dir = 'C:\hoge'

[[datasets]]
resolution = 768

  [[datasets.subsets]]
  image_dir = 'C:\hoge'
```

## 与命令行参数的结合使用

配置文件中的一些选项与命令行参数的选项具有相同的作用。

以下命令行参数的选项在传递配置文件时将被忽略。

* `--train_data_dir`
* `--reg_data_dir`
* `--in_json`

以下命令行参数的选项在同时通过命令行和配置文件指定时，配置文件的选项将优先。没有特别说明的均为同名选项。

| 命令行参数选项     | 优先的配置文件选项 |
| ---------------------------------- | ---------------------------------- |
| `--bucket_no_upscale`              |                                    |
| `--bucket_reso_steps`              |                                    |
| `--caption_dropout_every_n_epochs` |                                    |
| `--caption_dropout_rate`           |                                    |
| `--caption_extension`              |                                    |
| `--caption_tag_dropout_rate`       |                                    |
| `--color_aug`                      |                                    |
| `--dataset_repeats`                | `num_repeats`                      |
| `--enable_bucket`                  |                                    |
| `--face_crop_aug_range`            |                                    |
| `--flip_aug`                       |                                    |
| `--keep_tokens`                    |                                    |
| `--min_bucket_reso`                |                                    |
| `--random_crop`                    |                                    |
| `--resolution`                     |                                    |
| `--shuffle_caption`                |                                    |
| `--train_batch_size`               | `batch_size`                       |

## 错误指南

目前，我们使用外部库来检查配置文件是否正确书写，但由于维护不够，导致错误消息不够清晰。
我们计划在将来解决这个问题。

作为临时解决方案，我们列出了一些常见的错误及其解决方法。
如果您确信配置正确却仍然出错，或者无法理解错误原因，请联系我们，因为可能是bug。

* `voluptuous.error.MultipleInvalid: required key not provided @ ...`：这是一个错误，表明缺少必须的选项。可能是忘记指定或错误地书写了选项名称。
  * `...` 部分显示了错误发生的位置。例如，如果出现 `voluptuous.error.MultipleInvalid: required key not provided @ data['datasets'][0]['subsets'][0]['image_dir']`，则意味着第0个 `datasets` 中的第0个 `subsets` 缺少 `image_dir`。
* `voluptuous.error.MultipleInvalid: expected int for dictionary value @ ...`：这是一个错误，表明值的格式不正确。可能是值的格式错误。`int` 部分会根据选项的不同而变化。本README中的“设置示例”可能会有所帮助。
* `voluptuous.error.MultipleInvalid: extra keys not allowed @ ...`：当存在不支持的选项名称时，会发生此错误。可能是错误地书写了选项名称或误写了其他内容。

## 其他

### 多行标题

设置 `enable_wildcard = true` 后，多行标题也会被启用。如果标题文件包含多行，则会随机选择一行作为标题。

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, microphone, stage
a girl with a microphone standing on a stage
detailed digital art of a girl with a microphone on a stage
```

也可以与通配符表示法结合使用。

元数据文件也支持多行标题。在JSON中，使用 `\n` 表示换行。如果标题文件包含多行，使用 `merge_captions_to_metadata.py` 将创建这种格式的元数据文件。

元数据中的标签 (`tags`) 将被添加到标题的每一行。

```json
{
    "/path/to/image.png": {
        "caption": "a cartoon of a frog with the word frog on it\ntest multiline caption1\ntest multiline caption2",
        "tags": "open mouth, simple background, standing, no humans, animal, black background, frog, animal costume, animal focus"
    },
    ...
}
```

在这种情况下，实际的标题可能是 `a cartoon of a frog with the word frog on it, open mouth, simple background ...` 或 `test multiline caption1, open mouth, simple background ...`、`test multiline caption2, open mouth, simple background ...` 等。

### 配置文件示例：附加分隔符、通配符表示法、`keep_tokens_separator` 等

```toml
[general]
flip_aug = true
color_aug = false
resolution = [1024, 1024]

[[datasets]]
batch_size = 6
enable_bucket = true
bucket_no_upscale = true
caption_extension = ".txt"
keep_tokens_separator= "|||"
shuffle_caption = true
caption_tag_dropout_rate = 0.1
secondary_separator = ";;;" # 也可以写在subset侧
enable_wildcard = true # 同上

  [[datasets.subsets]]
  image_dir = "/path/to/image_dir"
  num_repeats = 1

  # |||前后不需要逗号（会自动添加）
  caption_prefix = "1girl, hatsune miku, vocaloid |||" 

  # |||后面的部分不会被shuffle或drop
  # 简单地作为字符串连接，因此需要自己添加逗号
  caption_suffix = ", anime screencap ||| masterpiece, rating: general"
```

### 标题示例，secondary_separator 表示法：`secondary_separator = ";;;"` 的情况

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, sky;;;cloud;;;day, outdoors
```
`sky;;;cloud;;;day` 部分不会被shuffle或drop，而是被替换为 `sky,cloud,day`。如果启用了shuffle或drop，则会作为一个标签整体处理。结果可能是 `vocaloid, 1girl, upper body, sky,cloud,day, outdoors, hatsune miku`（shuffle）或 `vocaloid, 1girl, outdoors, looking at viewer, upper body, hatsune miku`（drop后的情况）等。

### 标题示例，通配符表示法：`enable_wildcard = true` 的情况

```txt
1girl, hatsune miku, vocaloid, upper body, looking at viewer, {simple|white} background
```
随机选择 `simple` 或 `white`，结果是 `simple background` 或 `white background`。

```txt
1girl, hatsune miku, vocaloid, {{retro style}}
```
如果要在标签字符串中包含 `{` 或 `}`，请像 `{{` 或 `}}` 这样写两次（在这个例子中，实际用于训练的标题是 `{retro style}`）。

### 标题示例，`keep_tokens_separator` 表示法：`keep_tokens_separator = "|||"` 的情况

```txt
1girl, hatsune miku, vocaloid ||| stage, microphone, white shirt, smile ||| best quality, rating: general
```
结果可能是 `1girl, hatsune miku, vocaloid, microphone, stage, white shirt, best quality, rating: general` 或 `1girl, hatsune miku, vocaloid, white shirt, smile, stage, microphone, best quality, rating: general` 等。