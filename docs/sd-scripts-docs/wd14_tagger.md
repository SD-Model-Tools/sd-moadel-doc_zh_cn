# 使用WD14Tagger进行标签化

此部分的说明参考了该github页面（https://github.com/toriato/stable-diffusion-webui-wd14-tagger#mrsmilingwolfs-model-aka-waifu-diffusion-14-tagger ）。

建议使用onnx进行推理。请使用以下命令安装onnx。

```powershell
pip install onnx==1.15.0 onnxruntime-gpu==1.17.1
```

模型的权重将从Hugging Face自动下载。

# 使用方法

运行脚本以对图像进行标签化。

```
python fintune/tag_images_by_wd14_tagger.py --onnx --repo_id <模型的repo id> --batch_size <批大小> <训练数据文件夹>
```

如果使用 `SmilingWolf/wd-swinv2-tagger-v3` 作为仓库ID，并将批大小设置为4，同时将训练数据放在父文件夹的 `train_data` 中，则命令如下。

```
python tag_images_by_wd14_tagger.py --onnx --repo_id SmilingWolf/wd-swinv2-tagger-v3 --batch_size 4 ..\train_data
```

首次运行时，模型文件将自动下载到 `wd14_tagger_model` 文件夹中（文件夹路径可通过选项更改）。

标签文件将与训练数据图像在同一目录下创建，具有相同的文件名，扩展名为.txt。

![生成的标签文件](https://user-images.githubusercontent.com/52813779/208910534-ea514373-1185-4b7d-9ae3-61eb50bc294e.png)

![标签与图像](https://user-images.githubusercontent.com/52813779/208910599-29070c15-7639-474f-b3e4-06bd5a3df29e.png)

## 命令示例

如果要以Animagine XL 3.1 方式输出，命令如下（实际输入时请在一行内输入）。

```
python tag_images_by_wd14_tagger.py --onnx --repo_id SmilingWolf/wd-swinv2-tagger-v3 
    --batch_size 4  --remove_underscore --undesired_tags "PUT,YOUR,UNDESIRED,TAGS" --recursive 
    --use_rating_tags_as_last_tag --character_tags_first --character_tag_expand 
    --always_first_tags "1girl,1boy"  ..\train_data
```

## 可用的仓库ID

可以使用[SmilingWolf先生的V2、V3模型](https://huggingface.co/SmilingWolf)。请指定为 `SmilingWolf/wd-vit-tagger-v3` 这样的形式。默认值为 `SmilingWolf/wd-v1-4-convnext-tagger-v2`。

# 选项

## 常规选项

- `--onnx` : 使用ONNX进行推理。如果不指定，则使用TensorFlow。使用TensorFlow时需要单独安装TensorFlow。
- `--batch_size` : 一次处理的图像数量。默认值为1。请根据VRAM容量调整。
- `--caption_extension` : 标签文件的扩展名。默认值为 `.txt`。
- `--max_data_loader_n_workers` : DataLoader的最大工作进程数。如果指定1或更大的数值，将使用DataLoader加速图像读取。如果未指定，则不使用DataLoader。
- `--thresh` : 输出标签的置信度阈值。默认值为0.35。降低此值会输出更多标签，但准确度会降低。
- `--general_threshold` : 一般标签的置信度阈值。如果省略，则与 `--thresh` 相同。
- `--character_threshold` : 角色标签的置信度阈值。如果省略，则与 `--thresh` 相同。
- `--recursive` : 如果指定，则递归处理指定文件夹内的子文件夹。
- `--append_tags` : 向现有的标签文件追加标签。
- `--frequency_tags` : 输出标签的频率。
- `--debug` : 调试模式。指定后输出调试信息。

## 模型下载

- `--model_dir` : 模型文件的保存目录。默认值为 `wd14_tagger_model`。
- `--force_download` : 如果指定，则重新下载模型文件。

## 标签编辑相关

- `--remove_underscore` : 从输出的标签中删除下划线。
- `--undesired_tags` : 指定不输出的标签。可以用逗号分隔多个标签。例如 `black eyes,black hair`。
- `--use_rating_tags` : 在标签的开头输出分级标签。
- `--use_rating_tags_as_last_tag` : 在标签的末尾添加分级标签。
- `--character_tags_first` : 首先输出角色标签。
- `--character_tag_expand` : 展开角色标签的系列名称。例如，将 `chara_name_(series)` 标签分割为 `chara_name, series`。
- `--always_first_tags` : 当图像中存在某个标签时，指定将其放在最前面的标签。可以用逗号分隔多个标签。例如 `1girl,1boy`。
- `--caption_separator` : 在输出文件中用此字符串分隔标签。默认值为 `, `。
- `--tag_replacement` : 进行标签替换。指定格式为 `tag1,tag2;tag3,tag4`。如果要使用 `,` 或 `;`，请用 `\` 进行转义。
    例如，`aira tsubase,aira tsubase (uniform)`（当要学习特定服装时），`aira tsubase,aira tsubase\, heir of shadows`（当系列名称未包含在标签中时）。

`tag_replacement` 在 `character_tag_expand` 之后应用。

指定 `remove_underscore` 时，`undesired_tags`、`always_first_tags`、`tag_replacement` 请不要包含下划线。

指定 `caption_separator` 时，`undesired_tags`、`always_first_tags` 请用 `caption_separator` 分隔。`tag_replacement` 请始终用 `,` 分隔。