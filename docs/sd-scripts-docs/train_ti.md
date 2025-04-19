这是关于 [Textual Inversion](https://textual-inversion.github.io/) 学习的说明。

请同时参考[关于学习的共同文档](./train.md)。

在实现时，我们大量参考了 https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion。

学习的模型可以直接在 Web UI 中使用。

# 学习步骤

请预先参考此存储库的 README，进行环境设置。

## 数据准备

请参考[关于学习数据的准备](./train.md)。

## 执行学习

使用 ``train_textual_inversion.py``。以下是命令行示例（DreamBooth 方法）。

```
accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py 
    --dataset_config=<数据准备时创建的.toml文件> 
    --output_dir=<学习模型的输出文件夹>  
    --output_name=<学习模型输出时的文件名> 
    --save_model_as=safetensors 
    --prior_loss_weight=1.0 
    --max_train_steps=1600 
    --learning_rate=1e-6 
    --optimizer_type="AdamW8bit" 
    --xformers 
    --mixed_precision="fp16" 
    --cache_latents 
    --gradient_checkpointing
    --token_string=mychar4 --init_word=cute --num_vectors_per_token=4
```

在 ``--token_string`` 中指定学习时的标记字符串。**学习时的提示词必须包含此字符串**（如果 token_string 是 mychar4，则为 ``mychar4 1girl`` 等）。提示词中的此字符串部分将被替换为 Textual Inversion 的新标记并进行学习。最简单且最可靠的方法是使用 DreamBooth、class+identifier 形式的数据集，并将 `token_string` 作为标记字符串。

要检查提示词中是否包含标记字符串，可以使用 ``--debug_dataset`` 查看替换后的 token id。如果存在 ``49408`` 之后的 token，则表示包含标记字符串。

```
input ids: tensor([[49406, 49408, 49409, 49410, 49411, 49412, 49413, 49414, 49415, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407]])
```

不能使用分词器已经拥有的单词（常见单词）。

在 ``--init_word`` 中指定初始化 embeddings 时的复制源标记字符串。选择与要学习的 concept 接近的单词似乎比较好。不能指定由两个以上标记组成的字符串。

在 ``--num_vectors_per_token`` 中指定在此学习中使用多少个标记。数量越多，表现力越强，但也会占用更多的标记。例如，如果 num_vectors_per_token=8，则指定的标记字符串将占用（一般提示词的 77 个标记限制中的）8 个标记。

以上是 Textual Inversion 的主要选项。之后与其他学习脚本相同。

`num_cpu_threads_per_process` 通常指定为 1。

在 `pretrained_model_name_or_path` 中指定作为基础进行额外学习的模型。可以指定 Stable Diffusion 的 checkpoint 文件（.ckpt 或 .safetensors）、Diffusers 的本地磁盘上的模型目录、Diffusers 的模型 ID（例如 "stabilityai/stable-diffusion-2"）。

在 `output_dir` 中指定学习后的模型的保存文件夹。在 `output_name` 中指定模型的文件名（不包括扩展名）。通过 `save_model_as` 指定保存为 safetensors 格式。

在 `dataset_config` 中指定 `.toml` 文件。在文件内，初始时请将批处理大小指定为 `1` 以减少内存消耗。

将学习步数 `max_train_steps` 指定为 10000。这里将学习率 `learning_rate` 指定为 5e-6。

为了减少内存使用，指定 `mixed_precision="fp16"`（在 RTX30 系列之后，可以指定 `bf16`。请与在环境设置时对 accelerate 进行的设置保持一致）。同时指定 `gradient_checkpointing`。

为了使用内存消耗较少的 8bit AdamW 作为优化器（将模型优化以适应学习数据），指定 `optimizer_type="AdamW8bit"`。

指定 `xformers` 选项，使用 xformers 的 CrossAttention。如果未安装 xformers 或出现错误（在某些环境下，例如 `mixed_precision="no"`），可以指定 `mem_eff_attn` 选项以使用节省内存的 CrossAttention（速度会变慢）。

如果内存足够，可以编辑 `.toml` 文件，将批处理大小增加到例如 `8`（可能会提高速度和精度）。

### 关于常用选项

在以下情况下，请参考有关选项的文档。

- 学习 Stable Diffusion 2.x 或其派生模型
- 学习 clip skip 为 2 或以上的模型
- 使用超过 75 个标记的标题进行学习

### 关于 Textual Inversion 的批处理大小

与学习整个模型的 DreamBooth 和 fine tuning 相比，Textual Inversion 的内存使用量较少，因此可以设置较大的批处理大小。

# Textual Inversion 的其他主要选项

有关所有选项，请参考其他文档。

* `--weights`
  * 在学习之前加载已学习的 embeddings，并从那里继续学习。
* `--use_object_template`
  * 不使用标题，而是使用预设的物体模板字符串（例如 ``a photo of a {}``）进行学习。这与官方实现相同。标题将被忽略。
* `--use_style_template`
  * 不使用标题，而是使用预设的样式模板字符串（例如 ``a painting in the style of {}``）进行学习。这与官方实现相同。标题将被忽略。

## 使用此存储库中的图像生成脚本生成图像

在 gen_img_diffusers.py 中，使用 ``--textual_inversion_embeddings`` 选项指定学习的 embeddings 文件（可指定多个）。在提示词中使用 embeddings 文件的文件名（不包括扩展名），将应用相应的 embeddings。