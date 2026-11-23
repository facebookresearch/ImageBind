# ImageBind：统一融合万物的多模态联合特征嵌入空间

**简体中文** | [**English**](README.md)

**[FAIR，Meta AI](https://ai.facebook.com/research/)**

Rohit Girdhar*、Alaaeldin El-Nouby*、Zhuang Liu、Mannat Singh、Kalyan Vasudev Alwala、Armand Joulin、Ishan Misra*

发表于 CVPR 2023（*Highlighted Paper 重点论文*）

[[`论文`](https://facebookresearch.github.io/ImageBind/paper)] [[`博客`](https://ai.facebook.com/blog/imagebind-six-modalities-binding-ai/)] [[`交互式演示`](https://imagebind.metademolab.com/)] [[`补充视频`](https://dl.fbaipublicfiles.com/imagebind/imagebind_video.mp4)] [[`BibTeX 引用`](#引用-imagebind)]

本仓库提供 ImageBind 的 PyTorch 官方实现与预训练模型权重。完整技术细节请参阅研究论文：**[ImageBind: One Embedding Space To Bind Them All](https://facebookresearch.github.io/ImageBind/paper)**。

ImageBind 学习跨越六种不同模态的联合特征嵌入空间——**图像**、**文本**、**音频**、**深度图**、**热成像**以及 **IMU 惯性测量单元数据**。它赋予了模型一系列开箱即用的涌现能力（Emergent Capabilities），包括跨模态检索、基于向量代数的模态特征组合、跨模态目标检测与多模态生成。

![ImageBind](https://user-images.githubusercontent.com/8495451/236859695-ffa13364-3e39-4d99-a8da-fbfab17f9a6b.gif)

## ImageBind 预训练模型

涌现的零样本分类（Zero-shot Classification）基准表现：

<table style="margin: auto">
  <tr>
    <th>模型</th>
    <th><span style="color:blue">IN1k</span></th>
    <th><span style="color:purple">K400</span></th>
    <th><span style="color:green">NYU-D</span></th>
    <th><span style="color:LightBlue">ESC</span></th>
    <th><span style="color:orange">LLVIP</span></th>
    <th><span style="color:purple">Ego4D</span></th>
    <th>权重下载</th>
  </tr>
  <tr>
    <td>imagebind_huge</td>
    <td align="right">77.7</td>
    <td align="right">50.0</td>
    <td align="right">54.0</td>
    <td align="right">66.9</td>
    <td align="right">63.4</td>
    <td align="right">25.0</td>
    <td><a href="https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth">checkpoint</a></td>
  </tr>
</table>

## 快速上手 (Usage)

安装 PyTorch 2.0+ 及其他第三方依赖库：

```shell
conda create --name imagebind python=3.10 -y
conda activate imagebind

pip install .
```

Windows 用户可能需要单独安装 `soundfile` 以支持读取和写入音频文件（感谢 @congyue1977）：

```
pip install soundfile
```

提取并对比跨模态特征表示（例如图像、文本与音频）：

```python
from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

text_list=["A dog.", "A car", "A bird"]
image_paths=[".assets/dog_image.jpg", ".assets/car_image.jpg", ".assets/bird_image.jpg"]
audio_paths=[".assets/dog_audio.wav", ".assets/car_audio.wav", ".assets/bird_audio.wav"]

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# 实例化预训练模型
model = imagebind_model.imagebind_huge(pretrained=True)
model.eval()
model.to(device)

# 加载并预处理多模态数据
inputs = {
    ModalityType.TEXT: data.load_and_transform_text(text_list, device),
    ModalityType.VISION: data.load_and_transform_vision_data(image_paths, device),
    ModalityType.AUDIO: data.load_and_transform_audio_data(audio_paths, device),
}

with torch.no_grad():
    embeddings = model(inputs)

print(
    "Vision x Text: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Audio x Text: ",
    torch.softmax(embeddings[ModalityType.AUDIO] @ embeddings[ModalityType.TEXT].T, dim=-1),
)
print(
    "Vision x Audio: ",
    torch.softmax(embeddings[ModalityType.VISION] @ embeddings[ModalityType.AUDIO].T, dim=-1),
)

# 预期输出结果：
#
# Vision x Text:
# tensor([[9.9761e-01, 2.3694e-03, 1.8612e-05],
#         [3.3836e-05, 9.9994e-01, 2.4118e-05],
#         [4.7997e-05, 1.3496e-02, 9.8646e-01]])
#
# Audio x Text:
# tensor([[1., 0., 0.],
#         [0., 1., 0.],
#         [0., 0., 1.]])
#
# Vision x Audio:
# tensor([[0.8070, 0.1088, 0.0842],
#         [0.1036, 0.7884, 0.1079],
#         [0.0018, 0.0022, 0.9960]])
```

## 模型卡片 (Model card)

完整模型指标与训练信息请参阅 [模型卡片 (model_card.md)](model_card.md)。

## 许可证 (License)

ImageBind 代码及预训练模型权重均基于 CC-BY-NC 4.0 许可证授权发布。详情请参阅 [LICENSE](LICENSE)。

## 贡献指南 (Contributing)

请参阅 [贡献指引 (CONTRIBUTING.md)](CONTRIBUTING.md) 与 [行为准则 (CODE_OF_CONDUCT.md)](CODE_OF_CONDUCT.md)。

## 引用 ImageBind

若本项目对研究有所帮助，欢迎为本仓库点亮星标 :star: 并引用原论文：

```bibtex
@inproceedings{girdhar2023imagebind,
  title={ImageBind: One Embedding Space To Bind Them All},
  author={Girdhar, Rohit and El-Nouby, Alaaeldin and Liu, Zhuang
and Singh, Mannat and Alwala, Kalyan Vasudev and Joulin, Armand and Misra, Ishan},
  booktitle={CVPR},
  year={2023}
}
```
