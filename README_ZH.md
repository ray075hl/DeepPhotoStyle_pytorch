# DeepPhotoStyle_pytorch
使用Pytorch重现文章 ["Deep Photo Style Transfer"](https://arxiv.org/abs/1703.07511) 的效果.
本工程提供图像语义切分的代码，无需手动抠图.
## 一些实验的效果图
![](./doc_image/ex_001.jpg)
![](./doc_image/ex_002.png)
![](./doc_image/ex_003.jpg)
## 搭建环境
安装 [pytorch](https://pytorch.org/)0.4.1的CUDA版本

```
git clone

https://github.com/ray075hl/DeepPhotoStyle_pytorch.git

cd DeepPhotoStyle_pytorch

sh download_seg_model.sh

python main.py --style_image path_style_image --content_image path_content_image
```

## 注意
图像的语义切分结果对最终的风格变化的质量影响很大，我们应该观察风格和内容图片的语义切分结果是否符合预期的匹配(例如：山对山，天空对天空等等)

## 引用
[1] 所有的语义切分代码均来自这里 [Semantic-segmentation-pytorch](https://github.com/CSAILVision/semantic-segmentation-pytorch). 非常感谢这个优秀的工程

[2] Pytorch的风格变换的基本框架借鉴于此.  [Neural Transfer with PyTorch](https://pytorch.org/tutorials/advanced/neural_style_tutorial.html)

[3] 如何计算拉普拉斯矩阵. [Closed-form-matting
](https://github.com/MarcoForte/closed-form-matting)

[4] 算法来源["Deep Photo Style Transfer"](https://arxiv.org/abs/1703.07511)

[5] 一些图像的后处理与原文不太一样，使用的是这篇文章的后处理.[Visual Attribute Transfer through Deep Image Analogy](https://arxiv.org/abs/1705.01088)