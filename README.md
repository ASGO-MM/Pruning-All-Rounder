# Pruning All-Rounder: Rethinking and Improving Inference Efficiency for Large Vision Language Models [ICCV' 25]

[![License: MIT](https://img.shields.io/badge/License-MIT-g.svg)](https://opensource.org/licenses/MIT) [![Arxiv](https://img.shields.io/badge/Paper-Arxiv-red)](https://arxiv.org/abs/2412.06458v2)

Although Large Vision-Language Models (LVLMs) have achieved impressive results, their high computational costs pose a significant barrier to wide application. To enhance inference efficiency, most existing approaches can be categorized as parameter-dependent or token-dependent strategies to reduce computational demands. However, parameter-dependent methods require retraining LVLMs to recover performance while token-dependent strategies struggle to consistently select the most relevant tokens. In this paper, we systematically analyze the above challenges and provide a series of valuable insights for inference acceleration. Based on these findings, we propose a novel framework, the Pruning All-Rounder (PAR). Different from previous works, PAR develops a meta-router to adaptively organize pruning flows across both tokens and layers. With a self-supervised learning manner, our method
achieves a superior balance between performance and efficiency. Notably, PAR is highly flexible, offering multiple pruning versions to address a range of acceleration scenarios.


## Motivation
<div align="center">
<img src=images\motivation.png>
</div>


## Method
<div align="center">
<img src=images\method.jpg>
</div>

## Quantitative Results
<div align="center">
<img src=images\quan.png>
</div>


## Qualitative Results
<div align="center">
<img src=images\qual.png>
</div>

## Installation

Coming soon.

## Acknowlegdements

This codebase is based on [LLaVA](https://github.com/haotian-liu/LLaVA), [Qwen-VL](https://github.com/QwenLM/Qwen-VL) and [FastV](https://github.com/pkunlp-icler/FastV). Many thanks to the authors for generously sharing their codes!
