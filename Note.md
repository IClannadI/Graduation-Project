# 毕设

## 1 理论知识与初步实践

### 1.1 Diffusion Model

#### 1.1.1 Generative Model

- 扩散模型*（Diffusion model）*是一种**生成模型（*Generative model*）**。
- 当今是GenAI的时代  [[6.S978_slide1_p2-p7]](https://github.com/IClannadI/Graduation-Project/tree/main/inference%20material/class/6.S978/slides)

##### 1.1.1.1 从判别模型到生成模型

- 判别模型*（discriminative model)*
  - 学习的概率分布：$p(y|x)$，其中x是图像，y时label，即输入一个图像，其对应label的概率
  - 基于CNN、RNN的图像分类器
  - 无法处理**不合理**的图像输入：总有某个label的概率最大
- 生成模型*(generative model)*
  - 学习的概率分布：$p(x)$，没有label，只学习输入图像的概率，即现实世界中所有合理的图像
  - VAE
  - 所有图像的概率之和为1，存在图像与图像之间的概率竞争关系
  - 能够拒绝不合理的图像输入
- 条件生成模型*(conditional model)*
  - 学习的概率分布：$p(x|y)$，给一个label，其图像应该长什么样
  - 文生图diffusion model
- 贝叶斯公式
  - $p(x|y)=\dfrac{p(y|x)}{p(y)}p(x)$
    - 三种模型通过贝叶斯公式*(Bayes' Rule)*联系起来
  - 但在实际应用中，条件生成模型不借助两个模型构建起来，而是从头自己训练
- **目前被谈论的*生成模型***一般指的是**条件生成模型**
- **为什么选择生成模型**
  - 当一个输入y对应多个不同的输出x时，我们需要将$p(x|y)$建模

##### 1.1.1.2 生成模型分类

- 显式概率密度（Explicit density)
  - 可解概率密度
    - 自回归模型 (Autoregressive model, AR)
      - 利用最大似然估计*(Maximum Likelihood Estimation)*进行建模与训练
      - 找到权重$W$，使得$\prod_{i}p(x^{(i)})$最大，即$\sum_{i}\log{p(x^{(i)})}$最大
        - $W^*=\arg\max_{W}\sum_{i}\log{f(x^{(i)},W)}$
      - 将某一个输入$x$分割成不同部分$x_1, x_2,....x_T$，可以使用RNN或Transformer建立AR，使用最大似然估计，使得$p(x)=\prod_i^T(p(x_i|x_1,...,x_{i-1)})$最大
        - 由于语言的离散型，易于分割，因此非常适合language modeling
  - 近似概率密度
    - 变分自动编码器VAE (Variational autoencoder, VAE)
      - VAE 通过编码器从原始数据$x$中学习潜变量的后验分布$q(z \mid x)$，训练完成后，可以从先验分布$p(z)$中取样潜变量 $z$，再通过解码器生成新的数据样本。
      - 目标：利用最大似然估计，那么就需要计算$p(x)$，根据贝叶斯定理，$p(x)=\dfrac{p(x|z)}{p(z|x)}p(z)$
        - $p(x|z)$可以用decoder计算
        - $p(z|x)$可以用另一个神经网络学习从而近似表示
- 隐式概率密度（implicit density)
  - 直接型
    - 生成对抗网络(Genrative Adversarial Networks, GANs)
      - 不直接对$p(x)$进行建模
      - 通过一个神经网络（Generative Model）来生成fake data，然后引入另一个神经网络（Discriminator Network）来判别fake data为real还是fake
      - 这两个网络相互对抗，互相促进：当判别器能够很好地识别fake data时，生成器必须生成更加逼真的fake data，最终我们希望生成器能够模拟物理真实世界中的生成器，生成近似真实的data
    - 难以训练与验证
  - 非直接型
    - Diffusion model

#### 1.1.2 什么是DM

- 先把真实图像一步一步加入噪音，然后学习给它去噪的过程，最后在一个随机噪音上一步步去噪，生成干净的图像。

#### 1.1.3 一类简单的DM：Rectified Flow

##### 训练（Training）

- 在每个iteration，取样以下元素
  - 噪音：$z \sim P_{noise}$
  - 样本：$x \sim P_{data}$
  - 时间步：$t \sim Uniform[0, 1]$
- $x_t=(1-t)x + tz.\ v=z-x$
- 训练一个神经网络来预测$v$：$L = \lVert f_\theta(x_t, t) - v \rVert_2^2$

##### 取样（Sampling）

- 设置一个$T$，对于$t$ in $[1. 1-\dfrac{1}{T}, 1-\dfrac{2}{T}, ...0]$:
  1. $v_t = f_\theta(x_t, t)$
  2. $x=x-v_t/T$
- 最后返回“干净”的$x$

##### 参考文献

- Liu et al, “Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow”, 2022
- Lipman et al, “Flow Matching for Generative Modeling”, 2022

#### 1.1.4 Latent Diffusion Models（现代用于高分辨率图像的扩散模型工作流）

##### 工作流程

- 训练一对encoder和decoder，让image在实际空间和潜在空间转换
  - 使用VAE训练encoder + decoder，然后使用GAN来训练Decoder，强化输出图像的清晰度
- 然后保持encoder与decoder的参数不变，训练一个diffusion model在潜在空间去噪
- 训练diffusion model完成后，在潜在空间随机取样，迭代式的去噪，最后通过decoder生成image

##### 参考文献

- Esser et al, “Scaling Rectified Flow Transformers for High-Resolution Image Synthesis”, arXiv 2024
- Rombach et al, “High-Resolution Image Synthesis with Latent Diffusion Models”, CVPR 2022

#### 1.1.5 Diffusion Transformer（DiT）

- 可用于文生图 + 文生视频
  - 额外需要一个Text encoder

##### 参考文献

- Peebles and Xie, ”Scalable Diffusion Models with Transformers", ICCV 2023
- Gupta et al, “Photorealistic Video Generation with Diffusion Models”, arXiv 2023 (Dec)
- OpenAI, “Sora: Creating Video from Text”, 2024 (Feb)
- Polyak et al, “Movie Gen: A Cast of Media Foundation Models”, arXiv 2024 (Oct)
- Kong et al, “HunyuanVideo: A Systematic Framework for Large Video Generative Models”, arXiv 2024 (Dec)
- NVIDIA, “Cosmos World Foundation Model Platform for Physical AI”, arXiv 2025 (Jan)
- Team Wan, “Wan: Open and Advanced Large-Scale Video Generative Models”, arXiv 2025 (March)

#### 1.1.6 Diffusion 蒸馏（Distillation）

- 由于采样阶段Diffusion model是逐步去噪，因此一次完整去噪需要使用Diffusion model多次，通过蒸馏可以减少去噪次数

##### 参考文献

- Salimans and Ho, “Progressive Distillation for Fast Sampling of Diffusion Models”, ICLR 2022
- Song et al, “Consistency Models”, ICML 2023
- Sauer et al, “Adversarial Diffusion Distillation”, ECCV 2024
- Sauer et al, “Fast High-Resolution Image Synthesis with Latent Adversarial Diffusion Distillation”, arXiv 2024
- Lu and Song, “Simplifying, Stabilizing and Scaling Consistency Models”, ICLR 2025
- Salimans et al, “Multistep Distillation of Diffusion Models via Moment Matching”, NeurIPS 2025

#### 1.1.7 广义扩散模型

##### 对于Rectified Flow：

- 取样 $x \sim P_{data}$， $z \sim P_{noise}$ 

- 取样$t \sim P_t$

- 设$x_t=(1-t)x + tz$
- 设$v_{gt}=z-x$
- 计算$v_{pred}= f_\theta(x_t, t)$

- 计算损失：$L = \lVert v_{gt}-v_{pred} \rVert_2^2$

##### 对于广义扩散模型

- 取样 $x \sim P_{data}$， $z \sim P_{noise}$ 

- 取样$t \sim P_t$

- 设$x_t=a(t)x + b(t)z$
- 设$v_{gt}=c(t)z+d(t)x$
- 计算$v_{pred}= f_\theta(x_t, t)$

- 计算损失：$L = \lVert v_{gt}-v_{pred} \rVert_2^2$

##### 从广义扩散模型到Rectified Flow：

- $a(t)=1-t,\ b(t)=t,\ c(t)=-1,\ d(t)=1$

##### 如何从不同视角看待Diffusion model？

- 参考[博客](https://sander.ai/2023/07/20/perspectives.html)



