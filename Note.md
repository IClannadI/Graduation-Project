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

##### 1.1.1.2 [生成模型分类](https://github.com/IClannadI/Graduation-Project/blob/main/inference%20material/papers/tutorial%20on%20generative%20model.pdf)

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
    - GAN
  - 非直接型
    - Diffusion model

