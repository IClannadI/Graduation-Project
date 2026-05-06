# 实验环境信息

生成时间：2026-05-05 10:50:32

## 1. 操作系统与基础环境

| 项目 | 信息 |
| :--- | :--- |
| 操作系统 | Ubuntu 22.04.1 LTS |
| Linux内核 | 5.15.0-94-generic |
| 主机名 | autodl-container-8b8a4382f2-5504fe12 |
| Shell | /bin/bash |

## 2. 硬件环境

| 项目 | 信息 |
| :--- | :--- |
| CPU型号 | Intel(R) Xeon(R) Platinum 8474C |
| CPU核心数 | 15 |
| 内存大小 | 1.0Ti |
| 数据盘容量 | 200G total, 128G available |

### 2.1 GPU信息

| GPU编号 | GPU型号 | 显存总量 | Driver版本 | CUDA版本 |
| :---: | :--- | :---: | :---: | :---: |
| 0 | NVIDIA GeForce RTX 4090 D | 24564 MiB | 570.124.04 | 12.8 |

## 3. Python与Conda环境

| 项目 | 信息 |
| :--- | :--- |
| Python路径 | /root/autodl-tmp/envs/dreammatcher/bin/python |
| Python版本 | Python 3.10.19 |
| pip版本 | pip 26.0.1 from /root/autodl-tmp/envs/dreammatcher/lib/python3.10/site-packages/pip (python 3.10) |
| 当前Conda环境 | dreammatcher |
| Conda环境路径 | /root/autodl-tmp/envs/dreammatcher |

## 4. PyTorch与CUDA环境

| 项目 | 信息 |
| :--- | :--- |
| PyTorch版本 | 2.7.1+cu118 |
| PyTorch CUDA版本 | 11.8 |
| CUDA是否可用 | True |
| GPU数量 | 1 |
| GPU 0 | NVIDIA GeForce RTX 4090 D，显存 23.5 GB |

## 5. 主要依赖库版本

| 依赖库 | 版本 |
| :--- | :--- |
| diffusers | 0.20.0.dev0 |
| transformers | 4.36.2 |
| accelerate | 0.24.1 |
| safetensors | 0.7.0 |
| peft | 未安装或无法导入 |
| datasets | 4.8.4 |
| Pillow | 9.5.0 |
| numpy | 2.2.6 |
| torchvision | 0.22.1+cu118 |
| tensorboard | 2.20.0 |

## 6. 模型、代码与数据路径

| 项目 | 路径 | 是否存在 |
| :--- | :--- | :---: |
| Stable Diffusion v1.4模型 | `/root/autodl-tmp/models/stable-diffusion/stable-diffusion-v1-4` | 是 |
| DreamMatcher代码目录 | `/root/autodl-tmp/code/DreamMatcher` | 是 |
| LoRA训练数据目录 | `/root/autodl-tmp/datasets/cat_statue_lora/train` | 是 |
| LoRA权重目录 | `/root/autodl-tmp/outputs/lora_cat_statue_dm/checkpoints` | 是 |
| 实验输出目录 | `/root/autodl-tmp/outputs` | 是 |
| 实验日志目录 | `/root/autodl-tmp/logs` | 是 |

## 7. 代码版本信息

| 项目 | 信息 |
| :--- | :--- |
| DreamMatcher分支 | main |
| DreamMatcher提交 | 1c8fe19 |
| 未提交改动数量 | 60 |

## 8. 论文表格精简版

| 环境类别 | 项目 | 配置 |
| :--- | :--- | :--- |
| 硬件环境 | GPU | NVIDIA GeForce RTX 4090 D,  24564 MiB |
| 硬件环境 | CPU | Intel(R) Xeon(R) Platinum 8474C |
| 硬件环境 | 内存 | 1.0Ti |
| 软件环境 | 操作系统 | Ubuntu 22.04.1 LTS |
| 软件环境 | Python | Python 3.10.19 |
| 软件环境 | CUDA | 12.8 |
| 软件环境 | PyTorch | 2.7.1+cu118 |
| 软件环境 | diffusers | 0.20.0.dev0 |
| 软件环境 | transformers | 4.36.2 |
| 软件环境 | accelerate | 0.24.1 |
| 模型环境 | 基础模型 | Stable Diffusion v1.4 |
| 方法环境 | 主要方法 | LoRA、DreamMatcher、SAG |

