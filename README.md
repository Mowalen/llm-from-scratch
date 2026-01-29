# 🚀 LLM Zero to Hero

从零开始掌握大模型核心技术。本项目旨在通过手动实现经典架构与算法，帮助开发者深入理解 LLM（Large Language Model）的底层原理与实现细节。

## 📖 关于本项目 (About)

本仓库汇集了 LLM 相关的核心技术实现，代码结构清晰，适合用于学习、练手以及作为大模型开发的参考。项目涵盖了从基础的 Transformer 架构到现代化的 Llama 2 模型，再到 LoRA 微调与 PPO 强化学习的全栈技术。

---

## ✅ 已实现 (Implemented)

### 🏗️ 核心架构 (Core Architectures)

#### 🔷 Transformer
基于 *Attention Is All You Need* 的经典架构复现，所有 LLM 的基石。
- **基础组件**: 多头自注意力机制 (Multi-Head Self-Attention)
- **编码**: 正弦位置编码 (Sinusoidal Positional Encoding)
- **网络**: 前馈神经网络 (Feed-Forward Networks) & LayerNorm
- **目录**: `Transformer/`

#### 🦙 Llama 2
参考 Meta Llama 2 的现代化大模型架构实现。
- **位置编码**: 旋转位置编码 (**RoPE**, Rotary Positional Embeddings) - 位于 `pe/rope.py`
- **激活函数**: **SwiGLU** 激活函数
- **归一化**: **RMSNorm** (Root Mean Square Normalization)
- **注意力**: 分组查询注意力 (**GQA**, Grouped Query Attention)
- **目录**: `Llama2/`

### 🔧 训练与微调 (Training & Fine-tuning)

#### ⚡ LoRA (Low-Rank Adaptation)
大模型参数高效微调 (PEFT) 技术。
- **原理**: 通过低秩分解矩阵适配下游任务，大幅降低显存需求。
- **实现**: 自定义 LoRA 层与 Attention 的集成。
- **目录**: `Lora/`

#### 🎯 PPO (Proximal Policy Optimization)
RLHF (Reinforcement Learning from Human Feedback) 流程中的核心强化学习算法。
- **架构**: 完整的 Actor-Critic 架构
- **优化**: 策略梯度 (Policy Gradient) 与裁剪目标函数 (Clipped Surrogate Objective)
- **文档**: [📄 PPO 算法详解与推导](ppo/PPO算法详解.md) (推荐阅读)
- **目录**: `ppo/`

### ⚙️ 基础组件 (Common Utils)
- **Common**: 包含常用的深度学习基础算子实现，如 Linear, Softmax, CrossEntropy, MSE 等。
- **目录**: `common/`

---

## 🔜 路线图 (Roadmap)

- [ ] **🎲 GRPO (Group Relative Policy Optimization)**
  - DeepSeek-R1 背后的核心强化学习算法
  - 旨在提升样本效率与训练稳定性

- [ ] **🧩 MoE (Mixture of Experts)**
  - 稀疏混合专家模型
  - 实现条件计算 (Conditional Computation) 与可扩展架构
