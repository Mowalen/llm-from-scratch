# LLM Zero to Hero

从零开始实现并理解 LLM 核心组件与训练方法。



## 项目结构

```text
llm-zeroToHero/
├── foundations/                  # 基础模块
│   ├── attention/                # MHA / GQA / KV Cache
│   ├── common/                   # 线性层、损失函数、BPE、SwiGLU 等
│   ├── functional/               # 激活函数实现
│   ├── norm/                     # LayerNorm / RMSNorm
│   └── pe/                       # 位置编码（RoPE）
├── architectures/                # 模型架构实现
│   ├── transformer/              # 原始 Transformer
│   ├── llama2/                   # Llama2 风格实现
│   ├── qwen2/                    # Qwen2 风格实现
│   └── qwen3/                    # Qwen3 / MoE 风格实现
├── training/                     # 训练与对齐算法
│   ├── lora/                     # LoRA 参数高效微调
│   ├── ppo/                      # PPO 与推导文档
│   └── rl_loss/                  # DPO / KL 等 RL 目标函数
├── pyproject.toml
└── README.md
```

## 已覆盖内容

- `foundations/attention/`: `mha.py`, `mha_kvcache.py`, `gqa.py`
- `foundations/common/`: `linear.py`, `linearLayer.py`, `softmax.py`, `cross_entropy.py`, `mse.py`, `bpe.py`, `swiglu.py`
- `architectures/transformer/`: 编码器-解码器结构与位置编码
- `architectures/llama2/`: Llama2 风格解码器组件
- `architectures/qwen2/`: Qwen2 配置、RMSNorm、MLP、Attention、Model
- `architectures/qwen3/`: Qwen3 配置、Attention、MoE、Model
- `training/lora/`: LoRA 线性层与注意力接入
- `training/ppo/`: PPO 训练代码与算法说明
- `training/rl_loss/`: DPO loss、KL loss 示例

## 快速开始

```bash
uv sync
```

或使用 pip:

```bash
pip install -e .
```

## 结构调整说明

本次重排主要做了三件事：

1. 将目录按职责收敛为 `foundations`、`architectures`、`training` 三层。
2. 统一顶层目录命名风格（小写），减少跨平台大小写歧义。
3. 保留原有实现文件，优先调整组织方式，不改变算法逻辑。


