import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from ._1_configuration import Qwen3Config

class Qwen3MoEGate(nn.Module):
    """
    MoE 门控机制 (Router)。
    计算路由概率并选择 Top-K 专家。
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.gate_network = nn.Linear(config.hidden_size, self.num_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # hidden_states: [batch_size, seq_len, hidden_size]
        # logic: [batch_size, seq_len, num_experts]
        logits = self.gate_network(hidden_states)
        
        # 计算路由概率
        routing_weights = F.softmax(logits, dim=-1, dtype=torch.float)
        
        # 选择 Top-K 专家
        # selected_experts_weights: [batch, seq, top_k]
        # selected_experts_indices: [batch, seq, top_k]
        selected_experts_weights, selected_experts_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        
        # 归一化选中的权重
        selected_experts_weights = selected_experts_weights / selected_experts_weights.sum(dim=-1, keepdim=True)
        
        # 转换回输入的数据类型
        selected_experts_weights = selected_experts_weights.to(hidden_states.dtype)
        
        return selected_experts_weights, selected_experts_indices, logits

class Qwen3MoE(nn.Module):
    """
    Qwen3 MoE 模块实现。
    特点：
    1. Shared Experts (共享专家): 总是被激活，捕获通用知识。
    2. Routed Experts (路由专家): 动态选择，捕获特定知识。
    """
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.intermediate_size = config.moe_intermediate_size
        
        # 1. 共享专家 (Always Activated)
        self.num_shared_experts = config.num_shared_experts
        if self.num_shared_experts > 0:
            self.shared_expert_gate = nn.Linear(self.hidden_size, config.shared_expert_intermediate_size, bias=False)
            self.shared_expert_up = nn.Linear(self.hidden_size, config.shared_expert_intermediate_size, bias=False)
            self.shared_expert_down = nn.Linear(config.shared_expert_intermediate_size, self.hidden_size, bias=False)
        
        # 2. 路由专家 (Routed Experts)
        self.gate = Qwen3MoEGate(config)
        
        # 专家网络列表。为了简单起见，这里使用 ModuleList。
        # 在实际的大规模训练中，通常会将专家权重合并为一个大张量以进行批量计算 (Grouped GEMM)。
        self.experts_gate = nn.ModuleList([
            nn.Linear(self.hidden_size, self.intermediate_size, bias=False) for _ in range(self.num_experts)
        ])
        self.experts_up = nn.ModuleList([
            nn.Linear(self.hidden_size, self.intermediate_size, bias=False) for _ in range(self.num_experts)
        ])
        self.experts_down = nn.ModuleList([
            nn.Linear(self.intermediate_size, self.hidden_size, bias=False) for _ in range(self.num_experts)
        ])
        
        self.act_fn = nn.SiLU()

    def forward(self, hidden_states: torch.Tensor):
        final_hidden_states = 0
        
        # --- 共享专家前向传播 ---
        if self.num_shared_experts > 0:
            shared_out = self.shared_expert_down(
                self.act_fn(self.shared_expert_gate(hidden_states)) * self.shared_expert_up(hidden_states)
            )
            final_hidden_states = final_hidden_states + shared_out

        # --- 路由专家前向传播 ---
        bsz, seq_len, hidden_dim = hidden_states.shape
        
        # 展平以便处理
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        
        # 获取路由信息
        # weights: [bsz*seq_len, top_k]
        # indices: [bsz*seq_len, top_k]
        routing_weights, selected_experts, _ = self.gate(hidden_states)
        routing_weights = routing_weights.view(-1, self.num_experts_per_tok)
        selected_experts = selected_experts.view(-1, self.num_experts_per_tok)
        
        # 初始化路由专家的输出
        routed_expert_output = torch.zeros_like(hidden_states_flat)
        
        # 这是一个简单的参考实现，通过循环遍历专家。
        # 优化提示: 生产级代码会使用 permute + grouped matmul 或 scatter/gather 操作。
        for expert_idx in range(self.num_experts):
            # 找到选择了当前专家 expert_idx 的 token 索引
            # mask: [total_tokens, top_k] boolean
            expert_mask = (selected_experts == expert_idx)
            
            if expert_mask.any():
                # 找到该专家对哪些 token 起作用，以及对应的权重索引
                # 我们需要知道是 top_k 中的第几个
                token_indices, top_k_indices = torch.where(expert_mask)
                
                # 提取这些 token 的 hidden states
                current_inputs = hidden_states_flat[token_indices]
                
                # 计算专家输出 (SwiGLU)
                gate = self.experts_gate[expert_idx](current_inputs)
                max_gate = self.act_fn(gate)
                up = self.experts_up[expert_idx](current_inputs)
                current_output = max_gate * up
                current_output = self.experts_down[expert_idx](current_output)
                
                # 获取对应的路由权重
                # routing_weights[token_indices, top_k_indices] 获取对应 token 在该专家上的权重
                current_weights = routing_weights[token_indices, top_k_indices].unsqueeze(-1)
                
                # 加权累加到总输出
                # 注意：一个 token 可能被多个专家处理，这里使用 index_add_ 更高效，但简单的 += 也可以
                routed_expert_output.index_add_(0, token_indices, current_output * current_weights)
        
        # 恢复形状
        routed_expert_output = routed_expert_output.view(bsz, seq_len, hidden_dim)
        
        # 合并共享专家和路由专家的输出
        final_hidden_states = final_hidden_states + routed_expert_output
        
        return final_hidden_states
