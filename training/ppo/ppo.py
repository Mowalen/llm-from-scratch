# 导入PyTorch深度学习框架
import torch
# 导入神经网络模块
import torch.nn as nn
# 导入多元正态分布，用于连续动作空间
from torch.distributions import MultivariateNormal
# 导入类别分布，用于离散动作空间
from torch.distributions import Categorical

################################## 设置设备 ##################################
# 默认使用CPU设备
device = torch.device('cpu')
# 检查是否有可用的CUDA GPU
if(torch.cuda.is_available()): 
    # 如果有GPU，使用第一个GPU设备
    device = torch.device('cuda:0') 
    # 清空GPU缓存，释放未使用的显存
    torch.cuda.empty_cache()
    # 打印当前使用的GPU设备名称
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    # 如果没有GPU，提示使用CPU
    print("Device set to : cpu")

################################## 经验回放缓冲区 ##################################
class RolloutBuffer:
    """经验回放缓冲区类，用于存储智能体与环境交互过程中的经验数据"""
    def __init__(self):
        # 存储动作序列
        self.actions = []
        # 存储状态序列
        self.states = []
        # 存储动作的对数概率序列
        self.logprobs = []
        # 存储奖励序列
        self.rewards = []
        # 存储状态价值序列 网络对每个状态的价值估计 V(s)
        self.state_values = []
        # 存储是否为终止状态的标志序列
        self.is_terminals = []
    
    def clear(self):
        """清空缓冲区中的所有数据"""
        # 删除动作列表中的所有元素
        del self.actions[:]
        # 删除状态列表中的所有元素
        del self.states[:]
        # 删除对数概率列表中的所有元素
        del self.logprobs[:]
        # 删除奖励列表中的所有元素
        del self.rewards[:]
        # 删除状态价值列表中的所有元素
        del self.state_values[:]
        # 删除终止状态标志列表中的所有元素
        del self.is_terminals[:]

################################## Actor-Critic 网络 ##################################
# Actor-Critic网络结构，同时包含策略网络(Actor)和价值网络(Critic)
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        """
        初始化Actor-Critic网络
        参数:
            state_dim: 状态空间的维度
            action_dim: 动作空间的维度
            has_continuous_action_space: 是否为连续动作空间
            action_std_init: 连续动作空间的初始标准差
        """
        # 调用父类nn.Module的初始化方法
        super(ActorCritic, self).__init__()

        # 保存是否为连续动作空间的标志
        self.has_continuous_action_space = has_continuous_action_space
        
        # 如果是连续动作空间
        if has_continuous_action_space:
            # 保存动作维度
            self.action_dim = action_dim
            # 初始化动作方差，创建一个包含action_dim个元素的张量，每个元素值为action_std_init的平方
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)
        
        # Actor 网络：决定动作
        # 如果是连续动作空间
        if has_continuous_action_space :
            # 构建连续动作的Actor网络
            self.actor = nn.Sequential(
                            # 输入层到第一隐藏层，state_dim -> 64
                            nn.Linear(state_dim, 64),
                            # Tanh激活函数，输出范围[-1, 1]
                            nn.Tanh(),
                            # 第一隐藏层到第二隐藏层，64 -> 64
                            nn.Linear(64, 64),
                            # Tanh激活函数
                            nn.Tanh(),
                            # 第二隐藏层到输出层，64 -> action_dim
                            nn.Linear(64, action_dim),
                            # Tanh激活函数，确保输出在[-1, 1]范围内
                            nn.Tanh()
                        )
        else:
            # 构建离散动作的Actor网络
            self.actor = nn.Sequential(
                            # 输入层到第一隐藏层，state_dim -> 64
                            nn.Linear(state_dim, 64),
                            # Tanh激活函数
                            nn.Tanh(),
                            # 第一隐藏层到第二隐藏层，64 -> 64
                            nn.Linear(64, 64),
                            # Tanh激活函数
                            nn.Tanh(),
                            # 第二隐藏层到输出层，64 -> action_dim
                            nn.Linear(64, action_dim),
                            # Softmax激活函数，将输出转换为概率分布，所有动作概率和为1
                            nn.Softmax(dim=-1)
                        )
        
        # Critic 网络：评价状态价值 V(s)
        self.critic = nn.Sequential(
                        # 输入层到第一隐藏层，state_dim -> 64
                        nn.Linear(state_dim, 64),
                        # Tanh激活函数
                        nn.Tanh(),
                        # 第一隐藏层到第二隐藏层，64 -> 64
                        nn.Linear(64, 64),
                        # Tanh激活函数
                        nn.Tanh(),
                        # 第二隐藏层到输出层，64 -> 1，输出单一的状态价值
                        nn.Linear(64, 1)
                    )
        
    def set_action_std(self, new_action_std):
        """
        设置新的动作标准差（仅适用于连续动作空间）
        参数:
            new_action_std: 新的标准差值
        """
        # 如果是连续动作空间
        if self.has_continuous_action_space:
            # 更新动作方差为新标准差的平方
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            # 如果是离散动作空间，打印警告信息
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")

    def forward(self):
        """前向传播方法，未实现，需要在子类中实现"""
        raise NotImplementedError
    
    def act(self, state):
        """
        根据当前状态选择动作
        参数:
            state: 当前状态
        返回:
            action: 选择的动作
            action_logprob: 动作的对数概率
            state_val: 状态价值估计
        """
        # 如果是连续动作空间
        if self.has_continuous_action_space:
            # 通过Actor网络得到动作均值
            action_mean = self.actor(state)
            # 构建协方差矩阵（对角矩阵），并增加批次维度
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            # 创建多元正态分布
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            # 如果是离散动作空间，通过Actor网络得到动作概率
            action_probs = self.actor(state)
            # 创建类别分布
            dist = Categorical(action_probs)

        # 从分布中采样一个动作
        action = dist.sample()
        # 计算该动作的对数概率
        action_logprob = dist.log_prob(action)
        # 通过Critic网络得到状态价值估计
        state_val = self.critic(state)

        # 返回动作、对数概率和状态价值，使用detach()切断梯度
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        """
        评估给定状态-动作对
        参数:
            state: 状态批次
            action: 动作批次
        返回:
            action_logprobs: 动作的对数概率
            state_values: 状态价值估计
            dist_entropy: 策略分布的熵
        """
        # 如果是连续动作空间
        if self.has_continuous_action_space:
            # 通过Actor网络得到动作均值
            action_mean = self.actor(state)
            # 将动作方差扩展为与action_mean相同的形状
            action_var = self.action_var.expand_as(action_mean)
            # 构建批次协方差矩阵（对角嵌入）
            cov_mat = torch.diag_embed(action_var).to(device)
            # 创建多元正态分布
            dist = MultivariateNormal(action_mean, cov_mat)
            # 如果动作维度为1，需要调整动作的形状
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            # 如果是离散动作空间，通过Actor网络得到动作概率
            action_probs = self.actor(state)
            # 创建类别分布
            dist = Categorical(action_probs)

        # 计算给定动作的对数概率
        action_logprobs = dist.log_prob(action)
        # 计算分布的熵（用于鼓励探索）
        dist_entropy = dist.entropy()
        # 通过Critic网络得到状态价值估计
        state_values = self.critic(state)
        
        # 返回对数概率、状态价值和熵
        return action_logprobs, state_values, dist_entropy

################################## PPO 算法逻辑 ##################################
class PPO:
    """Proximal Policy Optimization（近端策略优化）算法实现类"""
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std_init=0.6):
        """
        初始化PPO算法
        参数:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            lr_actor: Actor网络的学习率
            lr_critic: Critic网络的学习率
            gamma: 折扣因子
            K_epochs: 每次更新时的训练轮数
            eps_clip: PPO裁剪参数
            has_continuous_action_space: 是否为连续动作空间
            action_std_init: 连续动作空间的初始标准差，默认0.6
        """

        # 保存是否为连续动作空间的标志
        self.has_continuous_action_space = has_continuous_action_space

        # 如果是连续动作空间，保存动作标准差
        if has_continuous_action_space:
            self.action_std = action_std_init

        # 保存折扣因子（用于计算未来奖励的权重）
        self.gamma = gamma
        # 保存PPO裁剪参数（限制策略更新幅度）
        self.eps_clip = eps_clip
        # 保存每次更新时的训练轮数
        self.K_epochs = K_epochs
        
        # 创建经验回放缓冲区
        self.buffer = RolloutBuffer()

        # 创建当前策略网络（用于训练更新）
        self.policy = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # 创建优化器，分别为Actor和Critic设置不同的学习率
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        # 创建旧策略网络（用于收集经验和计算重要性采样比率）
        self.policy_old = ActorCritic(state_dim, action_dim, has_continuous_action_space, action_std_init).to(device)
        # 将当前策略的参数复制到旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 创建均方误差损失函数，用于计算价值函数的损失
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        """
        设置新的动作标准差（仅用于连续动作空间）
        参数:
            new_action_std: 新的标准差值
        """
        # 如果是连续动作空间
        if self.has_continuous_action_space:
            # 更新保存的标准差
            self.action_std = new_action_std
            # 更新当前策略的动作标准差
            self.policy.set_action_std(new_action_std)
            # 更新旧策略的动作标准差
            self.policy_old.set_action_std(new_action_std)
    
    def decay_action_std(self, action_std_decay_rate, min_action_std):
        """
        衰减动作标准差（用于逐渐减少探索）
        参数:
            action_std_decay_rate: 标准差衰减率
            min_action_std: 最小标准差限制
        """
        # 如果是连续动作空间
        if self.has_continuous_action_space:
            # 减小当前标准差
            self.action_std = self.action_std - action_std_decay_rate
            # 四舍五入到4位小数
            self.action_std = round(self.action_std, 4)
            # 如果标准差小于等于最小值，设置为最小值
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
            # 更新两个策略网络的标准差
            self.set_action_std(self.action_std)

    def select_action(self, state):
        """
        根据当前状态选择动作，并将经验存入缓冲区
        参数:
            state: 当前状态
        返回:
            选择的动作（numpy数组或标量）
        """
        # 如果是连续动作空间
        if self.has_continuous_action_space:
            # 在无需计算梯度的上下文中执行
            with torch.no_grad():
                # 将状态转换为FloatTensor并移到指定设备
                state = torch.FloatTensor(state).to(device)
                # 使用旧策略选择动作
                action, action_logprob, state_val = self.policy_old.act(state)

            # 将状态存入缓冲区
            self.buffer.states.append(state)
            # 将动作存入缓冲区
            self.buffer.actions.append(action)
            # 将动作对数概率存入缓冲区
            self.buffer.logprobs.append(action_logprob)
            # 将状态价值存入缓冲区
            self.buffer.state_values.append(state_val)

            # 将动作转换为numpy数组并展平返回
            return action.detach().cpu().numpy().flatten()
        else:
            # 如果是离散动作空间，在无需计算梯度的上下文中执行
            with torch.no_grad():
                # 将状态转换为FloatTensor并移到指定设备
                state = torch.FloatTensor(state).to(device)
                # 使用旧策略选择动作
                action, action_logprob, state_val = self.policy_old.act(state)
            
            # 将状态存入缓冲区
            self.buffer.states.append(state)
            # 将动作存入缓冲区
            self.buffer.actions.append(action)
            # 将动作对数概率存入缓冲区
            self.buffer.logprobs.append(action_logprob)
            # 将状态价值存入缓冲区
            self.buffer.state_values.append(state_val)

            # 返回动作的整数值
            return action.item()

    def update(self):
        """
        使用PPO算法更新策略网络和价值网络
        """
        # 1. 蒙特卡洛计算回报（从后向前计算折扣累积奖励）
        rewards = []
        # 初始化折扣奖励
        discounted_reward = 0
        # 反向遍历缓冲区中的奖励和终止标志
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            # 如果到达终止状态，重置折扣奖励
            if is_terminal:
                discounted_reward = 0
            # 计算折扣累积奖励：R_t = r_t + gamma * R_{t+1}
            discounted_reward = reward + (self.gamma * discounted_reward)
            # 将折扣奖励插入到列表开头
            rewards.insert(0, discounted_reward)
            
        # 2. 归一化奖励（提高训练稳定性）
        # 将奖励列表转换为张量
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # 归一化：(x - mean) / (std + epsilon)，epsilon防止除零
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 转换 buffer 为 tensor
        # 将状态列表堆叠成张量并压缩，detach切断梯度
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        # 将动作列表堆叠成张量并压缩
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        # 将对数概率列表堆叠成张量并压缩
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        # 将状态价值列表堆叠成张量并压缩
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # 3. 计算优势函数（Advantage = 实际回报 - 状态价值估计）
        advantages = rewards.detach() - old_state_values.detach()
        
        # 4. K 轮优化（在同一批数据上进行多轮训练）
        for _ in range(self.K_epochs):
            # 使用当前策略重新评估旧的状态-动作对
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            # 压缩状态价值张量
            state_values = torch.squeeze(state_values)
            
            # 计算重要性采样比率 ratio = π_new / π_old = exp(log π_new - log π_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算PPO的代理损失 (Surrogate Loss)
            # surr1: 未裁剪的策略梯度目标
            surr1 = ratios * advantages
            # surr2: 裁剪后的策略梯度目标，限制策略更新幅度在[1-eps, 1+eps]范围内
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # 总损失 = -策略损失 + 价值损失 - 熵bonus
            # min(surr1, surr2): 选择更保守的目标，防止策略更新过大
            # 0.5 * MSE: 价值函数的均方误差损失
            # -0.01 * entropy: 熵正则化，鼓励探索（负号因为我们要最大化熵）
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # 清空之前的梯度
            self.optimizer.zero_grad()
            # 反向传播计算梯度（取平均损失）
            loss.mean().backward()
            # 更新网络参数
            self.optimizer.step()
            
        # 将当前策略的参数复制到旧策略（用于下一轮经验收集）
        self.policy_old.load_state_dict(self.policy.state_dict())
        # 清空经验缓冲区
        self.buffer.clear()
    
    def save(self, checkpoint_path):
        """
        保存模型参数
        参数:
            checkpoint_path: 保存路径
        """
        # 保存旧策略的状态字典到指定路径
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        """
        加载模型参数
        参数:
            checkpoint_path: 模型文件路径
        """
        # 加载参数到旧策略（map_location用于处理CPU/GPU设备映射）
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        # 加载参数到当前策略，保持两者同步
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))