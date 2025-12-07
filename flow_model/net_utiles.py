import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torch.autograd import Variable
import math
import numpy as np
torch.set_default_dtype(torch.float32)


"""
Naive model Class
"""
class Simple_NN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, output_act, pred_type):
        super(Simple_NN, self).__init__()
        if network == 'mlp':
            self.net = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act)
        elif network == 'simple_hypernetwork':
            self.net = Simple_Hypernetwork(input_dim, output_dim, hidden_dim, num_layers, output_act)
        elif network == 'full_hypernetwork':
            self.net = Full_Hypernetwork(input_dim, output_dim, hidden_dim, num_layers, output_act)
        elif network == 'att':
            self.net = ATT(input_dim, 1, hidden_dim, num_layers, output_act, pred_type=pred_type)
        else:
            NotImplementedError
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def loss(self, y_pred, y_target):
        return self.criterion(y_pred, y_target)


class GMM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, num_cluster, output_act, pred_type):
        super(GMM, self).__init__()
        self.num_cluster = num_cluster
        self.output_dim = output_dim
        if network == 'mlp':
            self.Predictor = MLP(input_dim, output_dim * num_cluster, hidden_dim, num_layers, output_act)
            self.Classifier = MLP(input_dim, num_cluster, hidden_dim, num_layers, 'gumbel', output_dim)
        elif network == 'att':
            self.Predictor = ATT(input_dim, num_cluster, hidden_dim, num_layers, output_act, pred_type=pred_type)
            self.Classifier = ATT(input_dim, num_cluster, hidden_dim, num_layers, 'gumbel', 1, pred_type=pred_type)
        else:
            NotImplementedError
        self.criterion = nn.MSELoss()

    def forward(self, x):
        y_pred = self.Predictor(x).view(x.shape[0], -1, self.num_cluster)
        return y_pred

    def loss(self, x, y_pred, y_target):
        c_pred = self.Classifier(x, y_target).view(x.shape[0], -1, self.num_cluster)
        y_pred = (c_pred * y_pred).sum(-1)
        loss = self.criterion(y_pred, y_target)
        return loss
    
    def hindsight_loss(self, x, y_pred, y_target):
        loss = (y_pred - y_target.unsqueeze(-1))**2
        loss = loss.mean(dim=1)
        loss = loss.min(dim=1)[0]
        return loss.mean()



"""
GNN model Class
"""
class GNN(nn.Module):
    """
    TSP:
        Input: node: B*N*2, edge: 0
        Output: edge: B*N*N
    MCP/MIS/MCUT:
        Input: node: 0, edge: B*N*N
        Output: node: B*N
    """

    def __init__(self, hidden_dim, num_layer):
        self.num_layer = num_layer
        self.node_emb = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.edge_emb = ScalarEmbeddingSine(hidden_dim, normalize=False)
        self.gnn_emb = nn.ModuleList(
            [GNNLayer(hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True) for _
             in range(num_layer)])
        self.out_emb = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, node, edge):
        node_emb = self.node_emb(node)
        edge_emb = self.edge_emb(edge)
        for layer in zip(self.gnn_emb):
            node_emb, edge_emb = layer(node_emb, edge_emb)
        node_pred = self.out_emb(node_emb)
        return node_pred



"""
VAE Encoder Class with Dual Input (x and y_target)
"""
import torch
import torch.nn as nn

class VAE_Encoder(nn.Module):
    def __init__(self, input_dim, output_dim, latent_dim, hidden_dim, num_layers, act='relu'):
        super().__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 'tanh': nn.Tanh()}
        act_fn = act_list[act]

        # x-branch
        self.x_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim)
        )

        # y-branch
        self.y_encoder = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.LayerNorm(hidden_dim)
        )

        # FiLM 调制：用 y 特征调制 x 特征（也可反过来）
        self.gamma = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))
        self.beta  = nn.Sequential(nn.Linear(hidden_dim, hidden_dim))

        # 深层融合：Residual MLP
        blocks = []
        for _ in range(max(1, num_layers)):
            blocks += [
                nn.Linear(hidden_dim, hidden_dim),
                act_fn,
                nn.LayerNorm(hidden_dim)
            ]
        self.fusion_net = nn.Sequential(*blocks)

        self.mean_layer = nn.Linear(hidden_dim, latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, y_target):
        x_feat = self.x_encoder(x)           # [B,H]
        y_feat = self.y_encoder(y_target)    # [B,H]

        # FiLM: 让 y 决定在条件 x 下需要的残差信息
        gamma = self.gamma(y_feat)           # [B,H]
        beta  = self.beta(y_feat)            # [B,H]
        fused = gamma * x_feat + beta        # [B,H]

        # 残差堆叠（小技巧：加一个 skip 更稳）
        deep = self.fusion_net(fused)
        deep = deep + fused

        mean   = self.mean_layer(deep)
        logvar = self.logvar_layer(deep)     # 训练时可 clamp：logvar = logvar.clamp(-20, 20)
        return mean, logvar
    
    def encode_from_condition(self, x):
        """
        只基于条件x进行编码，用于推理时（无需y_target）
        
        这样做的好处：
        1. 推理时可以利用encoder学到的条件信息
        2. 避免训练-推理不一致
        3. 生成的分布是 q(z|x) 而非简单的 p(z)=N(0,I)
        
        Args:
            x: [batch_size, input_dim] 条件输入（负荷、碳税等）
            
        Returns:
            mean: [batch_size, latent_dim] 条件分布的均值
            logvar: [batch_size, latent_dim] 条件分布的对数方差
        """
        x_feat = self.x_encoder(x)           # [B,H]
        # 不使用y_target，直接通过fusion_net处理x特征
        deep = self.fusion_net(x_feat)
        deep = deep + x_feat  # skip connection
        
        mean   = self.mean_layer(deep)
        logvar = self.logvar_layer(deep)
        return mean, logvar


"""
Generative Adversarial model Class
"""
class VAE(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(VAE, self).__init__() 
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        if network == 'mlp':
            # 使用新的VAE_Encoder，能够同时处理x和y_target
            self.Encoder = MLP(input_dim, latent_dim*2, hidden_dim, num_layers, None)
            # self.Encoder = VAE_Encoder(input_dim, output_dim, latent_dim, hidden_dim, num_layers, act='relu')
            self.Decoder = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
        elif network == 'att':
            NotImplementedError
        else:
            NotImplementedError
        self.criterion = nn.MSELoss()          

    def forward(self, x, z=None, use_mean=False):
        """
        VAE的前向传播（推理）
        
        Args:
            x: [batch_size, input_dim] 条件输入
            z: [batch_size, latent_dim] 可选的潜在向量
            use_mean: 是否使用均值而非采样（用于确定性推理）
            
        Returns:
            y_pred: [batch_size, output_dim] 预测输出
        """
        if z is None:
            # z = torch.randn(size=[x.shape[0], self.latent_dim]).to(x.device)
            # 使用encoder基于条件x预测潜在分布
            if hasattr(self.Encoder, 'encode_from_condition'):
                mean, logvar = self.Encoder.encode_from_condition(x)
            else:
                para = self.Encoder(x)
                mean, logvar = torch.chunk(para, 2, dim=-1)
            
            if use_mean:
                # 确定性推理：直接使用均值
                z = mean
            else:
                # 随机推理：从预测的分布中采样
                z = self.reparameterize(mean, logvar)
        else:
            z = z.to(x.device)
        
        y_pred = self.Decoder(x, z)
        return y_pred

    def reparameterize(self, mean, logvar):
        z = torch.randn_like(mean).to(mean.device)
        return mean + z * torch.exp(0.5 * logvar)

    def encoder_decode(self, x):
        """
        编码-解码过程，将x和y_target编码到潜在空间，然后从潜在空间解码
        
        Args:
            x: [batch_size, input_dim] 条件输入
            y_target: [batch_size, output_dim] 目标值（电压和相角）
            
        Returns:
            y_recon: [batch_size, output_dim] 重构的目标值
            mean: [batch_size, latent_dim] 潜在分布的均值
            logvar: [batch_size, latent_dim] 潜在分布的对数方差
        """
        # 使用新的Encoder，同时编码x和y_target
        para = self.Encoder(x)
        mean, logvar = torch.chunk(para, 2, dim=-1)
        # mean, logvar = self.Encoder(x, y_target)  # 这个是加入了生成的data的信息
        z = self.reparameterize(mean, logvar)
        y_recon = self.Decoder(x, z)
        return y_recon, mean, logvar

    def loss(self, y_recon, y_target, mean, logvar, beta=1.0):
        """
        VAE损失函数 = 重建损失 + beta * KL散度
        
        Args:
            y_recon: 重建的输出 [batch_size, output_dim]
            y_target: 目标输出 [batch_size, output_dim]
            mean: 潜在分布均值 [batch_size, latent_dim]
            logvar: 潜在分布对数方差 [batch_size, latent_dim]
            beta: KL散度权重，默认为1.0 (beta-VAE)
        """
        # 重建损失 (MSE)
        recon_loss = self.criterion(y_recon, y_target)
        
        # KL散度：先对latent_dim求和，再对batch取平均
        # KL(q(z|x,y) || p(z)) = -0.5 * sum(1 + log(var) - mean^2 - var)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        kl_div = torch.mean(kl_div)
        
        # 总损失 = 重建损失 + beta * KL散度
        return recon_loss + beta * kl_div


class LatentFlowVAE(nn.Module):
    """
    Latent Flow VAE: 在VAE的潜在空间中运行Flow Matching
    
    架构设计:
    - VAE Encoder: (x, y_target) → (mean, logvar) → z_1 (目标分布)
    - Latent Flow: 学习 z_0 ~ N(0,I) → z_1 的映射 (Rectified Flow)
    - VAE Decoder: (x, z) → y_pred (学习满足约束的映射)
    
    训练流程:
    - 阶段1: 预训练VAE (encoder + decoder)
    - 阶段2: 固定decoder, 训练Latent Flow
    - 阶段3 (可选): 联合微调
    
    优势:
    - Flow在低维空间运行 (latent_dim vs output_dim)
    - Decoder专门学习 z → 满足约束的 y 映射
    - 可以端到端优化,将约束损失反传到Flow
    """
    
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, 
                 latent_dim=64, output_act=None, pred_type=None, time_step=1000):
        """
        初始化 Latent Flow VAE
        
        Args:
            network: 网络类型 ('mlp', 'att')
            input_dim: 输入条件维度 (负荷等)
            output_dim: 输出维度 (电压+相角, 如236)
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
            latent_dim: 潜在空间维度 (如64)
            output_act: 输出激活函数
            pred_type: 预测类型
            time_step: 时间步数 (用于时间嵌入)
        """
        super(LatentFlowVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.time_step = time_step
        self.min_sd = 0.01  # Flow的最小标准差
        
        if network == 'mlp':
            # VAE Encoder: x → (mean, logvar)
            # 输入: x (条件), 输出: latent_dim * 2 (mean + logvar)
            self.encoder = MLP(input_dim, latent_dim * 2, hidden_dim, num_layers, None)
            
            # VAE Decoder: (x, z) → y
            # 使用FiLM方式: x作为条件, z作为潜在输入
            self.decoder = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
            
            # Latent Flow Network: 预测速度场 v(x, z_t, t)
            # 输入: x (条件) + z_t (当前潜在状态), 输出: latent_dim (速度向量)
            # 使用FiLM方式处理: x作为条件, z_t作为输入, t作为时间
            self.flow_net = MLP(input_dim, latent_dim, hidden_dim, num_layers, None, latent_dim)
        else:
            raise NotImplementedError(f"Network type {network} not supported for LatentFlowVAE")
        
        self.criterion = nn.MSELoss()
        
        # 训练阶段控制
        self.training_stage = 'vae'  # 'vae', 'flow', 'joint'
        
    def set_training_stage(self, stage):
        """
        设置训练阶段
        
        Args:
            stage: 'vae' - 只训练VAE
                   'flow' - 固定decoder, 只训练flow
                   'joint' - 联合训练
        """
        assert stage in ['vae', 'flow', 'joint']
        self.training_stage = stage
        
        if stage == 'flow':
            # 固定encoder和decoder
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.flow_net.parameters():
                param.requires_grad = True
        elif stage == 'vae':
            # 只训练encoder和decoder
            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
            for param in self.flow_net.parameters():
                param.requires_grad = False
        else:  # joint
            # 所有参数都训练
            for param in self.parameters():
                param.requires_grad = True
    
    def encode(self, x):
        """
        编码: x → (mean, logvar)
        
        Args:
            x: [batch_size, input_dim] 条件输入
            
        Returns:
            mean: [batch_size, latent_dim] 均值
            logvar: [batch_size, latent_dim] 对数方差
        """
        para = self.encoder(x)
        mean, logvar = torch.chunk(para, 2, dim=-1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        """
        重参数化采样: z = mean + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps
    
    def decode(self, x, z):
        """
        解码: (x, z) → y
        
        Args:
            x: [batch_size, input_dim] 条件输入
            z: [batch_size, latent_dim] 潜在变量
            
        Returns:
            y_pred: [batch_size, output_dim] 预测输出
        """
        return self.decoder(x, z)
    
    def flow_forward(self, z_1, z_0, t):
        """
        Flow前向过程: 计算插值点 z_t 和目标速度
        使用 Rectified Flow: z_t = t * z_1 + (1-t) * z_0
        
        Args:
            z_1: [batch_size, latent_dim] 目标分布 (从encoder得到)
            z_0: [batch_size, latent_dim] 噪声 N(0,I)
            t: [batch_size, 1] 时间步
            
        Returns:
            z_t: [batch_size, latent_dim] 当前插值点
            v_target: [batch_size, latent_dim] 目标速度向量 (z_1 - z_0)
        """
        # Rectified Flow 插值
        z_t = t * z_1 + (1 - t) * z_0
        # 目标速度: 从 z_0 到 z_1 的直线
        v_target = z_1 - z_0
        return z_t, v_target
    
    def predict_velocity(self, x, z_t, t):
        """
        预测速度场 v(x, z_t, t)
        
        Args:
            x: [batch_size, input_dim] 条件
            z_t: [batch_size, latent_dim] 当前潜在状态
            t: [batch_size, 1] 时间步
            
        Returns:
            v_pred: [batch_size, latent_dim] 预测的速度
        """
        return self.flow_net(x, z_t, t)
    
    def flow_backward(self, x, z_0=None, num_steps=50, method='Euler'):
        """
        Flow反向采样: 从 z_0 ~ N(0,I) 采样到 z_T
        
        Args:
            x: [batch_size, input_dim] 条件输入
            z_0: [batch_size, latent_dim] 初始噪声 (如果为None则随机生成)
            num_steps: ODE求解步数
            method: 求解方法 ('Euler', 'RK4')
            
        Returns:
            z_T: [batch_size, latent_dim] 最终潜在变量
        """
        batch_size = x.shape[0]
        device = x.device
        
        if z_0 is None:
            z_0 = torch.randn(batch_size, self.latent_dim, device=device)
        
        z_t = z_0
        dt = 1.0 / num_steps
        
        with torch.no_grad():
            for step in range(num_steps):
                t = torch.full((batch_size, 1), step * dt, device=device)
                
                if method == 'Euler':
                    v = self.predict_velocity(x, z_t, t)
                    z_t = z_t + v * dt
                elif method == 'RK4':
                    # Runge-Kutta 4阶方法
                    k1 = self.predict_velocity(x, z_t, t)
                    k2 = self.predict_velocity(x, z_t + 0.5 * dt * k1, t + 0.5 * dt)
                    k3 = self.predict_velocity(x, z_t + 0.5 * dt * k2, t + 0.5 * dt)
                    k4 = self.predict_velocity(x, z_t + dt * k3, t + dt)
                    z_t = z_t + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
                else:
                    raise ValueError(f"Unknown method: {method}")
        
        return z_t
    
    def forward(self, x, z=None, use_mean=False, num_steps=50):
        """
        前向传播 (推理模式)
        
        Args:
            x: [batch_size, input_dim] 条件输入
            z: [batch_size, latent_dim] 可选的潜在向量
            use_mean: 如果True, 使用encoder均值而非flow采样
            num_steps: Flow采样步数
            
        Returns:
            y_pred: [batch_size, output_dim] 预测输出
        """
        if z is not None:
            # 直接使用提供的z
            z_final = z
        elif use_mean:
            # 使用encoder的均值 (不通过flow)
            mean, _ = self.encode(x)
            z_final = mean
        else:
            # 通过flow采样
            z_final = self.flow_backward(x, num_steps=num_steps)
        
        y_pred = self.decode(x, z_final)
        return y_pred
    
    def vae_loss(self, x, y_target, beta=1.0, constraint_fn=None, constraint_weight=0.0):
        """
        VAE损失函数 (阶段1训练) - 支持约束感知训练
        
        Args:
            x: [batch_size, input_dim] 条件输入
            y_target: [batch_size, output_dim] 目标输出
            beta: KL散度权重
            constraint_fn: 约束函数 constraint_fn(Vm, Va, x_input, reduction) (可选)
            constraint_weight: 约束损失权重 (默认0, 设为>0启用约束感知)
            
        Returns:
            loss: 总损失
            loss_dict: 各项损失的字典
            y_recon: 重建输出
            mean: 潜在均值
            logvar: 潜在对数方差
        """
        device = x.device
        
        # 编码
        mean, logvar = self.encode(x)
        
        # 重参数化采样
        z = self.reparameterize(mean, logvar)
        
        # 解码
        y_recon = self.decode(x, z)
        
        # 重建损失
        recon_loss = self.criterion(y_recon, y_target)
        
        # KL散度
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1)
        kl_div = torch.mean(kl_div)
        
        # === 约束损失 (可选) ===
        constraint_loss = torch.tensor(0.0, device=device)
        if constraint_fn is not None and constraint_weight > 0:
            # 计算约束违反
            half_dim = self.output_dim // 2
            Vm = y_recon[:, :half_dim]
            Va = y_recon[:, half_dim:]
            # 约束函数需要原始条件输入 (不包含碳税)
            constraint_result = constraint_fn(Vm, Va, x, reduction='mean')
            # 处理返回值可能是元组的情况 (loss, details)
            if isinstance(constraint_result, tuple):
                constraint_loss = constraint_result[0]
            else:
                constraint_loss = constraint_result
        
        # 总损失
        loss = recon_loss + beta * kl_div + constraint_weight * constraint_loss
        
        loss_dict = {
            'recon_loss': recon_loss.item(),
            'kl_div': kl_div.item(),
            'constraint_loss': constraint_loss.item() if isinstance(constraint_loss, torch.Tensor) else constraint_loss,
            'total_loss': loss.item()
        }
        
        return loss, loss_dict, y_recon, mean, logvar
    
    def flow_loss(self, x, y_target):
        """
        Flow损失函数 (阶段2训练)
        
        Args:
            x: [batch_size, input_dim] 条件输入
            y_target: [batch_size, output_dim] 目标输出 (用于获取encoder分布)
            
        Returns:
            loss: Flow匹配损失
            loss_dict: 损失字典
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 获取目标分布 z_1 (从encoder)
        with torch.no_grad():
            mean, logvar = self.encode(x)
            z_1 = self.reparameterize(mean, logvar)
        
        # 采样噪声 z_0 ~ N(0,I)
        z_0 = torch.randn(batch_size, self.latent_dim, device=device)
        
        # 随机时间步
        t = torch.rand(batch_size, 1, device=device)
        
        # Flow前向: 获取插值点和目标速度
        z_t, v_target = self.flow_forward(z_1, z_0, t)
        
        # 预测速度
        v_pred = self.predict_velocity(x, z_t, t)
        
        # 速度匹配损失
        flow_loss = self.criterion(v_pred, v_target)
        
        loss_dict = {
            'flow_loss': flow_loss.item()
        }
        
        return flow_loss, loss_dict
    
    def joint_loss(self, x, y_target, beta=1.0, flow_weight=1.0, constraint_fn=None, constraint_weight=0.0):
        """
        联合损失函数 (阶段3训练)
        
        Args:
            x: [batch_size, input_dim] 条件输入
            y_target: [batch_size, output_dim] 目标输出
            beta: KL散度权重
            flow_weight: Flow损失权重
            constraint_fn: 约束函数 (可选)
            constraint_weight: 约束损失权重
            
        Returns:
            loss: 总损失
            loss_dict: 损失字典
        """
        batch_size = x.shape[0]
        device = x.device
        
        # === VAE部分 ===
        mean, logvar = self.encode(x)
        z_1 = self.reparameterize(mean, logvar)
        y_recon = self.decode(x, z_1)
        
        recon_loss = self.criterion(y_recon, y_target)
        kl_div = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp(), dim=1).mean()
        
        # === Flow部分 ===
        z_0 = torch.randn(batch_size, self.latent_dim, device=device)
        t = torch.rand(batch_size, 1, device=device)
        z_t, v_target = self.flow_forward(z_1.detach(), z_0, t)  # detach避免梯度冲突
        v_pred = self.predict_velocity(x, z_t, t)
        flow_loss = self.criterion(v_pred, v_target)
        
        # === 约束损失 (可选) ===
        constraint_loss = torch.tensor(0.0, device=device)
        if constraint_fn is not None and constraint_weight > 0:
            # 通过flow采样得到z_final
            z_final = self.flow_backward(x, z_0=z_0.detach(), num_steps=20)
            y_flow = self.decode(x, z_final)
            
            # 计算约束违反
            half_dim = self.output_dim // 2
            Vm = y_flow[:, :half_dim]
            Va = y_flow[:, half_dim:]
            constraint_result = constraint_fn(Vm, Va, x, reduction='mean')
            # 处理返回值可能是元组的情况 (loss, details)
            if isinstance(constraint_result, tuple):
                constraint_loss = constraint_result[0]
            else:
                constraint_loss = constraint_result
        
        # 总损失
        loss = recon_loss + beta * kl_div + flow_weight * flow_loss + constraint_weight * constraint_loss
        
        loss_dict = {
            'recon_loss': recon_loss.item(),
            'kl_div': kl_div.item(),
            'flow_loss': flow_loss.item(),
            'constraint_loss': constraint_loss.item() if isinstance(constraint_loss, torch.Tensor) else constraint_loss,
            'total_loss': loss.item()
        }
        
        return loss, loss_dict, y_recon
    
    def sample(self, x, num_samples=1, num_steps=50):
        """
        采样生成
        
        Args:
            x: [batch_size, input_dim] 条件输入
            num_samples: 每个条件生成的样本数
            num_steps: Flow采样步数
            
        Returns:
            y_samples: [batch_size * num_samples, output_dim] 生成的样本
        """
        batch_size = x.shape[0]
        device = x.device
        
        # 重复条件
        x_repeated = x.repeat_interleave(num_samples, dim=0)
        
        # 通过flow采样
        z_final = self.flow_backward(x_repeated, num_steps=num_steps)
        
        # 解码
        y_samples = self.decode(x_repeated, z_final)
        
        return y_samples


class GAN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(GAN, self).__init__()
        self.latent_dim = latent_dim
        if network == 'mlp':
            self.Generator = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
            self.Discriminator = MLP(input_dim, 1, hidden_dim, num_layers, 'sigmoid', output_dim)
        elif network == 'att':
            self.Generator = ATT(input_dim, 1, hidden_dim, num_layers, output_act, latent_dim=1, pred_type=pred_type)
            self.Discriminator = ATT(input_dim, 1, hidden_dim, num_layers, 'sigmoid', latent_dim=1, agg=True)
        else:
            NotImplementedError
        self.criterion = nn.BCELoss()

    def forward(self, x, z):
        y_pred = self.Generator(x, z)
        return y_pred

    def loss_g(self, x, y_pred):
        valid = torch.ones([x.shape[0], 1]).to(x.device)
        return self.criterion(self.Discriminator(x, y_pred), valid)

    def loss_d(self, x, y_target, y_pred):
        valid = torch.ones([x.shape[0], 1]).to(x.device)
        fake = torch.zeros([x.shape[0], 1]).to(x.device)
        d_loss = (self.criterion(self.Discriminator(x, y_target), valid) +
                  self.criterion(self.Discriminator(x, y_pred.detach()), fake)) / 2
        return d_loss


class WGAN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        if network == 'mlp':
            self.Generator = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
            self.Discriminator = Lip_MLP(input_dim, 1, hidden_dim, 1, None, output_dim)
        elif network == 'att':
            self.Generator = ATT(input_dim, 1, hidden_dim, num_layers, output_act, latent_dim=1, pred_type=pred_type)
            self.Discriminator = ATT(input_dim, 1, hidden_dim, 1, None, latent_dim=1, agg=True,
                                     pred_type=pred_type)
        self.lambda_gp = 0.1

    def forward(self, x, z=None):
        if z is None:
            z = torch.randn(size=[x.shape[0], self.latent_dim]).to(x.device)
        else:
            z = z.to(x.device)
        y_pred = self.Generator(x, z)
        return y_pred

    def loss_g(self, x, y_pred):
        return -torch.mean(self.Discriminator(x, y_pred))

    def loss_d(self, x, y_target, y_pred):
        w_dis_dual = torch.mean(self.Discriminator(x, y_pred.detach())) \
                     - torch.mean(self.Discriminator(x, y_target))
        return w_dis_dual 

    def loss_d_gp(self, x, y_target, y_pred):
        return self.loss_d(x, y_target, y_pred) + self.lambda_gp * self.gradient_penalty(x, y_target, y_pred)

    def gradient_penalty(self, x, y_target, y_pred):
        batch_size = x.shape[0]
        epsilon = torch.rand(batch_size, 1).expand_as(y_target).to(x.device)
        interpolated = epsilon * y_target + (1 - epsilon) * y_pred
        interpolated.requires_grad_(True)
        d_interpolated = self.Discriminator(x, interpolated)
        grad_outputs = torch.ones(d_interpolated.shape).to(x.device)
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                        grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        return gp


class DWGAN(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, latent_dim, output_act, pred_type):
        super(WGAN, self).__init__()
        self.latent_dim = latent_dim
        if network == 'mlp':
            self.Generator = MLP(input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim)
            self.Discriminator = Lip_MLP(input_dim, 1, hidden_dim, 1, 'abs', output_dim)
            # self.Discriminator = MLP(input_dim, 1,  hidden_dim, num_layers, output_act, output_dim)
        elif network == 'att':
            self.Generator = ATT(input_dim, 1, hidden_dim, num_layers, output_act, latent_dim=1, pred_type=pred_type)
            self.Discriminator = ATT(input_dim, 1, hidden_dim, 1, None, latent_dim=1, agg=True,
                                     pred_type=pred_type)
        self.lambda_gp = 0.1

    def forward(self, x):
        z = torch.randn(size=[x.shape[0], self.latent_dim]).to(x.device)
        v_pred = self.Generator(x, z)
        return z + v_pred

    def loss_g(self, x, y_pred):
        return torch.mean(self.Discriminator(x, y_pred))

    def loss_d(self, x, y_target, y_pred):
        w_dis_dual = torch.mean(self.Discriminator(x, y_pred.detach())) \
                     - torch.mean(self.Discriminator(x, y_target))
        return -w_dis_dual + 50 * (torch.mean(self.Discriminator(x, y_target))) ** 2

    def loss_d_gp(self, x, y_target, y_pred):
        return self.loss_d(x, y_target, y_pred) + self.lambda_gp * self.gradient_penalty(x, y_target, y_pred)

    def gradient_penalty(self, x, y_target, y_pred):
        batch_size = x.shape[0]
        epsilon = torch.rand(batch_size, 1).expand_as(y_target).to(x.device)
        interpolated = epsilon * y_target + (1 - epsilon) * y_pred
        interpolated.requires_grad_(True)
        d_interpolated = self.Discriminator(x, interpolated)
        grad_outputs = torch.ones(d_interpolated.shape).to(x.device)
        gradients = torch.autograd.grad(outputs=d_interpolated, inputs=interpolated,
                                        grad_outputs=grad_outputs,
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        gp = ((grad_norm - 1) ** 2).mean()
        return gp



"""
Generative Diffusion model Class
"""
class DM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(DM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.con_dim = input_dim
        self.time_step = time_step
        self.output_dim = output_dim
        beta_max = 0.02
        beta_min = 1e-4
        self.betas = sigmoid_beta_schedule(self.time_step, beta_min, beta_max)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.criterion = nn.L1Loss()
        self.normalize = output_norm

    def predict_noise(self, x, y, t, noise):
        y_t = self.diffusion_forward(y, t, noise)
        noise_pred = self.model(x, y_t, t)
        return noise_pred

    def diffusion_forward(self, y, t, noise):
        if self.normalize:
            y = y * 2 - 1
        t_index = (t * self.time_step).to(dtype=torch.long).cpu()
        alphas_1 = self.sqrt_alphas_cumprod[t_index].to(y.device)
        alphas_2 = self.sqrt_one_minus_alphas_cumprod[t_index].to(y.device)
        return (alphas_1 * y + alphas_2 * noise)

    def diffusion_backward(self, x, z, inf_step=100, eta=0.5):
        if inf_step==self.time_step:
            """DDPM"""
            for t in reversed(range(0, self.time_step)):
                noise = torch.randn_like(z).to(x.device)
                t_tensor = torch.ones(size=[x.shape[0], 1]).to(x.device) * t / self.time_step
                pred_noise = self.model(x, z, t_tensor)
                z = self.sqrt_recip_alphas[t].to(x.device)*(z-self.betas[t].to(x.device) / self.sqrt_one_minus_alphas_cumprod[t].to(x.device) * pred_noise) \
                    + torch.sqrt(self.posterior_variance[t].to(x.device)) * noise
        else: 
            """DDIM"""
            sample_time_step = torch.linspace(self.time_step-1, 0, (inf_step + 1)).to(x.device).to(torch.long)
            for i in range(1, inf_step + 1):
                t = sample_time_step[i - 1] 
                prev_t = sample_time_step[i] 
                noise = torch.randn_like(z).to(x.device)
                t_tensor = torch.ones(size=[x.shape[0], 1]).to(x.device) * t / self.time_step
                pred_noise = self.model(x, z, t_tensor)
                t_cpu = t.cpu()
                prev_t_cpu = prev_t.cpu()
                y_0 = (z - self.sqrt_one_minus_alphas_cumprod[t_cpu].to(x.device) * pred_noise) / self.sqrt_alphas_cumprod[t_cpu].to(x.device)
                var = eta * self.posterior_variance[t_cpu].to(x.device)
                z = self.sqrt_alphas_cumprod[prev_t_cpu].to(x.device) * y_0 + torch.sqrt(torch.clamp(1 - self.alphas_cumprod[prev_t_cpu].to(x.device) - var, 0, 1)) * pred_noise + torch.sqrt(var) * noise
        if self.normalize:
            return (z + 1) / 2
        else:
            return z

    def loss(self, noise_pred, noise):
        return self.criterion(noise_pred, noise)

class FM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(FM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        elif network == 'carbon_tax_aware_mlp':
            # 使用我们的条件MLP网络，专门处理[负荷, 碳税, 锚点]的条件输入
            self.model = CarbonTaxAwareMLP(input_dim, output_dim, hidden_dim, num_layers, None, 
                                          latent_dim=output_dim, carbon_tax_dim=1, anchor_dim=output_dim) 
        elif network == 'sdp_lip':
            self.model = SDP_MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)    # include SDPBasedLipschitzLinearLayer in mlp
        else:
            NotImplementedError
        self.output_dim = output_dim
        self.con_dim = input_dim
        self.time_step = time_step
        self.min_sd = 0.01
        self.criterion = nn.L1Loss()
        self.normalize = output_norm

    def flow_forward(self, y, t, z, vec_type='gaussian'):
        if self.normalize:
            y = 2 * y - 1  # [0,1] normalize to [-1,1]
        if vec_type == 'gaussian':
            """
            t = 0:  N(0, 1)
            t = 1:  N(y, sd)
            Linear interpolation for mu and sigma
            """
            mu = y * t
            sigma = (self.min_sd) * t + 1 * (1 - t)
            noise = torch.randn_like(y).to(y.device)
            yt = mu + noise * sigma
            vec = (y - (1 - self.min_sd) * yt) / (1 - (1 - self.min_sd) * t)
        elif vec_type == 'conditional':
            mu = t * y + (1 - t) * z
            sigma = ((self.min_sd * t) ** 2 + 2 * self.min_sd * t * (1 - t)) ** 0.5
            noise = torch.randn_like(y).to(y.device)
            yt = mu + noise * sigma
            vec = (y - (1 - self.min_sd) * yt) / (1 - (1 - self.min_sd) * t)
        elif vec_type == 'rectified':
            """
            t = 0:  N(0, 1)
            t = 1:  N(y, sd)
            Linear interpolation for z and y
            """
            yt = t * y + (1 - t) * z
            vec = y-z
        elif vec_type == 'interpolation':
            """
            t = 0:  x
            t = 1:  N(0,1)
            Linear interpolation for z and y
            """
            # yt = (1 - t) * y + t * z
            yt = t * y + (1 - t) * z
            vec = None
            # return torch.cos(torch.pi/2*t) * y + torch.sin(torch.pi/2*t) * z
            # return (torch.cos(torch.pi*t) + 1)/2 * y + (torch.cos(-torch.pi*t) +1)/2  * z
        return yt, vec

    def flow_backward(self, x, z, step=0.01, method='Euler', direction='forward', 
                     objective_fn=None, guidance_config=None, 
                     evolutionary_config=None, projection_config=None):
        """
        带梯度引导、演化算法增强和约束切空间投影的流模型反向采样
        
        Args:
            x: 条件输入
            z: 初始状态
            step: 步长
            method: ODE求解方法
            direction: 流动方向 ('forward' 或 'backward')
            objective_fn: 目标函数，用于计算引导梯度和投影约束
            guidance_config: 引导配置字典
            evolutionary_config: 演化算法配置字典（可选）
            projection_config: 约束切空间投影配置字典（可选），包含:
                - 'enabled': 是否启用投影
                - 'start_time': 开始投影的时间阈值
                - 'env': 电网环境对象
                - 'single_target': 是否为单目标模式
        
        Returns:
            最终采样结果，以及约束违反值（如果提供了objective_fn）
        """
        step = step if direction == 'forward' else -step
        t = 0 if direction == 'forward' else 1
        while (direction == 'forward' and t < 1) or (direction == 'backward' and t > 0): 
            z += ode_step(self.model, x, z, t, step, method, 
                        objective_fn=objective_fn, guidance_config=guidance_config, 
                        evolutionary_config=evolutionary_config, projection_config=projection_config)
            t += step
        
        # # 计算最终状态的约束违反值
        if objective_fn is not None:
            output_dim = z.shape[1]
            final_z = (z + 1) / 2 if self.normalize else z
            
            # 调用objective_fn计算平均损失（用于guidance）
            # 修复: 处理 guidance_config 为 None 的情况
            single_target = guidance_config.get('single_target', False) if guidance_config is not None else True
            x_real = x[:, :-1] if not single_target else x
            loss_value = objective_fn(
                final_z[:, :output_dim//2], 
                final_z[:, output_dim//2:], 
                x_real,
                'none'
            )

            constraint_violation = loss_value
                
            if self.normalize:
                return (z + 1) / 2, constraint_violation
            else:
                return z, constraint_violation 
        else: 
            if self.normalize:
                return (z + 1) / 2, None
            else:
                return z, None

    def predict_vec(self, x, yt, t):
        vec_pred = self.model(x, yt, t)
        # x_0 = self.model(x, yt, t)
        # vec_pred = (x_0 - yt)/(1-t+1e-5)
        return vec_pred

    def loss(self, y, z, vec_pred, vec, vec_type='gaussian'):
        if vec_type in ['gaussian', 'rectified', 'conditional']:
            return self.criterion(vec_pred, vec)
        elif vec_type in ['interpolation']:
            loss = 1 / 2 * torch.sum(vec_pred ** 2, dim=1, keepdim=True) \
                   - torch.sum((y - z) * vec_pred, dim=1, keepdim=True)
            return loss.mean()
            # loss = 1/2 * torch.sum(vec **2, dim=-1, keepdim=True) - torch.sum((-torch.pi/2*torch.sin(torch.pi/2*t) * y +  torch.pi/2*torch.cos(torch.pi/2*t) * z) * vec, dim=-1, keepdim=True)
            # loss = 1/2 * torch.sum(vec **2, dim=-1, keepdim=True) - torch.sum((-torch.pi*torch.sin(torch.pi*t)*y +    torch.pi*torch.sin(-torch.pi*t) * z) * vec, dim=-1, keepdim=True)
        else:
            NotImplementedError

class AM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(AM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, 1, hidden_dim, num_layers, None, output_dim, 'silu')
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.output_dim = output_dim
        self.con_dim = input_dim
        self.time_step = time_step
        self.min_sd = 0.01
        self.criterion = nn.MSELoss()
        self.normalize = output_norm
    
    def loss(self, x, y, z, t):
        s0 = self.model(x, y, torch.zeros_like(t))
        s1 = self.model(x, y, torch.ones_like(t))
        yt = t*y + (1-t)*z 
        yt.requires_grad = True
        st = self.model(x, yt, t)
        vec = torch.autograd.grad(st, yt, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        loss = self.criterion(vec, y-z)
        # loss =  1 / 2 * torch.sum(vec ** 2, dim=1, keepdim=True) \
        #            - torch.sum((y - z) * vec, dim=1, keepdim=True)
        # t.requires_grad = True
        # st = self.model(x, yt, t)
        # dsdt = torch.autograd.grad(st, t, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        # dsdy = torch.autograd.grad(st, yt, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        # loss = s0 - s1 + 0.5 * torch.sum(dsdy**2, dim=1, keepdim=True) + dsdt
        return loss.mean()

    def flow_backward(self, x, step=0.01, method='Euler', direction='forward'):
        z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device)
        step = step if direction == 'forward' else -step
        t = 0 if direction == 'forward' else 1
        while (direction == 'forward' and t < 1) or (direction == 'backward' and t > 0):
            z.requires_grad = True
            st = self.model(x, z, torch.ones(size=[z.shape[0],1]).to(x.device)*t)
            dsdz = torch.autograd.grad(st, z, torch.ones_like(st), create_graph=True, retain_graph=True, only_inputs=True,)[0]
            z.requires_grad = False
            z += dsdz.detach() * step
            t += step
        if self.normalize:
            return (z + 1) / 2
        else:
            return z

class CM(nn.Module):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super(CM, self).__init__()
        if network == 'mlp':
            self.model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
            # self.target_model = MLP(input_dim, output_dim, hidden_dim, num_layers, None, output_dim)
        elif network == 'att':
            self.model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
            # self.target_model = ATT(input_dim, 1, hidden_dim, num_layers, latent_dim=1, pred_type=pred_type)
        else:
            NotImplementedError
        self.output_dim = output_dim
        self.con_dim = input_dim
        self.time_step = time_step
        self.criterion = nn.MSELoss()
        self.normalize = output_norm
        self.std = 80
        self.eps = 0.002
    
    def flow_forward(self, y, z, t):
        yt = y + z * t * self.std
        # yt = (1-t) * y + z * t 
        return yt

    def predict(self, x, yt, t, model):
        return self.c_skip_t(t) * yt + self.c_out_t(t) * model(x, yt, t)

    def sampling(self, x, inf_step):
        z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) * self.std
        t = torch.ones(size=[x.shape[0],1]).to(x.device)
        y0 =  self.predict(x, z, t, self.target_model)
        return y0

    def loss(self, x, y, z, t1, t2, data, vec):
        # y = data.scaling_v(y)
        yt1 = self.flow_forward(y, z, t1)
        y01 = self.predict(x, yt1, t1, self.model)
        # y01_scale = data.scaling_v(y01)
        # x_scale = data.scaling_load(x)
        # pg, qg, vm, va = data.power_flow_v(x_scale, y01_scale)
        # pel = torch.abs(data.eq_resid(x_scale, torch.cat([pg,qg, vm,va],dim=1)))
        with torch.no_grad():
            yt2 = self.flow_forward(y, z, t2)
            y02 = self.predict(x, yt2, t2, self.target_model)
            # y02_scale = data.scaling_v(y02)
        return (self.criterion(y01, y02)).mean() #+ 0.0001*pel.mean()

    def c_skip_t(self, t):
        t = t * self.std
        return 0.25 / (t.pow(2) + 0.25)
    
    def c_out_t(self, t):
        t = t * self.std
        return 0.25 * t / ((t + self.eps).pow(2)).pow(0.5)

    def kerras_boundaries(self, sigma, eps, N, T):
        # This will be used to generate the boundaries for the time discretization
        return torch.tensor(
            [
                (eps ** (1 / sigma) + i / (N - 1) * (T ** (1 / sigma) - eps ** (1 / sigma)))
                ** sigma
                for i in range(N)
            ]
        )

class CD(CM):
    def __init__(self, network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type):
        super().__init__(network, input_dim, output_dim, hidden_dim, num_layers, time_step, output_norm, pred_type)
        self.criterion = nn.L1Loss()

    def loss(self, x, y, z, t1, step, forward_step, data, vec_model):
        # print(self.c_out_t(torch.zeros(1)), self.c_skip_t(torch.zeros(1)))
        yt1 = (1-t1) * z + t1 * y
        y01 = self.predict(x, yt1, t1, self.model)
        # y01_scale = data.scaling_v(y01)
        # x_scale = data.scaling_load(x)
        # pg, qg, vm, va = data.power_flow_v(x_scale, y01_scale)
        # pel = torch.abs(data.eq_resid(x_scale, torch.cat([pg,qg, vm,va],dim=1)))
        with torch.no_grad():
            v_pred_1 = vec_model.predict_vec(x, yt1, t1) * step
            yt2 = yt1 + v_pred_1
            t2 = t1 + step
            for _ in range(forward_step):
                v_pred_2 = vec_model.predict_vec(x, yt2, t2) * step
                yt2 = yt2 + v_pred_2
                t2 = t2 + step
            y02 = self.predict(x, yt2, t2, self.model)

            # v_pred_1 = vec_model.predict_vec(x, yt2, t2) * step
            # yt2 = yt1 + (v_pred_0 + v_pred_1)/2 
            # v_pred_0 = vec_model.predict_vec(x, yt1, t1) * step
            # v_pred_1 = vec_model.predict_vec(x, yt1 + v_pred_0 * 0.5, t1 + step * 0.5) * step
            # v_pred_2 = vec_model.predict_vec(x, yt1 + v_pred_1 * 0.5, t1 + step * 0.5) * step
            # v_pred_3 = vec_model.predict_vec(x, yt1 + v_pred_2, t1 + step) * step
            # yt2 = yt1 + (v_pred_0 + 2 * v_pred_1 + 2 * v_pred_2 + v_pred_3) / 6
        return (self.criterion(y01, y02)).mean() #+ 0.001*pel.mean()
    
    def continnuous_loss(self, x, y, z, t, stpe, forward_step, data, vec_model):
        yt = (1-t) * z + t * y
        with torch.no_grad():
            vec = vec_model.predict_vec(x, yt, t)
        yt.requires_grad_(True)
        t.requires_grad_(True)
        y0 = self.predict(x, yt, t, self.model)
        dy = torch.autograd.grad(y0, yt, torch.ones_like(y0), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        dt = torch.autograd.grad(y0, t, torch.ones_like(y0), create_graph=True, retain_graph=True, only_inputs=True,)[0]
        return torch.square(dy * vec + dt).mean()



    def predict(self, x, yt, t, model):
        return yt + (1-t) * model(x, yt, t)

    def sampling(self, x, inf_step):
        # z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) 
        # t = torch.zeros(size=[x.shape[0],1]).to(x.device)
        # y0 =  self.predict(x, z, t, self.target_model)
        y0 = 0
        for dt in torch.linspace(0,1,1):
            z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) 
            t = torch.ones(size=[x.shape[0],1]).to(x.device) * dt
            yt = (1-t) * z + t * y0
            y0 =  self.predict(x, yt, t, self.model)
        # z = torch.randn(size=[x.shape[0], self.output_dim]).to(x.device) 
        # yt1 = (y0+z)/2
        # y0 =  self.predict(x, yt1, t+0.5, self.target_model)
        return y0

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def quadratic_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start

def ode_step(model: torch.nn.Module, x: torch.Tensor, z: torch.Tensor, t: float, step: float, 
              method: str = 'Euler', objective_fn=None, guidance_config=None,
              evolutionary_config=None, projection_config=None):
    """
    改进的ODE步进函数，支持基于论文的梯度引导、演化算法增强和约束切空间投影
    
    Args:
        model: 流模型
        x: 条件输入 (batch_size, input_dim) - [负荷特征, 碳税, 锚点]
        z: 当前状态 (batch_size, output_dim) - 对应论文中的 x_t
        t: 当前时间步
        step: 步长
        method: ODE求解方法
        objective_fn: 目标函数 l(x_0; G)，接受 (Vm, Va, 负荷特征, reduction)，返回约束损失
        guidance_config: 引导配置字典，包含:
            - 'enabled': 是否启用引导
            - 'scale': 引导强度
            - 'perp_scale': 垂直方向引导强度
        evolutionary_config: 演化算法配置字典（参见adaptive_evolutionary_guidance）
        projection_config: 约束切空间投影配置字典，包含:
            - 'enabled': 是否启用投影（默认False）
            - 'start_time': 开始投影的时间阈值（默认0.5）
            - 'env': 电网环境对象（必需）
            - 'single_target': 是否为单目标模式（默认True）
            - 'verbose': 是否打印调试信息
    
    Returns:
        v_pred: 预测的速度 * 步长
    """
    model.eval()
    t_tensor = torch.ones(size=[x.shape[0], 1]).to(x.device) * t

    def model_eval(z_eval, t_eval):
        return model(x, z_eval, t_eval)

    # 统一转换为首字母大写，避免大小写不匹配问题
    method = method.capitalize()

    # 步骤1: 计算基础ODE步进（无引导）
    with torch.no_grad():
        if method == 'Euler':
            v_pred = model_eval(z, t_tensor) * step
        else:
            v_pred_0 = model_eval(z, t_tensor) * step
            if method == 'Heun':
                v_pred_1 = model_eval(z + v_pred_0, t_tensor + step) * step
                v_pred = (v_pred_0 + v_pred_1) / 2
            elif method == 'Mid':
                v_pred = model_eval(z + v_pred_0 * 0.5, t_tensor + step * 0.5) * step
            elif method == 'Rk4':
                v_pred_1 = model_eval(z + v_pred_0 * 0.5, t_tensor + step * 0.5) * step
                v_pred_2 = model_eval(z + v_pred_1 * 0.5, t_tensor + step * 0.5) * step
                v_pred_3 = model_eval(z + v_pred_2, t_tensor + step) * step
                v_pred = (v_pred_0 + 2 * v_pred_1 + 2 * v_pred_2 + v_pred_3) / 6
            else:
                # 如果method不在支持列表中，回退到Euler方法
                v_pred = v_pred_0
                
    # ============================================================================
    # 步骤1: 应用梯度引导（如果启用）- 使用预计算的约束
    # ============================================================================
    if (guidance_config is not None) and guidance_config.get('enabled', False):
        guidance_scale = guidance_config.get('scale', 0.1)
        perp_scale = guidance_config.get('perp_scale', 0.001)
        should_guidance = (t + step >= guidance_config.get('start_time', 0.8))
        if should_guidance and objective_fn is not None:
            # 使用requires_grad计算梯度
            z_for_grad = z.detach().requires_grad_(True)
            output_dim = z_for_grad.shape[1]
            
            # 计算梯度（需要带梯度） 
            x_real = x[:, :-1] if not guidance_config.get('single_target', False) else x
            constraint_violations = objective_fn(z_for_grad[:, :output_dim//2], z_for_grad[:, output_dim//2:], x_real, 'none')
            
            # 为每个样本计算个性化梯度 
            grad_z = torch.autograd.grad(
                constraint_violations,
                z_for_grad,
                grad_outputs=torch.ones_like(constraint_violations),
                create_graph=False,
                retain_graph=False,
                only_inputs=True
            )[0]  
            
            # guidance 拆成两部分：沿基础流方向和垂直基础流方向的分量
            v_base = v_pred  # 基础流（不含引导）
            base_norm = v_base.norm(dim=1, keepdim=True)
            safe_mask = (base_norm > 1e-8).float()
            g = (-grad_z) * step * guidance_scale  # 原始引导
            v_base_norm = v_base / (base_norm + 1e-8)
            g_parallel = (g * v_base_norm).sum(dim=1, keepdim=True) * v_base_norm * safe_mask
            g_perp = (g - g_parallel) * safe_mask
            v_guidance = g_parallel + perp_scale * g_perp 
    else:
        constraint_violations = None
        v_guidance = torch.zeros_like(v_pred)
    # ============================================================================
    # 步骤1.5: 应用演化算法引导（如果启用）- 使用预计算的约束
    # ============================================================================
    if evolutionary_config is not None and evolutionary_config.get('enabled', False):
        if evolutionary_config.get('start_time', 0.8) <= t + step:
            if constraint_violations is None: 
                x_real = x[:, :-1] if not evolutionary_config.get('single_target', False) else x
                constraint_violations = objective_fn(z[:, :z.shape[1]//2], z[:, z.shape[1]//2:], x_real, 'none')
            v_pred = adaptive_evolutionary_guidance(
                z=z,
                v_pred=v_pred,
                v_guidance=v_guidance,
                t=t,
                step=step,
                x_input=x_real,
                objective_fn=objective_fn,
                evo_config=evolutionary_config,
                constraint_violations=constraint_violations  # 传递预计算的约束
            )
    else:
        v_pred = v_pred + v_guidance
    
    # ============================================================================
    # 步骤2: 应用 Drift-Correction 流形稳定化（如果启用）
    # 核心公式: v_final = P_tan @ v_pred + correction
    #   - P_tan @ v_pred: 切向投影（保持流动方向）
    #   - correction = -λ * F^+ @ f(x): 法向修正（拉回可行域）
    # ============================================================================
    if projection_config is not None and projection_config.get('enabled', False):
        start_time = projection_config.get('start_time', 0.5)
        should_project = (t + step >= start_time)
        
        if should_project:
            env = projection_config.get('env', None)
            if env is not None:
                from post_processing import compute_drift_correction_batch, apply_drift_correction
                
                single_target = projection_config.get('single_target', True)
                x_real = x if single_target else x[:, :-1]
                lambda_cor = projection_config.get('lambda_cor', 5.0)
                
                # 计算 Drift-Correction: 切向投影矩阵 + 法向修正向量
                P_tan, correction = compute_drift_correction_batch(z, x_real, env, lambda_cor)
                
                # 应用 Drift-Correction
                v_pred_before = v_pred.clone()
                v_pred = apply_drift_correction(v_pred, P_tan, correction)
                
                if projection_config.get('verbose', False):
                    v_norm_before = v_pred_before.norm(dim=1).mean().item()
                    v_norm_after = v_pred.norm(dim=1).mean().item()
                    correction_norm = correction.norm(dim=1).mean().item()
                    print(f"  [Drift-Correction] t={t:.2f}, v_norm: {v_norm_before:.4f} -> {v_norm_after:.4f}, correction_norm={correction_norm:.4f}")
    
    return v_pred


def feasibility_projection(z_pred, x_input, objective_fn, num_iters=10, lr=0.01, lambda_cons=10.0, verbose=False):
    """
    可微分的可行性投影 - 将预测状态投影到可行域
    
    优化目标：
        min_{z} ||z - z_pred||² + λ * constraint_violation(z)
    
    方法：梯度下降迭代求解
    
    Args:
        z_pred: 预测状态 (batch, output_dim) - 归一化的 [Vm, Va]
        x_input: 条件输入 (batch, input_dim) - [负荷特征, 碳税, 锚点]
        objective_fn: 约束计算函数，接受 (Vm, Va, 负荷特征, reduction) 返回约束违反值
        num_iters: 投影迭代次数
        lr: 学习率
        lambda_cons: 约束违反的权重
        verbose: 是否打印调试信息
    
    Returns:
        z_proj: 投影后的状态 (batch, output_dim)
    """
    if objective_fn is None:
        # 如果没有提供约束函数，直接返回原始预测
        return z_pred
    
    batch_size = z_pred.shape[0]
    output_dim = z_pred.shape[1]
    half_dim = output_dim // 2
    
    z_proj = z_pred.clone().detach()
    
    for iter_idx in range(num_iters):
        z_proj.requires_grad_(True)
        
        # 分割为Vm和Va
        vm = z_proj[:, :half_dim]
        va = z_proj[:, half_dim:]
        
        # 计算约束违反（每个样本单独）
        # x_input 的结构: [负荷特征, 碳税(1维), 锚点(output_dim维)]，objective_fn只需要负荷特征
        constraint_loss = objective_fn(
            vm, va, x_input[:, :-(1 + output_dim)],  # 去掉碳税和锚点维度
            reduction='none'  # 返回 (batch,) 向量
        )
        
        # 计算接近度损失
        proximity_loss = torch.sum((z_proj - z_pred) ** 2, dim=1)  # (batch,)
        
        # 总损失
        total_loss = proximity_loss + lambda_cons * constraint_loss
        
        # 计算梯度
        grad = torch.autograd.grad(
            outputs=total_loss.sum(),
            inputs=z_proj,
            create_graph=False,  # 推理时不需要二阶导
        )[0]
        
        # 梯度下降更新
        with torch.no_grad():
            z_proj = z_proj - lr * grad
            
            # 可选：裁剪到合理范围（防止发散）
            z_proj = torch.clamp(z_proj, -1.0, 1.0)
        
        if verbose and iter_idx % 5 == 0:
            print(f"  Projection iter {iter_idx}: constraint_loss={constraint_loss.mean().item():.6f}, "
                  f"proximity_loss={proximity_loss.mean().item():.6f}")
    
    return z_proj.detach()


# ============================================================================
# 演化算法增强模块 (Evolutionary Algorithm Enhancement)
# ============================================================================


def differential_evolution_guidance(z, v_pred, v_guidance, x_input, objective_fn, 
                                   F=0.5, CR=0.7, strategy='best/1', 
                                   clip_norm=None, blend_temp=1.0,
                                   curr_viol=None, bounds=None):
    """
    差分进化 (Differential Evolution) 引导 - 完全向量化实现
    
    在batch维度上将多个z样本视为一个种群，通过DE的变异和交叉操作
    生成更好的演化方向，替代或修正原始的v_pred。
    
    Args:
        z: 当前状态 (batch_size, dim)
        v_pred: flow matching预测的速度 (batch_size, dim)
        x_input: 条件输入 (batch_size, input_dim)
        objective_fn: 约束计算函数
        F: 缩放因子，控制变异强度 [0.3-1.0]
        CR: 交叉概率 [0.5-0.9]
        strategy: DE变异策略 ('best/1', 'rand/1', 'current-to-best/1')
        constraint_violations: 预计算的约束违反值 (可选，用于避免重复计算)
    
    Returns:
        v_pred_enhanced: DE增强后的速度向量 (batch_size, dim)
    """
    if objective_fn is None:
        return v_pred

    B, D = z.shape
    device = z.device
    half_dim = D // 2 
    
    # 评估当前种群的约束违反（如果未提供）
    if curr_viol is None:  
        with torch.no_grad():
            curr_viol = objective_fn(
                z[:, :half_dim], z[:, half_dim:], 
                x_input,
                reduction='none'
            )  # (batch_size,)
    
    # 找到最优个体
    best_idx = torch.argmin(curr_viol)
    z_best = z[best_idx:best_idx+1]  # (1, dim)
    
    # ---------- 生成 r0, r1, r2 索引（避免 i、自身且彼此不同） ----------
    # 用循环移位的打乱表法，避免 O(B^2 logB) 的 argsort
    base = torch.arange(B, device=device)
    perm = torch.stack([torch.randperm(B, device=device) for _ in range(3)], dim=1)  # (B,3)
    # 确保不包含自身：如果撞到了 i，则循环移位修正
    for k in range(3):
        hit = perm[:, k].eq(base)
        if hit.any():
            perm[hit, k] = (perm[hit, k] + 1) % B
    # 确保三者彼此不同
    # 若出现重复，做一次简单修复（概率极低，多次修复可加 while）
    same01 = perm[:,0].eq(perm[:,1])
    perm[same01,1] = (perm[same01,1] + 1) % B
    same02 = perm[:,0].eq(perm[:,2])
    perm[same02,2] = (perm[same02,2] + 1) % B
    same12 = perm[:,1].eq(perm[:,2])
    perm[same12,2] = (perm[same12,2] + 1) % B

    r0, r1, r2 = perm[:,0], perm[:,1], perm[:,2]
    zr0, zr1, zr2 = z[r0], z[r1], z[r2]

    # ---------- 变异 ----------
    if strategy == 'rand/1':
        z_mut = zr0 + F * (zr1 - zr2)
    elif strategy == 'current-to-best/1':
        z_mut = z + F * (z_best - z) + F * (zr1 - zr2) + F * v_guidance * 0.1    # 变异的时候同时考虑随机性和梯度信息
    else:  # 'best/1'
        z_mut = z_best + F * (zr1 - zr2)
    
    # ============================================================================
    # 向量化交叉操作
    # ============================================================================
    # ---------- 交叉（trial 个体） ----------
    cross_mask = (torch.rand(B, D, device=device) < CR)
    no_x = ~cross_mask.any(dim=1)
    if no_x.any():
        cross_mask[no_x, torch.randint(0, D, (no_x.sum(),), device=device)] = True
    u = torch.where(cross_mask, z_mut, z) 
    
    # ---------- 边界修复（可选） ----------
    if bounds is not None:
        if callable(bounds):
            lower, upper = bounds(u)
        else:
            lower, upper = bounds
        if lower is not None:
            u = torch.maximum(u, torch.as_tensor(lower, device=device, dtype=z.dtype))
        if upper is not None:
            u = torch.minimum(u, torch.as_tensor(upper, device=device, dtype=z.dtype))

     # ---------- trial 评价 + 择优 ----------
    with torch.no_grad(): 
        u_viol = objective_fn(u[:, :half_dim], u[:, half_dim:], x_input, reduction='none').to(z.dtype)
        u_viol = torch.nan_to_num(u_viol, nan=1e6, posinf=1e6, neginf=1e6)

        improve = u_viol < curr_viol  # 可行性/惩罚更小视为改进
    
    # ---------- 形成 DE 建议速度（仅在改进时生效） ----------
    v_de = u - z
    # 归一化并按 v_pred 尺度缩放
    v_pred_norm = torch.linalg.norm(v_pred, dim=1, keepdim=True)  # (B,1)
    v_de_norm   = torch.linalg.norm(v_de,   dim=1, keepdim=True).clamp_min(1e-8)
    v_de_scaled = v_de / v_de_norm * (v_pred_norm + 1e-8)

    # 范数裁剪（可选）
    if clip_norm is not None:
        v_de_scaled = v_de_scaled * (clip_norm / torch.maximum(
            clip_norm * torch.ones_like(v_de_norm), v_de_norm))

        # ---------- 自适应融合权 g（违反越大，g 越大） ----------
    with torch.no_grad():
        # 软归一化：越大越接近 1
        g = (curr_viol / (curr_viol + 1.0)).unsqueeze(1)  # (B,1)
        if blend_temp is not None and blend_temp != 1.0:
            # 温度调整（让分布更平/更尖）
            g = torch.sigmoid(torch.logit(g.clamp(1e-6, 1-1e-6)) / blend_temp)

    # 只对“有改进”的样本启用 DE 速度，否则为 0
    v_take = torch.where(improve.unsqueeze(1), v_de_scaled, torch.zeros_like(v_de_scaled))

    # ---------- 最终速度 ----------
    v_out = (1 - g) * v_pred + g * v_take 
    # v_out = v_take
    return v_out


def sep_cma_es_guidance(z, v_pred, x_input, objective_fn, 
                       constraint_violations=None, cma_state=None,
                       blend_ratio=0.7, update_state=True):
    """
    对角CMA-ES (Separable CMA-ES) 引导 - 带记忆功能
    
    使用对角协方差矩阵的轻量级CMA-ES，避免完整协方差矩阵的O(d²)存储和O(d³)计算。
    通过CMAESState保存演化路径和协方差信息，实现跨时间步的记忆。
    
    Args:
        z: 当前状态 (batch_size, dim)
        v_pred: flow matching预测的速度 (batch_size, dim)
        x_input: 条件输入 (batch_size, input_dim)
        objective_fn: 约束计算函数
        constraint_violations: 预计算的约束违反值 (可选)
        cma_state: CMAESState对象，保存历史信息 (可选)
        blend_ratio: CMA方向和flow方向的混合比例
        update_state: 是否更新CMA状态
    
    Returns:
        v_pred_enhanced: CMA-ES增强后的速度向量 (batch_size, dim)
    """
    if objective_fn is None:
        return v_pred
    
    batch_size = z.shape[0]
    dim = z.shape[1]
    
    if batch_size < 4:
        return v_pred
    
    # 评估当前种群
    output_dim = dim
    half_dim = output_dim // 2
    vm = z[:, :half_dim]
    va = z[:, half_dim:]
 
    if constraint_violations is None:
        with torch.no_grad():
            constraint_violations = objective_fn(
                vm, va,
                x_input[:, :-(1 + output_dim)],
                reduction='none'
            )  # (batch_size,)
    
    # 排序：找到表现最好的一半
    sorted_indices = torch.argsort(constraint_violations)
    mu = batch_size // 2  # 选择前50%
    elite_indices = sorted_indices[:mu]
    
    # 计算加权重心（更好的个体权重更高）
    weights = torch.log(torch.tensor(mu + 0.5, device=z.device)) - \
              torch.log(torch.arange(1, mu + 1, device=z.device, dtype=torch.float32))
    weights = weights / weights.sum()  # 归一化
    
    # 计算加权平均（分布中心）
    z_elite = z[elite_indices]  # (mu, dim)
    
    # 如果有CMA状态，使用状态中的均值作为old_mean
    if cma_state is not None and cma_state.mean is not None:
        old_mean = cma_state.mean
    else:
        # 第一次调用，使用当前种群平均
        old_mean = z.mean(dim=0)
    
    # 计算新均值
    mean_z = (weights.unsqueeze(1) * z_elite).sum(dim=0)  # (dim,)
    
    # 更新CMA-ES状态（如果提供了状态对象且需要更新）
    if cma_state is not None and update_state:
        cma_state.update(z_elite, weights, old_mean)
        # 使用状态中的采样标准差
        std = cma_state.get_sampling_std()
    else:
        # 无状态模式：估计对角协方差（每个维度的方差）
        centered = z_elite - mean_z.unsqueeze(0)  # (mu, dim)
        variance = (weights.unsqueeze(1) * (centered ** 2)).sum(dim=0)  # (dim,)
        variance = torch.clamp(variance, min=1e-8)  # 防止退化
        std = torch.sqrt(variance)  # (dim,)
    
    # 为整个batch生成改进方向（向量化操作）
    direction_to_mean = (mean_z.unsqueeze(0) - z)  # (batch, dim)
    
    # 使用CMA-ES学到的标准差生成自适应噪声
    adaptive_noise = torch.randn_like(z) * std.unsqueeze(0)  # (batch, dim)
    
    # CMA方向 = 向均值移动 + 自适应探索噪声
    if cma_state is not None:
        # 有状态时，使用状态中的sigma
        v_cma = direction_to_mean + adaptive_noise * cma_state.sigma
    else:
        # 无状态时，使用固定的探索强度
        v_cma = direction_to_mean * 0.2 + adaptive_noise * 0.1
    
    # 与原始flow方向混合
    v_pred_enhanced = blend_ratio * v_cma + (1.0 - blend_ratio) * v_pred
    
    return v_pred_enhanced


def adaptive_evolutionary_guidance(z, v_pred, t, step, x_input, objective_fn, v_guidance,
                                   evo_config=None, constraint_violations=None):
    """
    演化算法引导 - 默认使用DE，可选CMA-ES
    
    DE（差分进化）更稳健，适合有约束的优化问题：
    - 基于种群中的最优个体
    - 不会过度偏离可行域
    - 对约束违反的容忍度更好
    
    CMA-ES（协方差矩阵适应）可选，适合约束较松的情况：
    - 带记忆功能，学习问题结构
    - 更强的探索能力
    - 但可能在强约束下偏离
    
    Args:
        z: 当前状态 (batch_size, dim)
        v_pred: flow matching预测的速度 (batch_size, dim)
        t: 当前时间步 [0, 1]
        step: 时间步长
        x_input: 条件输入
        objective_fn: 约束计算函数
        evo_config: 演化算法配置字典
        constraint_violations: 预计算的约束违反值 (可选，用于避免重复计算)
            {
                'enabled': bool,              # 是否启用
                'method': str,                # 'DE' (default) 或 'CMA-ES'
                'start_time': float,          # 开始应用的时间步 (default 0.9)
                
                # DE参数
                'de_F': float,                # 变异强度 (default 0.5)
                'de_CR': float,               # 交叉概率 (default 0.7)
                'de_strategy': str,           # 变异策略 (default 'best/1')
                
                # CMA-ES参数（仅当method='CMA-ES'时使用）
                'blend_ratio': float,         # CMA方向混合比例 (default 0.7)
                'c_sigma': float,             # 步长学习率 (default 0.3)
                'c_c': float,                 # 协方差路径学习率 (default 0.4)
                'c_cov': float,               # 协方差矩阵学习率 (default 0.6)
                'damps': float,               # 步长阻尼 (default 1.0)
                'cma_state': CMAESState,      # 自动创建
                
                'verbose': bool               # 是否打印详细信息 (default False)
            }
    
    Returns:
        v_pred_enhanced: 演化算法增强后的速度向量
    """
    
    # 获取配置参数
    method = evo_config.get('method', 'DE')  # 默认使用DE 
    de_F = evo_config.get('de_F', 0.5)
    de_CR = evo_config.get('de_CR', 0.7)
    de_strategy = evo_config.get('de_strategy', 'best/1')
    verbose = evo_config.get('verbose', False) 
    
    batch_size = z.shape[0] 
    
    if batch_size < 4:
        if verbose:
            print(f"  [Evo] Batch size {batch_size} too small, skipping")
        return v_pred
    
    # 计算平均约束违反（用于verbose输出和CMA-ES）
    if verbose:
        avg_violation = constraint_violations.mean().item()
    
    # 根据配置选择方法
    if method.upper() == 'DE':
        # 使用差分进化（默认，更稳健）
        v_enhanced = differential_evolution_guidance(
            z, v_pred, v_guidance, x_input, objective_fn,
            F=de_F, CR=de_CR, strategy=de_strategy,
            curr_viol=constraint_violations  # 传递预计算的约束
        )
        
        if verbose:
            best_violation = constraint_violations.min().item()
            worst_violation = constraint_violations.max().item()
            print(f"  [DE t={t:.3f}] Strategy: {de_strategy}, F: {de_F:.2f}, CR: {de_CR:.2f}, "
                  f"Avg violation: {avg_violation:.6f}, "
                  f"Best: {best_violation:.6f}, Worst: {worst_violation:.6f}")
    
    else:
        # 未知方法，返回原始v_pred
        if verbose:
            print(f"  [Evo] Unknown method '{method}', using original v_pred")
        v_enhanced = v_pred
    
    return v_enhanced


# ============================================================================
# 演化算法增强模块结束
# ============================================================================


class GumbelSoftmax(nn.Module):
    def __init__(self, temperature=1.0, hard=False):
        super(GumbelSoftmax, self).__init__()
        self.temperature = temperature
        self.hard = hard

    def forward(self, x):
        gumbel_noise = self.sample_gumbel(x.size())
        y = x + gumbel_noise.to(x.device)
        soft_sample = F.softmax(y / self.temperature, dim=-1)

        if self.hard:
            hard_sample = torch.zeros_like(soft_sample).scatter(-1, soft_sample.argmax(dim=-1, keepdim=True), 1.0)
            sample = hard_sample - soft_sample.detach() + soft_sample
        else:
            sample = soft_sample

        return sample

    @staticmethod
    def sample_gumbel(shape, eps=1e-20):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)


class Abs(nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return torch.abs(x)


class Time_emb(nn.Module):
    def __init__(self, emb_dim, time_steps, max_period):
        super(Time_emb, self).__init__()
        self.emb_dim = emb_dim
        self.time_steps = time_steps
        self.max_period = max_period

    def forward(self, t):
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        t = t.view(-1) * self.time_steps
        half = self.emb_dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.emb_dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation, latent_dim=0, act='relu'):
        super(MLP, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act = act_list[act]
        if latent_dim > 0:
            self.w = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            input_dim = latent_dim
        self.temb = nn.Sequential(Time_emb(hidden_dim, time_steps=1000, max_period=1000))
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), act)
        net = []
        for _ in range(num_layer):
            net.extend([nn.Linear(hidden_dim, hidden_dim), act])
            # net.append(ResBlock(hidden_dim, hidden_dim//4))
        net.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)
        if output_activation:
            self.out_act = act_list[output_activation]
        else:
            self.out_act = nn.Identity()

    def forward(self, x, z=None, t=None):
        if z is None:
            emb = self.emb(x)
        else:
            emb = self.w(x) * self.emb(z) + self.b(x) #self.emb(torch.cat([x,z],dim=1))#
            if t is not None:
                emb = emb + self.temb(t)
        y = self.net(emb) 
        return self.out_act(y)

class SDP_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation, latent_dim=0, act='relu'):
        super(SDP_MLP, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act = act_list[act]
        if latent_dim > 0:
            self.w = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            input_dim = latent_dim
        self.temb = nn.Sequential(Time_emb(hidden_dim, time_steps=1000, max_period=1000))
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), act)
        net = []
        for _ in range(num_layer):
            net.extend([SDPBasedLipschitzLinearLayer(hidden_dim, hidden_dim), act])
            # net.extend([nn.Linear(hidden_dim, hidden_dim), act])
            # net.append(ResBlock(hidden_dim, hidden_dim//4))
        net.append(SDPBasedLipschitzLinearLayer(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)
        if output_activation:
            self.out_act = act_list[output_activation]
        else:
            self.out_act = nn.Identity()

    def forward(self, x, z=None, t=None):
        if z is None:
            emb = self.emb(x)
        else:
            emb = self.w(x) * self.emb(z) + self.b(x) #self.emb(torch.cat([x,z],dim=1))#
            if t is not None:
                emb = emb + self.temb(t)
        y = self.net(emb) 
        return self.out_act(y)

class Simple_Hypernetwork(nn.Module):
    """
    简单超网络实现：根据条件（碳税率tau）动态生成调制参数
    
    架构：
        特征编码器: x_feat → feat_emb
        超网络: tau → gamma, beta (FiLM风格的调制参数)
        主网络: modulated_emb → output
    
    与完全超网络的区别：
        - 完全超网络：生成整个网络的所有权重
        - 这个版本：只生成调制参数，主网络参数是共享的
        - 优点：参数量更小，训练更稳定
        - 缺点：表达能力略弱于完全超网络
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, output_act, latent_dim=0, act='relu'):
        super(Simple_Hypernetwork, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act = act_list[act]
        
        # ===== 1. 特征编码器 =====
        # 将输入特征（除tau外）编码到隐空间
        self.feat_encoder = nn.Sequential(
            nn.Linear(input_dim - 1, hidden_dim),  # -1 因为tau单独处理
            act
        )
        
        # ===== 2. 超网络：tau → 调制参数 (gamma, beta) =====
        # 这里生成FiLM风格的调制参数，而不是完整的权重矩阵
        self.hypernet = nn.Sequential(
            nn.Linear(1, hidden_dim),              # tau维度是1
            act,
            nn.Linear(hidden_dim, hidden_dim * 2)  # 输出 gamma 和 beta
        )
        
        # ===== 3. 时间嵌入（如果需要） =====
        self.temb = nn.Sequential(Time_emb(hidden_dim, time_steps=1000, max_period=1000))
        
        # ===== 4. 主网络（共享参数） =====
        net = []
        for _ in range(num_layers):
            net.extend([nn.Linear(hidden_dim, hidden_dim), act])
        net.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)
        
        # ===== 5. 输出激活 =====
        if output_act:
            self.out_act = act_list[output_act]
        else:
            self.out_act = nn.Identity()

    def forward(self, x, z=None, t=None):
        """
        Args:
            x: [B, input_dim] 包含特征和tau，最后一列是tau
            z: [B, latent_dim] 可选的隐变量（用于flow matching）
            t: [B, 1] 可选的时间步（用于flow matching）
        
        Returns:
            y: [B, output_dim]
        """
        # === 步骤1：拆分输入 ===
        x_feat = x[:, :-1]  # [B, input_dim-1] 特征部分
        tau = x[:, -1:]     # [B, 1] 碳税率
        
        # === 步骤2：特征编码 ===
        feat_emb = self.feat_encoder(x_feat)  # [B, hidden_dim]
        
        # === 步骤3：超网络生成调制参数 ===
        hyper_params = self.hypernet(tau)  # [B, hidden_dim * 2]
        gamma = hyper_params[:, :hyper_params.shape[1]//2]  # [B, hidden_dim]
        beta = hyper_params[:, hyper_params.shape[1]//2:]   # [B, hidden_dim]
        
        # === 步骤4：FiLM调制 ===
        # 使用超网络生成的参数调制特征
        emb = gamma * feat_emb + beta  # [B, hidden_dim]
        
        # === 步骤5：加入时间嵌入（如果有） ===
        if t is not None:
            emb = emb + self.temb(t)
        
        # === 步骤6：如果有z，可以进一步融合 ===
        # 这里保留接口兼容性，实际使用中可能不需要
        if z is not None:
            # 可以选择加入z的信息，例如：
            # emb = emb + some_z_encoder(z)
            pass
        
        # === 步骤7：主网络计算 ===
        y = self.net(emb)  # [B, output_dim]
        
        return self.out_act(y)


class Full_Hypernetwork(nn.Module):
    """
    完全超网络实现（符合PSL论文架构）：根据条件（碳税率tau）生成整个decoder的权重
    
    架构：
        特征编码器: x_feat → feat_emb [固定参数]
        超网络: tau → W1, b1, W2, b2, ... (生成decoder的所有权重)
        动态decoder: 使用生成的权重进行计算
    
    关键特性：
        - decoder的所有参数都由超网络生成
        - 不同的tau → 完全不同的decoder
        - 符合Pareto Set Learning论文的核心思想
    
    参数复杂度：
        - 假设decoder有L层，每层hidden_dim维度
        - 需要生成的参数量：L * (hidden_dim^2 + hidden_dim)
        - 超网络的输出维度会很大，训练难度较高
    """
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, 
                 output_act, latent_dim=0, act='relu'):
        super(Full_Hypernetwork, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act_fn = act_list[act]
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.act_fn = act_fn
        
        # ===== 1. 特征编码器（固定参数）=====
        self.feat_encoder = nn.Sequential(
            nn.Linear(input_dim - 1, hidden_dim),
            act_fn
        )
        
        # ===== 2. 计算需要生成的参数总量 =====
        # 第一层: hidden_dim -> hidden_dim (W1, b1)
        # 中间层: hidden_dim -> hidden_dim (Wi, bi) * (num_layers - 1)
        # 输出层: hidden_dim -> output_dim (Wout, bout)
        
        param_counts = []
        # 第一层和中间层
        for i in range(num_layers):
            param_counts.append(hidden_dim * hidden_dim + hidden_dim)  # W + b
        # 输出层
        param_counts.append(hidden_dim * output_dim + output_dim)
        
        self.param_counts = param_counts
        self.total_params = sum(param_counts)
        
        print(f"[Full_Hypernetwork] 完全超网络统计:")
        print(f"   - Decoder层数: {num_layers + 1}")
        print(f"   - 每层参数量: {param_counts}")
        print(f"   - 总参数量: {self.total_params}")
        
        # ===== 3. 超网络：tau → decoder的所有参数 =====
        # 这是核心：一个小网络生成大量参数
        hypernet_hidden = min(512, self.total_params // 4)  # 超网络的隐藏层大小
        self.hypernet = nn.Sequential(
            nn.Linear(1, hypernet_hidden),
            act_fn,
            nn.Linear(hypernet_hidden, hypernet_hidden),
            act_fn,
            nn.Linear(hypernet_hidden, self.total_params)
        )
        
        # ===== 4. 输出激活 =====
        if output_act:
            self.out_act = act_list[output_act]
        else:
            self.out_act = nn.Identity()
    
    def forward(self, x, z=None, t=None):
        """
        Args:
            x: [B, input_dim] 包含特征和tau，最后一列是tau
            z, t: 兼容性参数（本实现中未使用）
        
        Returns:
            y: [B, output_dim]
        """
        batch_size = x.shape[0]
        
        # === 步骤1：拆分输入 ===
        x_feat = x[:, :-1]  # [B, input_dim-1]
        tau = x[:, -1:]     # [B, 1]
        
        # === 步骤2：编码特征（使用固定参数）===
        feat_emb = self.feat_encoder(x_feat)  # [B, hidden_dim]
        
        # === 步骤3：超网络生成decoder的所有参数 ===
        all_params = self.hypernet(tau)  # [B, total_params]
        
        # === 步骤4：拆分参数到各层 ===
        params_list = []
        offset = 0
        for i, count in enumerate(self.param_counts):
            params_list.append(all_params[:, offset:offset+count])
            offset += count
        
        # === 步骤5：使用生成的参数构建动态decoder ===
        h = feat_emb  # [B, hidden_dim]
        
        # 隐藏层
        for i in range(self.num_layers):
            W_flat = params_list[i][:, :self.hidden_dim * self.hidden_dim]
            b = params_list[i][:, self.hidden_dim * self.hidden_dim:]
            
            # 重塑权重: [B, hidden_dim * hidden_dim] -> [B, hidden_dim, hidden_dim]
            W = W_flat.view(batch_size, self.hidden_dim, self.hidden_dim)
            
            # 批量矩阵乘法: [B, hidden_dim] @ [B, hidden_dim, hidden_dim] -> [B, hidden_dim]
            h = torch.bmm(h.unsqueeze(1), W).squeeze(1) + b  # [B, hidden_dim]
            h = self.act_fn(h)
        
        # 输出层
        W_out_flat = params_list[-1][:, :self.hidden_dim * self.output_dim]
        b_out = params_list[-1][:, self.hidden_dim * self.output_dim:]
        
        W_out = W_out_flat.view(batch_size, self.hidden_dim, self.output_dim)
        y = torch.bmm(h.unsqueeze(1), W_out).squeeze(1) + b_out  # [B, output_dim]
        
        return self.out_act(y)
 

class CarbonTaxAwareMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer,
                 output_activation, latent_dim=0, act='relu',
                 carbon_tax_dim=1, anchor_dim=None):
        """
        input_dim: x 的总维度 = 负荷特征 + 碳税率 (tau) + 锚点 (anchor)
        carbon_tax_dim: 碳税率占用的维度，一般=1
        anchor_dim: 锚点占用的维度，一般等于 output_dim (如果为None则自动设为output_dim)
        latent_dim: z 的维度（或者说 emb(z) 的输入维）
        """

        super(CarbonTaxAwareMLP, self).__init__()

        # 激活函数表
        act_list = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'softplus': nn.Softplus(), 
            'sigmoid':  nn.Sigmoid(),
            'softmax': nn.Softmax(dim=-1),
            'gumbel': GumbelSoftmax(hard=True),
            'abs': Abs()
        }
        act_fn = act_list[act]

        # ------- 1. 基本设置 -------
        self.total_input_dim = input_dim
        self.carbon_tax_dim = carbon_tax_dim
        self.anchor_dim = anchor_dim if anchor_dim is not None else output_dim
        self.feature_dim = input_dim - carbon_tax_dim  #  - self.anchor_dim  # 除碳税率和锚点外的部分

        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # ------- 2. 针对 tau(碳税率) 的专门门控网络 -------
        # tau_gate: tau -> [hidden_dim] 缩放向量 # 作用：控制"碳税敏感度"，系统性地影响不同策略分支
        self.tau_gate = nn.Sequential(nn.Linear(self.carbon_tax_dim, hidden_dim), nn.Sigmoid())

        # ------- 3. 针对 anchor(锚点) 的专门编码网络 -------
        # anchor_encoder: anchor -> [hidden_dim] 嵌入向量  # 作用：捕获锚点信息，提供"起始状态"的上下文
        self.anchor_encoder = nn.Sequential(nn.Linear(self.anchor_dim, hidden_dim), act_fn)

        # ------- 4. 原本的 W(x) 和 B(x) / emb(z) 结构 -------
        # 注意：这里用的是除碳税率和锚点外的 x_feat (负荷等特征)
        if latent_dim > 0:
            # W 和 B 只依赖 x_feat (负荷等特征)
            self.w = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim)) 
            self.emb_z = nn.Sequential(nn.Linear(latent_dim, hidden_dim), act_fn)
        else:
            # 没有 latent，则直接把 x_feat 过 emb_x
            self.emb_x = nn.Sequential(nn.Linear(self.feature_dim, hidden_dim), act_fn)

        # ------- 5. 时间embedding 保留 -------
        self.temb = nn.Sequential(
            Time_emb(hidden_dim, time_steps=1000, max_period=1000)
        )

        # ------- 6. 主干网络 (和原来一致) -------
        net = []
        for _ in range(num_layer):
            net.extend([
                nn.Linear(hidden_dim, hidden_dim),
                act_fn
            ])
        net.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)

        # 输出激活
        if output_activation:
            self.out_act = act_list[output_activation]
        else:
            self.out_act = nn.Identity()


    def forward(self, x, z=None, t=None):
        """
        x: [B, total_input_dim] = [features..., tau, anchor]
        z: [B, latent_dim] or None
        t: [B, 1] or [B]
        """

        # --- 拆分出负荷特征、碳税率tau、锚点anchor ---
        # 假设结构为: [负荷特征 | 碳税率 | 锚点]
        x_feat = x[..., :self.feature_dim]                           # [B, feature_dim]
        tau    = x[..., self.feature_dim:self.feature_dim + self.carbon_tax_dim]  # [B, carbon_tax_dim]
        anchor = x[..., self.feature_dim + self.carbon_tax_dim:]     # [B, anchor_dim]

        # --- 分别编码三个关键因素 ---
        # 1. tau 门控向量，形状 [B, hidden_dim], 值域(0,1)
        tau_gate = self.tau_gate(tau)
        # 2. anchor 编码向量，形状 [B, hidden_dim]
        anchor_emb = self.anchor_encoder(anchor)

        # --- 走两条路：有 z (FiLM 风格) / 无 z (直接 encode x_feat) ---
        if z is None or self.latent_dim == 0:
            # 没有 z：就像原始分支的 if z is None
            emb = self.emb_x(x_feat)               # [B, hidden_dim]
        else:
            # 有 z：FiLM/调制
            # emb_z(z) -> [B, hidden_dim]
            # w(x_feat), b(x_feat) -> [B, hidden_dim]
            w_x  = self.w(x_feat)                  # [B, hidden_dim]
            b_x  = self.b(x_feat)                  # [B, hidden_dim]
            zemb = self.emb_z(z)                   # [B, hidden_dim]

            # emb = w(x)*emb(z) + b(x)
            emb = w_x * zemb + b_x                 # [B, hidden_dim]

        # --- 注入时间信息 ---
        if t is not None:
            emb = emb + self.temb(t)               # [B, hidden_dim]

        # --- 关键改动：融合三个因素 ---
        # 1. 先加入锚点信息（加法融合）
        emb = emb + anchor_emb                     # [B, hidden_dim]
        
        # 2. 再用碳税率进行门控（乘法调制）
        # 解释：当碳税率升高时，网络会系统性地偏向"高碳惩罚/低排放解"方向
        # tau_gate 是 [0,1]，可以理解成在每个隐藏通道上开/关不同策略分支
        emb = emb * tau_gate                       # [B, hidden_dim]

        # --- 后续主干 MLP + 输出激活 ---
        y = self.net(emb)                          # [B, output_dim]
        return self.out_act(y)



class Lip_MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation, latent_dim=0):
        super(Lip_MLP, self).__init__()
        if latent_dim > 0:
            w = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
            b = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)]
            t = [Time_emb(hidden_dim, time_steps=1000, max_period=1000)]
            self.w = nn.Sequential(*w)
            self.b = nn.Sequential(*b)
            self.t = nn.Sequential(*t)
            net = []
        else:
            latent_dim = input_dim

        emb = [LinearNormalized(latent_dim, hidden_dim), nn.ReLU()]
        self.emb = nn.Sequential(*emb)
        net = []
        for _ in range(num_layer):
            net.extend([LinearNormalized(hidden_dim, hidden_dim), nn.ReLU()])
        net.append(LinearNormalized(hidden_dim, output_dim))
        self.net = nn.Sequential(*net)

        if output_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif output_activation == 'gumbel':
            self.act = GumbelSoftmax(hard=True)
        elif output_activation == 'abs':
            self.act = Abs()
        else:
            self.act = nn.Identity()

    def forward(self, x, z=None, t=None):
        if z is None:
            emb = self.emb(x)
        else:
            emb = self.w(x) * self.emb(z) + self.b(x)
            if t is not None:
                emb = emb + self.t(t)
        y = self.net(emb)
        return self.act(y)

    def project_weights(self):
        self.net.project_weights()


class ATT(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layer, output_activation=None, latent_dim=0, agg=False,
                 pred_type='node', act='relu'):
        super(ATT, self).__init__()
        act_list = {'relu': nn.ReLU(), 'silu': nn.SiLU(), 'softplus': nn.Softplus(), 
                    'sigmoid':  nn.Sigmoid(), 'softmax': nn.Softmax(dim=-1),
                      'gumbel': GumbelSoftmax(hard=True), 'abs': Abs()}
        act = act_list[act]
        if latent_dim > 0:
            self.w = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            self.b = nn.Sequential(nn.Linear(input_dim, hidden_dim))
            input_dim = latent_dim
        self.temb = nn.Sequential(Time_emb(hidden_dim, time_steps=1000, max_period=1000))
        self.emb = nn.Sequential(nn.Linear(input_dim, hidden_dim), act)

        net = []
        # for _ in range(num_layer):
            # net.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            # net.extend([MHA(hidden_dim, 64, hidden_dim // 64),
            #             ResBlock(hidden_dim, hidden_dim//4)])
        # net.append(nn.Linear(hidden_dim, output_dim))
        # self.net = nn.Sequential(*net)
        self.net = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=max(hidden_dim // 64, 1))
        self.trans = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)
        # self.mha  = MHA(hidden_dim, 64, hidden_dim // 64)

        self.agg = agg
        self.pred_type = pred_type

        if output_activation == 'sigmoid':
            self.act = nn.Sigmoid()
        elif output_activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
        elif output_activation == 'gumbel':
            self.act = GumbelSoftmax(hard=True)
        elif output_activation == 'abs':
            self.act = Abs()
        else:
            self.act = nn.Identity()

    def forward(self, x, z=None, t=None):
        ### x: B * N * F
        ### z: B * N
        ### t: B * 1
        batch_size = x.shape[0]
        node_size = x.shape[1]
        if z is None:
            # print(x.shape, self.emb)
            emb = self.emb(x)
        else:
            z = z.view(batch_size, -1, 1)
            emb = self.w(x) * self.emb(z) + self.b(x)
            if t is not None:
                emb = emb + self.temb(t).view(batch_size, 1, -1)
        emb = emb.permute(1, 0, 2)
        emb = self.trans(emb)
        emb = emb.permute(1, 0, 2)
        y = self.net(emb)  # B * N * 1
        
        if self.agg:
            y = y.mean(1)
        else:
            if self.pred_type == 'node':
                y = y.view(x.shape[0], -1)  # B * N
            else:
                y = torch.matmul(y.view(batch_size, node_size, 1), y.view(batch_size, 1, node_size))  # B * N * N
                col, row = torch.triu_indices(node_size, node_size, 1)
                y = y[:, col, row]
        return self.act(y)


class MHA(nn.Module):
    def __init__(self, n_in, n_emb, n_head):
        super().__init__()
        self.n_emb = n_emb
        self.n_head = n_head
        self.key = nn.Linear(n_in, n_in)
        self.query = nn.Linear(n_in, n_in)
        self.value = nn.Linear(n_in, n_in)
        self.proj = nn.Linear(n_in, n_in)

    def forward(self, x):
        # x: B * node * n_in
        batch = x.shape[0]
        node = x.shape[1]
        ### softmax
        #### key: B H node emb
        #### que: B H emb node
        key = self.key(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,1,2)
        query = self.query(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,2,1)
        value = self.key(x).view(batch, node, self.n_emb, self.n_head).permute(0,3,1,2)
        score = torch.matmul(key, query)/(self.n_emb**0.5) # x: B * H * node * node
        prob = torch.softmax(score, dim=-1) # B * H * node * node (prob)
        out = torch.matmul(prob, value) # B * H * Node * 64
        out = out.permute(0,2,3,1).contiguous() # B * N * F * H
        out = out.view(batch, -1, self.n_emb*self.n_head)
        return x + self.proj(out)


class SimpleResBlock(nn.Module):
    """简单的残差块，使用固定的ReLU激活函数"""
    def __init__(self, n_in, n_hid):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, n_hid), 
                                 nn.ReLU(),
                                 nn.Linear(n_hid, n_in))
    def forward(self, x):
        return x + self.net(x)


class SDPBasedLipschitzLinearLayer(nn.Module):
    def __init__(self, cin, cout, epsilon=1e-6):
        super(SDPBasedLipschitzLinearLayer, self).__init__()

        self.activation = nn.ReLU(inplace=False)
        self.weights = nn.Parameter(torch.empty(cout, cin))
        self.bias = nn.Parameter(torch.empty(cout))
        self.q = nn.Parameter(torch.rand(cout))

        nn.init.xavier_normal_(self.weights)
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        bound = 1 / (fan_in ** 0.5)
        nn.init.uniform_(self.bias, -bound, bound)  # bias init

        self.epsilon = epsilon

    def forward(self, x):
        res = F.linear(x, self.weights, self.bias)
        res = self.activation(res)
        q_abs = torch.abs(self.q)
        q = q_abs[None, :]
        q_inv = (1 / (q_abs + self.epsilon))[:, None]
        T = 2 / (torch.abs(q_inv * self.weights @ self.weights.T * q).sum(1) + self.epsilon)
        res = T * res
        res = F.linear(res, self.weights.t())
        out = x - res
        return out


class LinearNormalized(nn.Linear):

    def __init__(self, in_features, out_features, bias=True):
        super(LinearNormalized, self).__init__(in_features, out_features, bias)
        self.linear = spectral_norm(nn.Linear(in_features, out_features))

    def forward(self, x):
        return self.linear(x)


class PartialLinearNormalized(nn.Module):
    def __init__(self, input_dim, output_dim, con_dim):
        super(PartialLinearNormalized, self).__init__()
        self.con_dim = con_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear_1 = spectral_norm(nn.Linear(input_dim - con_dim, output_dim))

    def forward(self, x):
        with torch.no_grad():
            weight_copy = self.linear_1.weight.data.clone()
            self.linear.weight.data[:, self.con_dim:] = weight_copy
        return self.linear(x)


class distance_estimator(nn.Module):
    def __init__(self, n_feats, n_hid_params, hidden_layers, n_projs=2, beta=0.5):
        super().__init__()
        self.hidden_layers = hidden_layers  # number of hidden layers in the network
        self.n_projs = n_projs  # number of projections to use for weights onto Steiffel manifold
        self.beta = beta  # scalar in (0,1) for stabilizing feed forward operations

        # Intialize initial, middle, and final layers
        self.fc_one = torch.nn.Linear(n_feats, n_hid_params, bias=True)
        self.fc_mid = nn.ModuleList(
            [torch.nn.Linear(n_hid_params, n_hid_params, bias=True) for i in range(self.hidden_layers)])
        self.fc_fin = torch.nn.Linear(n_hid_params, 1, bias=True)

        # Normalize weights (helps ensure stability with learning rate)
        self.fc_one.weight = nn.Parameter(self.fc_one.weight / torch.norm(self.fc_one.weight))
        for i in range(self.hidden_layers):
            self.fc_mid.weight = nn.Parameter(self.fc_mid[i].weight / torch.norm(self.fc_mid[i].weight))
        self.fc_fin.weight = nn.Parameter(self.fc_fin.weight / torch.norm(self.fc_fin.weight))

    def forward(self, u):
        u = self.fc_one(u).sort(1)[0]  # Apply first layer affine mapping
        for i in range(self.hidden_layers):  # Loop for each hidden layer
            u = u + self.beta * (self.fc_mid[i](u).sort(1)[0] - u)  # Convex combo of u and sort(W*u+b)
        u = self.fc_fin(u)  # Final layer is scalar (no need to sort)
        J = torch.abs(u)
        return J

    def project_weights(self):
        self.fc_one.weight.data = self.proj_Stiefel(self.fc_one.weight.data, self.n_projs)
        for i in range(self.hidden_layers):
            self.fc_mid[i].weight.data = self.proj_Stiefel(self.fc_mid[i].weight.data, self.n_projs)
        self.fc_fin.weight.data = self.proj_Stiefel(self.fc_fin.weight.data, self.n_projs)

    def proj_Stiefel(self, Ak, proj_iters):  # Project to closest orthonormal matrix
        n = Ak.shape[1]
        I = torch.eye(n)
        for k in range(proj_iters):
            Qk = I - Ak.permute(1, 0).matmul(Ak)
            Ak = Ak.matmul(I + 0.5 * Qk)
        return Ak


