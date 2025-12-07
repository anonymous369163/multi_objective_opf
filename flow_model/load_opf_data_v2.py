"""
修改版数据加载器 - 为流模型优化
让流模型真正从anchor流向target
"""
import torch
import numpy as np
import os


def filter_training_data(x_train, y_train, env, actor, max_violation=0.1):
    """过滤掉严重违反约束的训练样本"""
    print("\n数据清洗中...")

    # 判断actor模型所在设备，并将输入数据移到相同设备
    device = next(actor.parameters()).device if hasattr(actor, 'parameters') else torch.device('cpu')
    x_train_device = x_train.to(device) if isinstance(x_train, torch.Tensor) else torch.tensor(x_train, device=device)
    y_train_device = y_train.to(device) if isinstance(y_train, torch.Tensor) else torch.tensor(y_train, device=device)
    
    output_dim = y_train_device.shape[1]
    vm = y_train_device[:, :output_dim//2]
    va = y_train_device[:, output_dim//2:]

    with torch.no_grad():
        violations = actor.compute_constraint_loss(
            vm, va, x_train_device, env, reduction='none'
        )

    # 只保留约束违反小于阈值的样本
    valid_mask = violations < max_violation

    x_train_filtered = x_train_device[valid_mask]
    y_train_filtered = y_train_device[valid_mask]

    # 若数据原本在CPU上，则转回CPU，避免潜在环境不一致
    x_train_filtered = x_train_filtered.cpu()
    y_train_filtered = y_train_filtered.cpu()

    print(f"原始样本数: {len(x_train)}")
    print(f"过滤后样本数: {len(x_train_filtered)}")
    print(f"保留率: {len(x_train_filtered)/len(x_train)*100:.1f}%")

    return x_train_filtered, y_train_filtered


def create_toy_dataset_with_clustering(data_path, n_clusters=30, train_samples=800, test_samples=200,
                                        random_seed=42, device='cpu', add_carbon_tax=False):
    """
    通过 K-means 聚类创建分布相似的训练集和测试集
    
    确保训练集和测试集来自相同的聚类分布，但样本不重叠。
    
    Args:
        data_path: 数据文件路径
        n_clusters: K-means 聚类数量
        train_samples: 训练集样本数量
        test_samples: 测试集样本数量
        random_seed: 随机种子
        device: 设备
        add_carbon_tax: 是否添加碳税特征
        
    Returns:
        dict: {
            'x_train': 训练输入,
            'y_train': 训练目标,
            'x_test': 测试输入,
            'y_test': 测试目标,
            'cluster_labels': 聚类标签,
            'cluster_info': 聚类统计信息
        }
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    print("=" * 60)
    print("创建 Toy 数据集 (K-means 聚类采样)")
    print("=" * 60)
    
    # 加载原始数据
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    data = np.load(data_path)
    train_inputs = data['train_inputs']      # [N, load_dim]
    train_targets = data['train_targets']    # [N, target_dim]
    
    n_total = train_inputs.shape[0]
    print(f"\n原始数据: {n_total} 样本")
    print(f"输入维度: {train_inputs.shape[1]}")
    print(f"输出维度: {train_targets.shape[1]}")
    
    # 标准化输入特征用于聚类
    np.random.seed(random_seed)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(train_inputs)
    
    # K-means 聚类
    print(f"\n正在进行 K-means 聚类 (k={n_clusters})...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
    cluster_labels = kmeans.fit_predict(x_scaled)
    
    # 统计每个簇的样本数
    cluster_counts = np.bincount(cluster_labels, minlength=n_clusters)
    print(f"聚类完成:")
    print(f"  - 簇大小范围: [{cluster_counts.min()}, {cluster_counts.max()}]")
    print(f"  - 簇平均大小: {cluster_counts.mean():.1f}")
    
    # 计算每个簇应采样的训练/测试样本数
    total_samples = train_samples + test_samples
    train_ratio = train_samples / total_samples
    
    # 从每个簇按比例采样
    train_indices = []
    test_indices = []
    
    for cluster_id in range(n_clusters):
        # 获取该簇的所有样本索引
        cluster_mask = (cluster_labels == cluster_id)
        cluster_indices = np.where(cluster_mask)[0]
        n_cluster = len(cluster_indices)
        
        if n_cluster == 0:
            continue
        
        # 计算该簇应采样的数量（按簇大小比例）
        cluster_ratio = n_cluster / n_total
        n_train_from_cluster = max(1, int(train_samples * cluster_ratio))
        n_test_from_cluster = max(1, int(test_samples * cluster_ratio))
        
        # 确保不超过簇大小
        n_total_from_cluster = n_train_from_cluster + n_test_from_cluster
        if n_total_from_cluster > n_cluster:
            scale = n_cluster / n_total_from_cluster
            n_train_from_cluster = max(1, int(n_train_from_cluster * scale))
            n_test_from_cluster = max(1, int(n_test_from_cluster * scale))
        
        # 随机打乱并分配
        np.random.shuffle(cluster_indices)
        train_indices.extend(cluster_indices[:n_train_from_cluster])
        test_indices.extend(cluster_indices[n_train_from_cluster:n_train_from_cluster + n_test_from_cluster])
    
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    
    # 如果样本数不足，从剩余样本中补充
    all_used = set(train_indices) | set(test_indices)
    remaining = np.array([i for i in range(n_total) if i not in all_used])
    
    if len(train_indices) < train_samples and len(remaining) > 0:
        n_need = min(train_samples - len(train_indices), len(remaining))
        np.random.shuffle(remaining)
        train_indices = np.concatenate([train_indices, remaining[:n_need]])
        remaining = remaining[n_need:]
    
    if len(test_indices) < test_samples and len(remaining) > 0:
        n_need = min(test_samples - len(test_indices), len(remaining))
        test_indices = np.concatenate([test_indices, remaining[:n_need]])
    
    # 最终打乱顺序
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)
    
    # 提取数据
    x_train = train_inputs[train_indices]
    y_train = train_targets[train_indices]
    x_test = train_inputs[test_indices]
    y_test = train_targets[test_indices]
    
    print(f"\nToy 数据集创建完成:")
    print(f"  - 训练集: {len(x_train)} 样本")
    print(f"  - 测试集: {len(x_test)} 样本")
    print(f"  - 训练/测试比例: {len(x_train)/(len(x_train)+len(x_test))*100:.1f}%/{len(x_test)/(len(x_train)+len(x_test))*100:.1f}%")
    
    # 验证分布相似性
    train_cluster_dist = np.bincount(cluster_labels[train_indices], minlength=n_clusters) / len(train_indices)
    test_cluster_dist = np.bincount(cluster_labels[test_indices], minlength=n_clusters) / len(test_indices)
    
    # 计算分布差异 (Jensen-Shannon divergence 的简化版)
    dist_diff = np.abs(train_cluster_dist - test_cluster_dist).mean()
    print(f"  - 训练/测试分布差异: {dist_diff:.4f} (越小越好)")
    
    # 转换为 tensor
    x_train = torch.as_tensor(x_train, dtype=torch.float32, device=device)
    y_train = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    x_test = torch.as_tensor(x_test, dtype=torch.float32, device=device)
    y_test = torch.as_tensor(y_test, dtype=torch.float32, device=device)
    
    print("=" * 60)
    
    return {
        'x_train': x_train,
        'y_train': y_train,
        'x_test': x_test,
        'y_test': y_test,
        'train_indices': train_indices,
        'test_indices': test_indices,
        'cluster_labels': cluster_labels,
        'cluster_info': {
            'n_clusters': n_clusters,
            'cluster_counts': cluster_counts,
            'train_cluster_dist': train_cluster_dist,
            'test_cluster_dist': test_cluster_dist,
            'dist_diff': dist_diff
        }
    }


class OPF_Flow_Dataset_V2:
    """
    改进的数据集类，专门为从anchor到target的流模型设计
    
    数据格式：
    - x: [负荷c, 碳税λ]  (不包含anchor，因为anchor是流的起点)
    - y: [目标target]
    - y_anchor: 单独保存，用作流模型的x_0
    
    改进：在加载时自动分割训练集和测试集，确保验证模型泛化能力
    """
    
    def __init__(self, data_path, device='cpu', test_ratio=0.2, random_seed=42, add_carbon_tax=True, single_target=False):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            device: 设备 ('cpu' 或 'cuda')
            test_ratio: 测试集比例，默认0.2 (即20%测试，80%训练)
            random_seed: 随机种子，确保可复现性
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        print(f"正在加载数据: {data_path}")
        data = np.load(data_path)
        self.single_target = single_target
        
        # 加载各个数据字段
        train_inputs = data['train_inputs']      # shape: [N, load_dim]
        train_targets = data['train_targets']    # shape: [N, target_dim]
        if not single_target:
            preferences = data['preferences']        # shape: [N, 1] 或 [N,]
            y_anchors = data['y_anchors']           # shape: [N, target_dim]
            actions = data['actions']                # shape: [N, output_dim]
        
        # 确保preferences是2D数组
        if not single_target and preferences.ndim == 1:
            preferences = preferences.reshape(-1, 1)
        
        print(f"原始数据形状:")
        print(f"  - train_inputs (负荷c): {train_inputs.shape}")
        print(f"  - train_targets (目标y): {train_targets.shape}")
        if not single_target:
            print(f"  - preferences (碳税率): {preferences.shape}")
            print(f"  - y_anchors (锚点): {y_anchors.shape}")
            print(f"  - actions (动作): {actions.shape}")
        
        # V2版本: x只包含[负荷c, 碳税λ]，不包含anchor
        # anchor作为流的起点单独处理
        if add_carbon_tax:
            x_combined = np.concatenate([
                train_inputs,    # 负荷
                preferences,     # 碳税率（目标权重）
            ], axis=1)
        else:
            x_combined = train_inputs 
        
        # 输出就是目标决策变量
        y_combined = train_targets
        
        # ===== 新增：自动分割训练集和测试集 =====
        np.random.seed(random_seed)  # 设置随机种子确保可复现
        n_samples = x_combined.shape[0]
        indices = np.random.permutation(n_samples)
        
        # 计算分割点
        n_test = int(n_samples * test_ratio)
        n_train = n_samples - n_test
        
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        print(f"\n数据分割:")
        print(f"  - 总样本数: {n_samples}")
        print(f"  - 训练集: {n_train} ({100*(1-test_ratio):.0f}%)")
        print(f"  - 测试集: {n_test} ({100*test_ratio:.0f}%)")
        print(f"  - 随机种子: {random_seed}")
        
        # 分割数据
        x_train_split = x_combined[train_indices]
        y_train_split = y_combined[train_indices]
        if not single_target:
            y_anchor_train = y_anchors[train_indices]
        
        x_test_split = x_combined[test_indices]
        y_test_split = y_combined[test_indices]
        if not single_target:
            y_anchor_test = y_anchors[test_indices]
        
        # 转换为torch tensor - 训练集
        self.x_train = torch.as_tensor(x_train_split, dtype=torch.float32)
        self.y_train = torch.as_tensor(y_train_split, dtype=torch.float32)
        if not single_target:
            self.y_anchor_train = torch.as_tensor(y_anchor_train, dtype=torch.float32)
        
        # 转换为torch tensor - 测试集
        self.x_test = torch.as_tensor(x_test_split, dtype=torch.float32)
        self.y_test = torch.as_tensor(y_test_split, dtype=torch.float32)
        if not single_target:
            self.y_anchor_test = torch.as_tensor(y_anchor_test, dtype=torch.float32)
        
        # 保存原始数据（用于分析）- 分割后的
        self.train_inputs = train_inputs[train_indices]
        self.train_targets = train_targets[train_indices]
        if not single_target:
            self.preferences_train = preferences[train_indices]
            self.y_anchors_train = y_anchors[train_indices]
            self.actions_train = actions[train_indices]
        
        self.test_inputs = train_inputs[test_indices]
        self.test_targets = train_targets[test_indices]
        if not single_target:
            self.preferences_test = preferences[test_indices]
            self.y_anchors_test = y_anchors[test_indices]
            self.actions_test = actions[test_indices]

        # 设置设备
        self.device = torch.device(device)
        self.analyze_data()
        
        # 数据统计信息
    def analyze_data(self):
        self.num_train_samples = self.x_train.shape[0]
        self.num_test_samples = self.x_test.shape[0]
        self.num_samples = self.num_train_samples  # 保持向后兼容
        self.input_dim = self.x_train.shape[1]
        self.output_dim = self.y_train.shape[1]
        self.load_dim = self.train_inputs.shape[1]
        self.target_dim = self.train_targets.shape[1]
        
        print(f"\n数据集构建完成 (为流模型优化 + 训练/测试分割):")
        print(f"  - 训练样本: {self.num_train_samples}")
        print(f"  - 测试样本: {self.num_test_samples}")
        print(f"  - 输入维度: {self.input_dim} (负荷:{self.load_dim} + 偏好:1)")
        print(f"  - 输出维度: {self.output_dim}")
        if not self.single_target:
            print(f"  - 偏好维度: {self.preferences_train.shape[1]}")
            print(f"  - 锚点维度: {self.y_anchor_train.shape[1]}")
            print(f"  - 动作维度: {self.actions_train.shape[1]}") 
        print(f"  - 设备: {self.device}")
        
    def to(self, device):
        """将数据移动到指定设备"""
        self.device = torch.device(device)
        # 移动训练集
        self.x_train = self.x_train.to(self.device)
        self.y_train = self.y_train.to(self.device)
        self.y_anchor_train = self.y_anchor_train.to(self.device)
        # 移动测试集
        self.x_test = self.x_test.to(self.device)
        self.y_test = self.y_test.to(self.device)
        self.y_anchor_test = self.y_anchor_test.to(self.device)
        return self
    
    def get_data_info(self):
        """获取数据集信息"""
        return {
            'num_train_samples': self.num_train_samples,
            'num_test_samples': self.num_test_samples,
            'total_samples': self.num_train_samples + self.num_test_samples,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'load_dim': self.load_dim,
            'target_dim': self.target_dim,
            'train_preference_range': (self.preferences_train.min(), self.preferences_train.max()),
            'test_preference_range': (self.preferences_test.min(), self.preferences_test.max()),
        }
    
    def get_train_data(self):
        """获取训练集数据"""
        return self.x_train, self.y_train, self.y_anchor_train
    
    def get_test_data(self):
        """获取测试集数据"""
        return self.x_test, self.y_test, self.y_anchor_test


class OPF_Flow_Dataset_Grouped:
    """
    分组数据集类：将同一负荷场景下不同碳税率(偏好)的样本组织在一起
    
    这个类专门为设计对比学习损失函数而设计，使得在训练时：
    - 一个batch内包含同一负荷场景下多个偏好的最优解
    - 可以计算同一场景下不同偏好解之间的关系
    - 有助于模型学习偏好对解的影响规律
    
    数据组织：
    - 识别独特的负荷场景
    - 每个场景包含多个(preference, target, anchor)三元组
    - 提供场景级别的批次采样
    """
    
    def __init__(self, data_path, device='cpu', test_ratio=0.2, random_seed=42, 
                 add_carbon_tax=True, scenario_test_ratio=0.2):
        """
        初始化分组数据集
        
        Args:
            data_path: 数据文件路径
            device: 设备 ('cpu' 或 'cuda')
            test_ratio: 未使用（保持API兼容）
            random_seed: 随机种子，确保可复现性
            add_carbon_tax: 是否将碳税加入输入特征
            scenario_test_ratio: 场景级别的测试集比例（按场景划分，非样本）
        """
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件不存在: {data_path}")
        
        print(f"=" * 70)
        print(f"正在加载分组数据集: {data_path}")
        print(f"=" * 70)
        
        data = np.load(data_path)
        
        # 加载各个数据字段
        train_inputs = data['train_inputs']      # shape: [N, load_dim]
        train_targets = data['train_targets']    # shape: [N, target_dim]
        preferences = data['preferences']        # shape: [N, 1] 或 [N,]
        y_anchors = data['y_anchors']           # shape: [N, target_dim]
        
        # 确保preferences是2D数组
        if preferences.ndim == 1:
            preferences = preferences.reshape(-1, 1)
        
        print(f"\n原始数据形状:")
        print(f"  - train_inputs (负荷c): {train_inputs.shape}")
        print(f"  - train_targets (目标y): {train_targets.shape}")
        print(f"  - preferences (碳税率): {preferences.shape}")
        print(f"  - y_anchors (锚点): {y_anchors.shape}")
        
        # ===== 核心：识别和分组负荷场景 =====
        print(f"\n正在分析负荷场景...")
        self._group_by_scenario(train_inputs, train_targets, preferences, y_anchors)
        
        # ===== 场景级别的训练/测试分割 =====
        print(f"\n正在进行场景级别的数据分割...")
        np.random.seed(random_seed)
        n_scenarios = len(self.scenario_indices)
        scenario_ids = np.arange(n_scenarios)
        np.random.shuffle(scenario_ids)
        
        n_test_scenarios = int(n_scenarios * scenario_test_ratio)
        n_train_scenarios = n_scenarios - n_test_scenarios
        
        self.train_scenario_ids = scenario_ids[n_test_scenarios:]
        self.test_scenario_ids = scenario_ids[:n_test_scenarios]
        
        print(f"  - 总场景数: {n_scenarios}")
        print(f"  - 训练场景: {n_train_scenarios} ({100*(1-scenario_test_ratio):.0f}%)")
        print(f"  - 测试场景: {n_test_scenarios} ({100*scenario_test_ratio:.0f}%)")
        
        # 统计训练和测试样本数
        n_train_samples = sum(len(self.scenario_indices[sid]) for sid in self.train_scenario_ids)
        n_test_samples = sum(len(self.scenario_indices[sid]) for sid in self.test_scenario_ids)
        print(f"  - 训练样本总数: {n_train_samples}")
        print(f"  - 测试样本总数: {n_test_samples}")
        
        # 保存原始数据
        self.train_inputs_raw = train_inputs
        self.train_targets_raw = train_targets
        self.preferences_raw = preferences
        self.y_anchors_raw = y_anchors
        
        self.add_carbon_tax = add_carbon_tax
        self.device = torch.device(device)
        self.load_dim = train_inputs.shape[1]
        self.target_dim = train_targets.shape[1]
        
        print(f"\n" + "=" * 70)
        print(f"分组数据集构建完成！")
        print(f"=" * 70)
    
    def _group_by_scenario(self, train_inputs, train_targets, preferences, y_anchors):
        """
        根据负荷场景分组数据
        
        使用哈希来识别相同的负荷场景（容忍数值误差）
        """
        from collections import defaultdict
        
        # 用于存储场景的字典：scenario_hash -> list of sample indices
        scenario_dict = defaultdict(list)
        
        # 将每个负荷场景转换为哈希值（四舍五入以容忍浮点误差）
        n_samples = train_inputs.shape[0]
        for i in range(n_samples):
            # 将负荷向量四舍五入到小数点后6位并转为元组作为key
            load_key = tuple(np.round(train_inputs[i], decimals=6))
            scenario_dict[load_key].append(i)
        
        # 转换为列表格式
        self.unique_scenarios = []  # 存储独特的负荷场景
        self.scenario_indices = []  # 每个场景对应的样本索引列表
        self.scenario_preference_counts = []  # 每个场景有多少个偏好
        
        for load_key, indices in scenario_dict.items():
            self.unique_scenarios.append(np.array(load_key))
            self.scenario_indices.append(indices)
            self.scenario_preference_counts.append(len(indices))
        
        # 转换为numpy数组
        self.unique_scenarios = np.array(self.unique_scenarios)
        
        # 统计信息
        n_unique_scenarios = len(self.unique_scenarios)
        min_prefs = min(self.scenario_preference_counts)
        max_prefs = max(self.scenario_preference_counts)
        avg_prefs = np.mean(self.scenario_preference_counts)
        
        print(f"  - 识别出 {n_unique_scenarios} 个独特的负荷场景")
        print(f"  - 每个场景的偏好数量: 最小={min_prefs}, 最大={max_prefs}, 平均={avg_prefs:.1f}")
        
        # 显示场景分布直方图
        pref_counts = np.array(self.scenario_preference_counts)
        unique_counts = np.unique(pref_counts)
        print(f"\n  场景-偏好分布:")
        for count in unique_counts:
            n_scenarios_with_count = np.sum(pref_counts == count)
            print(f"    - {count} 个偏好: {n_scenarios_with_count} 个场景")
    
    def get_scenario_batch(self, scenario_ids, split='train'):
        """
        获取指定场景的批次数据
        
        Args:
            scenario_ids: 场景ID列表或单个场景ID
            split: 'train' 或 'test'（暂未使用，但保留以便扩展）
            
        Returns:
            x_batch: [total_samples, input_dim] 输入特征
            y_batch: [total_samples, target_dim] 目标输出
            y_anchor_batch: [total_samples, target_dim] 锚点
            scenario_masks: list，每个元素是一个bool数组，标识属于哪个场景
        """
        if isinstance(scenario_ids, int):
            scenario_ids = [scenario_ids]
        
        x_list = []
        y_list = []
        y_anchor_list = []
        scenario_masks = []
        
        current_idx = 0
        for sid in scenario_ids:
            indices = self.scenario_indices[sid]
            n_samples_in_scenario = len(indices)
            
            # 提取该场景的所有样本
            loads = self.train_inputs_raw[indices]
            targets = self.train_targets_raw[indices]
            prefs = self.preferences_raw[indices]
            anchors = self.y_anchors_raw[indices]
            
            # 构建输入 x
            if self.add_carbon_tax:
                x = np.concatenate([loads, prefs], axis=1)
            else:
                x = loads
            
            x_list.append(x)
            y_list.append(targets)
            y_anchor_list.append(anchors)
            
            # 记录该场景的样本掩码
            mask = np.zeros(current_idx + n_samples_in_scenario + 
                          sum(len(self.scenario_indices[s]) for s in scenario_ids[len(scenario_masks)+1:]), 
                          dtype=bool)
            mask[current_idx:current_idx + n_samples_in_scenario] = True
            scenario_masks.append(mask[:current_idx + n_samples_in_scenario])
            current_idx += n_samples_in_scenario
        
        # 合并所有场景的数据
        x_batch = np.concatenate(x_list, axis=0)
        y_batch = np.concatenate(y_list, axis=0)
        y_anchor_batch = np.concatenate(y_anchor_list, axis=0)
        
        # 转换为tensor
        x_batch = torch.as_tensor(x_batch, dtype=torch.float32, device=self.device)
        y_batch = torch.as_tensor(y_batch, dtype=torch.float32, device=self.device)
        y_anchor_batch = torch.as_tensor(y_anchor_batch, dtype=torch.float32, device=self.device)
        
        # 修正scenario_masks
        scenario_masks_corrected = []
        start = 0
        for sid in scenario_ids:
            n = len(self.scenario_indices[sid])
            mask = torch.zeros(len(x_batch), dtype=torch.bool, device=self.device)
            mask[start:start+n] = True
            scenario_masks_corrected.append(mask)
            start += n
        
        return x_batch, y_batch, y_anchor_batch, scenario_masks_corrected
    
    def create_scenario_batches(self, batch_size=32, split='train', shuffle=True):
        """
        创建场景级别的批次迭代器
        
        Args:
            batch_size: 每个batch包含的场景数量（不是样本数）
            split: 'train' 或 'test'
            shuffle: 是否打乱场景顺序
            
        Yields:
            每次yield一个batch的数据：(x_batch, y_batch, y_anchor_batch, scenario_masks)
        """
        if split == 'train':
            scenario_ids = self.train_scenario_ids.copy()
        else:
            scenario_ids = self.test_scenario_ids.copy()
        
        if shuffle:
            np.random.shuffle(scenario_ids)
        
        # 按batch_size分批
        n_batches = (len(scenario_ids) + batch_size - 1) // batch_size
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(scenario_ids))
            batch_scenario_ids = scenario_ids[start_idx:end_idx]
            
            yield self.get_scenario_batch(batch_scenario_ids, split=split)
    
    def get_data_info(self):
        """获取数据集信息"""
        n_train_scenarios = len(self.train_scenario_ids)
        n_test_scenarios = len(self.test_scenario_ids)
        n_train_samples = sum(len(self.scenario_indices[sid]) for sid in self.train_scenario_ids)
        n_test_samples = sum(len(self.scenario_indices[sid]) for sid in self.test_scenario_ids)
        
        return {
            'n_unique_scenarios': len(self.unique_scenarios),
            'n_train_scenarios': n_train_scenarios,
            'n_test_scenarios': n_test_scenarios,
            'n_train_samples': n_train_samples,
            'n_test_samples': n_test_samples,
            'load_dim': self.load_dim,
            'target_dim': self.target_dim,
            'input_dim': self.load_dim + (1 if self.add_carbon_tax else 0),
            'preference_range': (self.preferences_raw.min(), self.preferences_raw.max()),
            'avg_preferences_per_scenario': np.mean(self.scenario_preference_counts),
            'min_preferences_per_scenario': min(self.scenario_preference_counts),
            'max_preferences_per_scenario': max(self.scenario_preference_counts),
        }
    
    def to(self, device):
        """将数据移动到指定设备"""
        self.device = torch.device(device)
        return self


# 使用示例
if __name__ == "__main__":
    import torch
    
    data_path = "saved_data/training_data_case118_40k_preferences.npz"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {DEVICE}")
    
    print("\n" + "="*80)
    print("测试 1: OPF_Flow_Dataset_V2 (原始版本)")
    print("="*80)
    
    try:
        # 加载数据（默认20%测试集，80%训练集）
        data = OPF_Flow_Dataset_V2(data_path, device=DEVICE, test_ratio=0.2, random_seed=42)
        
        # 打印数据统计
        print("\n数据统计信息:")
        info = data.get_data_info()
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        print("\n训练集数据形状:")
        print(f"  - x_train: {data.x_train.shape}")
        print(f"  - y_train: {data.y_train.shape}")
        print(f"  - y_anchor_train: {data.y_anchor_train.shape}")
        
        print("\n测试集数据形状:")
        print(f"  - x_test: {data.x_test.shape}")
        print(f"  - y_test: {data.y_test.shape}")
        print(f"  - y_anchor_test: {data.y_anchor_test.shape}")
        
        # 获取数据的便捷方法
        x_train, y_train, y_anchor_train = data.get_train_data()
        x_test, y_test, y_anchor_test = data.get_test_data()
        
        print("\n✓ OPF_Flow_Dataset_V2 测试成功！")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
    
    print("\n" + "="*80)
    print("测试 2: OPF_Flow_Dataset_Grouped (分组版本)")
    print("="*80)
    
    try:
        # 加载分组数据集
        grouped_data = OPF_Flow_Dataset_Grouped(
            data_path, 
            device=DEVICE, 
            scenario_test_ratio=0.2, 
            random_seed=42,
            add_carbon_tax=True
        )
        
        # 打印数据统计
        print("\n数据统计信息:")
        info = grouped_data.get_data_info()
        for key, value in info.items():
            print(f"  - {key}: {value}")
        
        # 测试场景批次生成
        print("\n测试场景批次生成:")
        print("-" * 70)
        
        # 获取一个训练批次
        batch_iter = grouped_data.create_scenario_batches(batch_size=2, split='train', shuffle=False)
        x_batch, y_batch, y_anchor_batch, scenario_masks = next(batch_iter)
        
        print(f"批次数据形状:")
        print(f"  - x_batch: {x_batch.shape}")
        print(f"  - y_batch: {y_batch.shape}")
        print(f"  - y_anchor_batch: {y_anchor_batch.shape}")
        print(f"  - 场景数: {len(scenario_masks)}")
        
        # 显示每个场景的样本数
        print(f"\n每个场景的样本数:")
        for i, mask in enumerate(scenario_masks):
            n_samples = mask.sum().item()
            print(f"  - 场景 {i}: {n_samples} 个偏好样本")
        
        # 展示如何使用mask来分离场景内的数据
        print(f"\n示例：使用mask提取场景0的数据:")
        scene0_x = x_batch[scenario_masks[0]]
        scene0_y = y_batch[scenario_masks[0]]
        scene0_prefs = scene0_x[:, -1]  # 最后一列是偏好
        print(f"  - 场景0的输入形状: {scene0_x.shape}")
        print(f"  - 场景0的偏好值: {scene0_prefs.cpu().numpy()}")
        
        print("\n✓ OPF_Flow_Dataset_Grouped 测试成功！")
        print("\n💡 使用提示:")
        print("  1. 使用 create_scenario_batches() 迭代训练/测试数据")
        print("  2. 每个batch包含多个场景，每个场景有多个偏好样本")
        print("  3. 使用 scenario_masks 来区分不同场景的样本")
        print("  4. 可以基于场景内的样本设计对比学习损失函数")
        print("     例如：同一场景下，高碳税应该产生更低碳的解")
        
    except FileNotFoundError as e:
        print(f"错误: {e}")
    except Exception as e:
        print(f"运行错误: {e}")
        import traceback
        traceback.print_exc()

