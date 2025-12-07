import os

def toy_args():
    args = {'data_set': 'toy',
            'data_type': 2,
            'data_size': 10000,
            'network': 'mlp',
            'output_dim': 1,
            'latent_dim': 1,
            'num_iteration': 10000,
            'test_freq': 3000,
            'batch_dim': 1024,
            'hidden_dim': 128,
            'num_layer': 3,
            'output_act': None,
            'learning_rate': 1e-3,
            'learning_rate_decay': [1000,0.9],
            'weight_decay': 1e-6,
            'num_cluster': 3,
            'update_generator_freq': 5,
            'output_norm': False,
            'test_dim': 512,
            'time_step': 1000,
            'inf_step': 100,
            'eta':0.5,
            'ode_solver': 'Euler',
            'vae_beta': 0.01}  # VAE的KL散度权重，默认0.01可以更好地平衡重建损失和KL散度
    return args


def opt_args():
    args = toy_args()
    args['data_set'] = ['max_cut', 'min_cover'][0]
    args['graph_dim'] = 10
    args['graph_sparsity'] = [0.3,0.7]
    args['output_dim'] = args['graph_dim']
    args['data_type'] = str(args['graph_dim'])
    args['test_freq'] = 3000
    args['learning_rate'] = 1e-3
    args['output_act'] = None
    args['network'] = 'att'
    args['data_dim'] = 10000
    args['batch_dim'] = 256
    args['hidden_dim'] = 256
    args['num_layer'] = 1
    args['num_cluster'] = 4
    args['latent_dim'] = args['graph_dim']
    args['test_dim'] = 1024
    args['time_step'] = 1000
    args['inf_step'] = 5
    args['inf_sample'] = 100
    args['eta'] = 1
    args['ode_solver'] = 'Euler'
    return args


def opf_args():
    args = toy_args()
    args['data_set'] = 'acpf'
    args['graph_dim'] = 5
    args['network'] = 'mlp'
    args['data_type'] = str(args['graph_dim'])
    args['num_iteration'] = 10000
    args['test_freq'] = 1000
    args['learning_rate'] = 1e-3
    args['learning_rate_decay'] = [1000, 0.9]
    args['output_act'] = None
    args['data_dim'] = -1
    args['batch_dim'] = 512
    args['hidden_dim'] = min(args['graph_dim']*4, 2048)
    args['num_layer'] = 3
    args['latent_dim'] = args['graph_dim']
    args['test_dim'] = 1024
    args['time_step'] = 1000
    args['inf_step'] = 100
    args['inf_sample'] = 10
    args['cor_step'] = 10
    return args  


def modify_args(args):
    """
    根据模型影响训练效果的关键参数，生成唯一的实验实例名称 instance，
    并自动创建对应的模型保存目录和结果保存目录。

    主要参数通常包括：模型类型、网络结构、损失相关超参数、隐藏层大小、层数、batch大小、学习率、约束引导、碳税等。
    你可以根据具体实验需要添加或减少参数。
    
    使用缩写以避免Windows路径长度限制（MAX_PATH = 260字符）
    """
    import os

    # 可自定义整个instance名称包含哪些参数
    instance_items = []  

    # 参数名缩写映射（缩短目录名长度）
    key_abbreviations = {
        'network': 'net',
        'output_act': 'act',
        'hidden_dim': 'h',
        'num_layer': 'l',
        'batch_dim': 'b',
        'learning_rate': 'lr',
        'weight_decay': 'wd',
        'w_constraints': 'wc',
        'constraints_guided': 'cg',
        'add_carbon_tax': 'ct',
        'train_mode': 'tm',
    }

    # 影响模型训练效果的参数
    affect_keys = [ 
        'network',
        'output_act',
        'hidden_dim',
        'num_layer',
        'batch_dim',
        'learning_rate',
        'weight_decay',
        'w_constraints',   # 约束损失权重
        'constraints_guided',  # 是否约束引导
        'add_carbon_tax',    # 是否有碳税作为输入
        'train_mode',        # 训练模式（比如joint_training, separate_training）
    ]
    
    # 集合成键=值的形式，跳过None和未定义项
    for k in affect_keys:
        v = args.get(k, None)
        if v is not None:
            # 针对True/False类型做简化
            if v is True:
                v = "T"
            elif v is False:
                v = "F"
            # 训练模式缩写
            elif k == 'train_mode':
                if v == 'joint_training':
                    v = 'jt'
                elif v == 'separate_training':
                    v = 'st'
            # 使用缩写的键名
            abbr_key = key_abbreviations.get(k, k)
            instance_items.append(f"{abbr_key}{v}")
    
    # 拼接实例名称（文件夹名称禁止空格）
    instance = "_".join(instance_items)
    instance = instance.replace(" ", "")

    args['instance'] = instance

    # 自动创建文件夹
    model_dir = f"models/{instance}" 
    if not os.path.exists(model_dir):
        os.makedirs(model_dir) 

    return args



