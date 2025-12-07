"""
生成监督数据并对电压相角预测模型进行监督预训练
每个场景都需要给定不同的碳税率，生成不同的监督数据
"""

# 预训练模型
import pandapower as pp 
from torch.utils.data import Dataset
from env import PowerGridEnv
import torch
import numpy as np 
import os
import copy 
from add_objective.carbon_tax_utils import update_cost_coefficients_with_carbon_tax 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
loss_func = torch.nn.MSELoss()


# 训练函数
def train_epoch(model, train_loader, loss_func, optimizer, max_grad_norm=1.0):
    model.train()  
    total_loss = 0
    total_vm_loss = 0
    total_va_loss = 0
    # total_cons_loss = 0
    # total_g1, total_g2, total_g3, total_g4 = 0, 0, 0, 0
    # total_g5, total_g6, total_g9, total_g10 = 0, 0, 0, 0
    total_grad_norm = 0
    num_batches = 0
    
    for batch_inputs, batch_targets in train_loader:
        # 将数据移动到CUDA设备
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        
        Vm, Va = model(batch_inputs) 

        # 使用模型的方法计算约束损失
        # loss_cons, constraint_details = model.compute_constraint_loss(Vm, Va, batch_inputs, env, return_details=True)

        trainyvm = batch_targets[:, :env.num_buses]
        trainyva = batch_targets[:, env.num_buses:]

        vm_loss = loss_func(Vm, trainyvm)
        va_loss = loss_func(Va, trainyva)
        loss = vm_loss + va_loss # + loss_cons / 100   # 不使用约束损失
        
        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪，防止梯度爆炸
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        optimizer.step() 
        
        # 累积损失和梯度范数
        total_loss += loss.item()
        total_vm_loss += vm_loss.item()
        total_va_loss += va_loss.item()
        # total_cons_loss += loss_cons.item()
        total_grad_norm += grad_norm.item()
        # total_g1 += constraint_details['g1_pmax']
        # total_g2 += constraint_details['g2_pmin']
        # total_g3 += constraint_details['g3_ramp_up']
        # total_g4 += constraint_details['g4_ramp_down']
        # total_g5 += constraint_details['g5_qmax']
        # total_g6 += constraint_details['g6_qmin']
        # total_g9 += constraint_details['g9_sf']
        # total_g10 += constraint_details['g10_st']
        num_batches += 1
    
    # 返回平均损失
    return {
        'total_loss': total_loss,
        'vm_loss': total_vm_loss / num_batches,
        'va_loss': total_va_loss / num_batches,
        # 'cons_loss': total_cons_loss / num_batches,
        'grad_norm': total_grad_norm / num_batches,
        # 'g1_pmax': total_g1 / num_batches,
        # 'g2_pmin': total_g2 / num_batches,
        # 'g3_ramp_up': total_g3 / num_batches,
        # 'g4_ramp_down': total_g4 / num_batches,
        # 'g5_qmax': total_g5 / num_batches,
        # 'g6_qmin': total_g6 / num_batches,
        # 'g9_sf': total_g9 / num_batches,
        # 'g10_st': total_g10 / num_batches,
    }

# 测试函数
def evaluate(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    total_vm_loss = 0
    total_va_loss = 0
    total_cons_loss = 0
    total_g1, total_g2, total_g3, total_g4 = 0, 0, 0, 0
    total_g5, total_g6, total_g9, total_g10 = 0, 0, 0, 0
    num_batches = 0
    
    with torch.no_grad():
        for batch_inputs, batch_targets in test_loader:
            # 将数据移动到CUDA设备
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            
            # 检查输入是否有NaN或Inf
            if torch.isnan(batch_inputs).any() or torch.isinf(batch_inputs).any():
                print(f"警告: 测试集输入数据包含 NaN 或 Inf!")
                continue
            
            # 使用与train_epoch相同的损失计算方式
            Vm, Va = model(batch_inputs)
            
            # 检查输出是否有NaN
            if torch.isnan(Vm).any() or torch.isnan(Va).any():
                print(f"警告: 测试集模型输出包含 NaN!")
                continue
            
            # 使用模型的方法计算约束损失
            # loss_cons, constraint_details = model.compute_constraint_loss(Vm, Va, batch_inputs, env, return_details=True)

            # 计算与train_epoch相同的总损失
            trainyvm = batch_targets[:, :env.num_buses]
            trainyva = batch_targets[:, env.num_buses:]
            vm_loss = criterion(Vm, trainyvm)
            va_loss = criterion(Va, trainyva)
            loss = vm_loss + va_loss # + loss_cons / 100
            
            # 累积损失
            total_loss += loss.item()
            total_vm_loss += vm_loss.item()
            total_va_loss += va_loss.item()
            # total_cons_loss += loss_cons.item()
            # total_g1 += constraint_details['g1_pmax']
            # total_g2 += constraint_details['g2_pmin']
            # total_g3 += constraint_details['g3_ramp_up']
            # total_g4 += constraint_details['g4_ramp_down']
            # total_g5 += constraint_details['g5_qmax']
            # total_g6 += constraint_details['g6_qmin']
            # total_g9 += constraint_details['g9_sf']
            # total_g10 += constraint_details['g10_st']
            num_batches += 1
    
    # 返回平均损失
    return {
        'total_loss': total_loss,
        'vm_loss': total_vm_loss / num_batches,
        'va_loss': total_va_loss / num_batches,
        # 'cons_loss': total_cons_loss / num_batches,
        # 'g1_pmax': total_g1 / num_batches,
        # 'g2_pmin': total_g2 / num_batches,
        # 'g3_ramp_up': total_g3 / num_batches,
        # 'g4_ramp_down': total_g4 / num_batches,
        # 'g5_qmax': total_g5 / num_batches,
        # 'g6_qmin': total_g6 / num_batches,
        # 'g9_sf': total_g9 / num_batches,
        # 'g10_st': total_g10 / num_batches,
    }


# 创建数据集类
class PowerGridDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = torch.FloatTensor(inputs)
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

if __name__ == "__main__":

    #################################### 相关参数配置 ############################################ 
    import os 
    from tqdm import tqdm
    from flow_model.models.actor import Actor, PowerSystemConfig
    ps_config = PowerSystemConfig(device=device, case_file_path='./saved_data/pglib_opf_case118.mat')
    # ps_config = None

    # 创建保存模型和图片的文件夹
    os.makedirs('saved_model', exist_ok=True)
    os.makedirs('training_plots', exist_ok=True)
    os.makedirs('saved_data', exist_ok=True) 

    load_data = False                                                                                       # pay attention to this parameter                 
    case_name = "case118" 
    debug_mode = False      

    data_save_path = f'saved_data/training_data_{case_name}_50_preferences.npz' 

    num_timesteps = 288 * (10000//288+1) if not debug_mode else 288 * 10  # 174天的数据  40000
    # num_timesteps = 288
    random_load = True    # 意味着大量的随机场景 false 意味着生成和强化学习场景一致的场景
    run_pp = True    # 和是否计算奖励有关系
    consider_renewable_generation = False                                                                      # pay attention to this parameter
    carbon_taxs = [0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0]  # 碳税率 ($/tCO2)，设置为0表示不考虑碳成本，可设置为 10, 20, 30 等不同值                   
    # carbon_taxs = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0]  # 碳税率 ($/tCO2)，设置为0表示不考虑碳成本，可设置为 10, 20, 30 等不同值                   
    
    env = PowerGridEnv(num_timesteps=num_timesteps, case_name=case_name, random_load=random_load, run_pp=run_pp,
                       consider_renewable_generation=consider_renewable_generation, PowerSystemConfig=ps_config,
                       device=device, carbon_tax=0.0)

    obs_dim = env.observation_space.shape[0] 
 
    # 创建模型实例
    model = Actor(input_dim=obs_dim, env=env, output_dim=env.num_buses).to(device)

    # 加载模型 
    train = True 
    batch_size = 50   # 根据deepopf-v里的描述
    lr = 1e-3     # learning rate origin: 1e-4

    ################################ 收集数据 ####################################################
    if not load_data:
        # 收集训练数据
        print("正在收集训练数据...")
        train_inputs = []
        train_targets = []
        y_anchors = []
        preferences = []
        actions = []

        # 收集多个episode的数据
        num_episodes = 1
        steps_per_episode = num_timesteps  # 24小时 * 12个5分钟间隔
        
        # 创建总进度条
        total_steps = num_episodes * steps_per_episode
        progress_bar = tqdm(total=total_steps, desc="数据收集进度")

        for episode in range(num_episodes):
            obs = env.reset()   # has been normalized by /100 
            last_valid_gen = env.net.gen.p_mw.values.copy()

            while not env.done: 
                try:   
                    # 保存当前负荷场景的初始网络状态（在运行任何OPF之前）
                    net_snapshot = copy.deepcopy(env.net)
                    
                    # 用于存储当前时间步不同碳税下的action，只用第一个action进行env.step
                    actions_for_timestep = []
                    
                    # 初始化 y_anchor（避免变量未定义的问题）
                    y_anchor = None
                    
                    # 修改目标函数，将碳排放量加入到最优潮流的目标函数里 
                    for idx, carbon_tax in enumerate(carbon_taxs):
                        # 在每次OPF之前，恢复到初始网络状态（确保独立性）
                        if idx > 0:
                            env.net = copy.deepcopy(net_snapshot)
                        
                        env.carbon_tax = carbon_tax 
                        update_cost_coefficients_with_carbon_tax(env)
                        
                        pp.runopp(env.net, verbose=False)  # 运行最优潮流
                        
                        # 保存成功的gen状态
                        last_valid_gen = env.net.gen.p_mw.values.copy()
                        
                        # 收集目标值(最优电压和相角)
                        vm_pu = env.net.res_bus.vm_pu.values
                        va_deg = env.net.res_bus.va_degree.values

                        vm_normalized = (vm_pu - 1) / 0.06  # 归一化到 tanh 范围
                        va_normalized = va_deg / 30         # 归一化到 tanh 范围
                        target = np.concatenate([vm_normalized, va_normalized])

                        # debug 测试下收集到的数据是否满足约束
                        # with torch.no_grad():
                        #     vm_normalized_tensor = torch.tensor(vm_normalized, dtype=torch.float32).to(device).unsqueeze(0)
                        #     va_normalized_tensor = torch.tensor(va_normalized, dtype=torch.float32).to(device).unsqueeze(0)
                        #     obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
                        #     violations = model.compute_constraint_loss(
                        #         vm_normalized_tensor, va_normalized_tensor, obs_tensor, env, reduction='none'
                        #     )
                        #     print(f" violations: {violations}")
                        
                        # 第一个碳税的结果作为anchor（参考点）
                        if idx == 0:
                            y_anchor = copy.deepcopy(target)

                        # 生成action  
                        vm_pu_act = vm_pu[env.gen_bus_idx]
                        res_gen_act = env.net.res_gen.p_mw.values[env.Pg_idx]
                        action = np.concatenate([vm_pu_act, res_gen_act]) # p.u. and mw
                        actions_for_timestep.append(action)
                        
                        train_inputs.append(obs)                # c (负荷场景)
                        preferences.append(carbon_tax)         # lambda_i (碳税偏好)
                        # 这块我想做个验证，有没有可能说我们模型根据当前的电压和相角，计算action这个最后计算出来的和实际的对不上
                        # vm_normalized_tensor = torch.tensor(vm_normalized, dtype=torch.float32).to(device).unsqueeze(0)
                        # va_normalized_tensor = torch.tensor(va_normalized, dtype=torch.float32).to(device).unsqueeze(0)
                        # obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device).unsqueeze(0)
                        # action_tensor = model.test(obs_tensor, vm_normalized_tensor, va_normalized_tensor)
                        # action_tensor = action_tensor.cpu().numpy().squeeze()
                        # print(f"action_tensor: {action_tensor}")
                        # print(f"action: {action}")
                        actions.append(action)                  # P_g, V_g (发电机输出)
                        train_targets.append(target)            # Vm, Va (电压和相角目标)
                        y_anchors.append(y_anchor)              # Vm_anchor, Va_anchor (参考点)

                    # 临时保存：
                    if len(train_inputs) > 1000:
                        np.savez(data_save_path + f'_{env.current_step}.npz',
                                train_inputs=train_inputs,
                                train_targets=train_targets,
                                preferences=preferences,
                                y_anchors=y_anchors,
                                actions=actions)
                        print(f"训练数据已保存至: {data_save_path + f'_{env.current_step}.npz'}")
                        train_inputs = []
                        train_targets = []
                        y_anchors = []
                        preferences = []
                        actions = []
                    # 使用第一个碳税对应的action来更新环境（或者可以选择使用某个特定的碳税）
                    # 注意：这里需要恢复到第一个碳税的网络状态
                    env.net = copy.deepcopy(net_snapshot) 
                    
                    # 使用第一个碳税的action进行step
                    obs, _, _, _ = env.step(actions_for_timestep[0])   

                except Exception as e:
                    print(f"报错，当前场景是不可行的，跳过: {e}")
                    
                    # 恢复上一次成功的res_gen状态，避免污染后续计算
                    if last_valid_gen is not None:
                        env.net.gen['p_mw'] = last_valid_gen
                    
                    try:
                        pp.runpp(env.net, init='flat', calculate_voltage_angles=True)
                    except Exception as pp_e:
                        print(f"runpp清理时失败(flat init): {pp_e}")
                        # 如果flat init也失败，尝试完全重新初始化
                        try:
                            pp.runpp(env.net, calculate_voltage_angles=True)
                        except Exception as pp_e2:
                            print(f"runpp清理最终失败: {pp_e2}")
                            break
                    
                    # 跳过当前时间步    
                    if env.current_step != num_timesteps - 1:
                        env.current_step += 1   # 更新时间步 
                        env.update_load_profiles()    # 更新当前时间步的负荷值 
                        obs = env._get_observation()   # step+1的负荷值
                    else:
                        env.done = True
                    continue
                finally:
                    # 无论成功失败，都更新进度条
                    progress_bar.update(1)
                
        print(f"收集到 {len(train_inputs)} 个有效数据点")
        print(f"  - 负荷场景数: {len(train_inputs)}")
        print(f"  - 碳税偏好数: {len(set(preferences))}")
        print(f"  - 每个场景的偏好数: {len(carbon_taxs)}")

        # 保存训练数据（包括所有收集的信息）
        np.savez(data_save_path + f'_{env.current_step}.npz',
        train_inputs=train_inputs,
        train_targets=train_targets,
        preferences=preferences,
        y_anchors=y_anchors,
        actions=actions)
        print(f"训练数据已保存至: {data_save_path + f'_{env.current_step}.npz'}")
    else:
        # 加载训练数据
        print(f"正在加载训练数据: {data_save_path}")
        data = np.load(data_save_path)
        train_inputs = data['train_inputs']
        train_targets = data['train_targets']
        
        # 尝试加载额外的数据（如果存在）
        if 'preferences' in data:
            preferences = data['preferences']
            print(f"  - 加载了偏好数据")
        if 'y_anchors' in data:
            y_anchors = data['y_anchors']
            print(f"  - 加载了anchor数据")
        if 'actions' in data:
            actions = data['actions']
            print(f"  - 加载了action数据")
    #################################数据预处理###############################################

    # # train_inputs = train_inputs[:50000]
    # # train_targets = train_targets[:50000]

    # train_inputs = np.array(train_inputs)   # P_d and Q_d   
    # train_targets = np.array(train_targets) # Vm and Va

    # # 划分训练集和测试集
    # X_train, X_test, y_train, y_test = train_test_split(
    #     train_inputs, train_targets, test_size=0.2, random_state=42
    # )

    # # 创建训练集和测试集的数据加载器
    # train_dataset = PowerGridDataset(X_train, y_train)
    # test_dataset = PowerGridDataset(X_test, y_test)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    # criterion = nn.MSELoss()
    # #################################模型训练###############################################
    # # 创建保存图表的文件夹
    # plot_dir = f'training_plots'
    # os.makedirs(plot_dir, exist_ok=True)
    # current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # if train:
    #     # 创建TensorBoard writer
    #     log_dir = os.path.join('runs', f'pretrain_{case_name}_{current_time}_50k')
    #     writer = SummaryWriter(log_dir)
    #     print(f"TensorBoard日志保存至: {log_dir}")
    #     print(f"启动TensorBoard命令: tensorboard --logdir=runs")
        
    #     optimizer = optim.Adam(model.parameters(), lr=lr)
    #     # 训练和评估
    #     model_save_path = os.path.join('saved_model', f'best_model_{current_time}.pth')
    #     print("开始训练...")
    #     num_epochs = 1000
    #     best_test_loss = float('inf')
    #     max_grad_norm = 1.0  # 梯度裁剪阈值，防止梯度爆炸
        
    #     print(f"训练配置:")
    #     print(f"  - 学习率: {optimizer.param_groups[0]['lr']}")
    #     print(f"  - 梯度裁剪阈值: {max_grad_norm}")
    #     print(f"  - 训练轮数: {num_epochs}")
    #     print(f"  - 批次大小: {batch_size}")
        
    #     # 创建记录训练过程的列表
    #     train_losses = []
    #     test_losses = []
                                
    #     for epoch in tqdm(range(num_epochs), desc="训练进度"):
    #         # 训练
    #         train_metrics = train_epoch(model, train_loader, criterion, optimizer, max_grad_norm)
            
    #         # 测试
    #         test_metrics = evaluate(model, test_loader, criterion)
            
    #         # 记录总损失到列表（用于后续绘图）
    #         train_losses.append(train_metrics['total_loss'])
    #         test_losses.append(test_metrics['total_loss'])
            
    #         # 记录到TensorBoard - 总损失
    #         writer.add_scalar('Loss/train_total', train_metrics['total_loss'], epoch)
    #         writer.add_scalar('Loss/test_total', test_metrics['total_loss'], epoch)
            
    #         # 记录到TensorBoard - 损失分量
    #         writer.add_scalar('Loss/train_vm', train_metrics['vm_loss'], epoch)
    #         writer.add_scalar('Loss/test_vm', test_metrics['vm_loss'], epoch)
    #         writer.add_scalar('Loss/train_va', train_metrics['va_loss'], epoch)
    #         writer.add_scalar('Loss/test_va', test_metrics['va_loss'], epoch)
    #         # writer.add_scalar('Loss/train_constraint', train_metrics['cons_loss'], epoch)
    #         # writer.add_scalar('Loss/test_constraint', test_metrics['cons_loss'], epoch)
            
    #         # 记录到TensorBoard - 各个约束违反项（训练集）  
    #         # writer.add_scalar('Constraint_Train/g1_Pg_max', train_metrics['g1_pmax'], epoch)
    #         # writer.add_scalar('Constraint_Train/g2_Pg_min', train_metrics['g2_pmin'], epoch)
    #         # writer.add_scalar('Constraint_Train/g3_ramp_up', train_metrics['g3_ramp_up'], epoch)
    #         # writer.add_scalar('Constraint_Train/g4_ramp_down', train_metrics['g4_ramp_down'], epoch)
    #         # writer.add_scalar('Constraint_Train/g5_Qg_max', train_metrics['g5_qmax'], epoch)
    #         # writer.add_scalar('Constraint_Train/g6_Qg_min', train_metrics['g6_qmin'], epoch)
    #         # writer.add_scalar('Constraint_Train/g9_line_from', train_metrics['g9_sf'], epoch)
    #         # writer.add_scalar('Constraint_Train/g10_line_to', train_metrics['g10_st'], epoch)
            
    #         # 记录到TensorBoard - 各个约束违反项（测试集）
    #         # writer.add_scalar('Constraint_Test/g1_Pg_max', test_metrics['g1_pmax'], epoch)
    #         # writer.add_scalar('Constraint_Test/g2_Pg_min', test_metrics['g2_pmin'], epoch)
    #         # writer.add_scalar('Constraint_Test/g3_ramp_up', test_metrics['g3_ramp_up'], epoch)
    #         # writer.add_scalar('Constraint_Test/g4_ramp_down', test_metrics['g4_ramp_down'], epoch)
    #         # writer.add_scalar('Constraint_Test/g5_Qg_max', test_metrics['g5_qmax'], epoch)
    #         # writer.add_scalar('Constraint_Test/g6_Qg_min', test_metrics['g6_qmin'], epoch)
    #         # writer.add_scalar('Constraint_Test/g9_line_from', test_metrics['g9_sf'], epoch)
    #         # writer.add_scalar('Constraint_Test/g10_line_to', test_metrics['g10_st'], epoch)
            
    #         # 记录学习率和梯度范数
    #         current_lr = optimizer.param_groups[0]['lr']
    #         writer.add_scalar('Hyperparameters/learning_rate', current_lr, epoch)
    #         writer.add_scalar('Hyperparameters/gradient_norm', train_metrics['grad_norm'], epoch)
            
    #         if test_metrics['total_loss'] < best_test_loss:
    #             best_test_loss = test_metrics['total_loss']
    #             torch.save(model.state_dict(), model_save_path)
    #             print(f"保存最佳模型至: {model_save_path}")
            
    #         # if (epoch + 1) % 10 == 0:
    #         #     print(f"Epoch [{epoch+1}/{num_epochs}]")
    #         #     print(f"训练损失: {train_metrics['total_loss']:.4f} (Vm: {train_metrics['vm_loss']:.4f}, Va: {train_metrics['va_loss']:.4f}, Cons: {train_metrics['cons_loss']:.4f}, GradNorm: {train_metrics['grad_norm']:.4f})")
    #         #     print(f"测试损失: {test_metrics['total_loss']:.4f} (Vm: {test_metrics['vm_loss']:.4f}, Va: {test_metrics['va_loss']:.4f}, Cons: {test_metrics['cons_loss']:.4f})")

    #     print("训练完成!")
        
    #     # 关闭TensorBoard writer
    #     writer.close()
    #     print(f"TensorBoard日志已保存")
        
    #     # 绘制损失曲线 
    #     plt.figure(figsize=(12, 8))
    #     epochs = range(1, num_epochs + 1)
    #     plt.plot(epochs, train_losses, 'b-', label='Train Total Loss')
    #     plt.plot(epochs, test_losses, 'r-', label='Test Total Loss')
    #     plt.title('Loss Changes During Training')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.legend()
    #     plt.savefig(f'{plot_dir}/total_loss_{current_time}.png')
    #     plt.close()
    # ################################# 加载最佳模型进行最终测试 ###############################################
    # model.load_state_dict(torch.load(model_save_path))
    # final_test_metrics = evaluate(model, test_loader, criterion)
    # print("\n最终测试结果:")
    # print(f"总损失: {final_test_metrics['total_loss']:.4f}")
    # print(f"电压损失: {final_test_metrics['vm_loss']:.4f}")
    # print(f"相角损失: {final_test_metrics['va_loss']:.4f}")
    # # print(f"约束损失: {final_test_metrics['cons_loss']:.4f}")

    # # 打印保存的模型名称
    # print(f"\n保存的模型文件名: {model_save_path}")
    # print(f"模型保存路径: {os.path.abspath(model_save_path)}")