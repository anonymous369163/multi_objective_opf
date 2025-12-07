from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import pandapower as pp 

def get_runopp_action(env):
    """
    使用pandapower的最优潮流求解器获得动作，返回action
    action构成: [V_gen, P_pg] = [所有发电机电压(p.u.), 可调节发电机功率(MW)]
    """
    pp.runopp(env.net, verbose=False)  # 运行最优潮流
    vm_pu = env.net.res_bus.vm_pu.values 
    V_gen = vm_pu[env.gen_bus_idx]
    P_pg = env.net.res_gen.p_mw.values[env.Pg_idx.values]
    action = np.concatenate([V_gen, P_pg])
    return action



class ReplayBuffer:
    def __init__(self, capacity):
        """
        优化的经验回放缓冲区，使用预分配的numpy数组提高性能
        """
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.position = 0
        self.size_counter = 0
        
        # 预分配内存用于批量转换 (可选优化)
        self._batch_states = None
        self._batch_next_states = None

    def add(self, state, action, reward, next_state, done): 
        """添加经验到缓冲区"""
        self.buffer.append((state, action, reward, next_state, done))
        self.size_counter = min(self.size_counter + 1, self.capacity)

    def sample(self, batch_size): 
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 

    def size(self): 
        return len(self.buffer)

def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size-1, 2)
    begin = np.cumsum(a[:window_size-1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def train_on_policy_agent(env, agent, num_episodes):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                agent.update(transition_dict)
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes/10 * i + i_episode+1), 'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    return return_list

def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, writer=None):
    return_list = []
    global_step = 0
    for i in range(10):
        with tqdm(total=int(num_episodes/10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes/10)):
                episode_return = 0
                episode_steps = 0
                episode_losses = {'actor_loss': 0.0, 'critic_loss': 0.0, 'constraint_loss': 0.0}
                episode_loss_count = 0
                
                state = env.reset()
                while not env.done:
                    # 智能体选择动作
                    action = agent.take_action(state)                     
                    # 在环境中执行动作
                    next_state, reward, done, info = env.step(action)
                    global_step += 1
                    episode_steps += 1
                    
                    # 记录每步奖励到TensorBoard
                    if writer is not None:
                        writer.add_scalar('Reward/step', reward, global_step)
                    
                    # 存储转移到经验回放缓冲区
                    replay_buffer.add(state, action, reward, next_state, done)  # action里是电压和有功功率，
                    
                    # 更新状态和累计奖励
                    state = next_state
                    episode_return += reward
                    
                    # 当经验回放缓冲区中的样本数量足够时，进行学习
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r, 'dones': b_d}
                        loss = agent.update(transition_dict)
                        
                        # 累计每个episode的平均损失值
                        for key in loss:
                            episode_losses[key] += loss[key]
                        episode_loss_count += 1
                
                # 记录当前episode的总回报和平均损失
                return_list.append(episode_return)
                
                # 计算每个episode的平均损失
                if episode_loss_count > 0:
                    for key in episode_losses:
                        episode_losses[key] /= episode_loss_count
                
                # 记录episode级别的指标到TensorBoard
                if writer is not None:
                    episode_idx = i * int(num_episodes/10) + i_episode
                    writer.add_scalar('Reward/episode', episode_return, episode_idx)
                    writer.add_scalar('Reward/avg_per_step', episode_return / max(1, episode_steps), episode_idx)
                    
                    # 记录每个episode的平均损失
                    for key in episode_losses:
                        writer.add_scalar(f'Loss/episode_{key}', episode_losses[key], episode_idx)
                
                # 更新进度条
                if (i_episode+1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes/10 * i + i_episode+1), 
                        'return': '%.3f' % np.mean(return_list[-10:]),
                        'buffer_size': '%d' % replay_buffer.size()
                    })
                pbar.update(1)
    
    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)
                