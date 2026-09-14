"""
农业多机器人协同任务规划
MAPPO + 匈牙利任务分配 + 加权A*路径规划
20×20栅格仿真环境
Author: Github
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import os
from collections import deque, namedtuple
from queue import PriorityQueue
from scipy.optimize import linear_sum_assignment

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# ====================== 1. 加权A*路径规划 ======================
class WeightedAStar:
    def __init__(self, grid_size, obstacle_grid):
        self.grid_size = grid_size
        self.obstacle_grid = obstacle_grid
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.heuristic_weight = 1.0
        self.obstacle_penalty = 0.3
        self.path_smoothing = True

    def _obstacle_penalty(self, pos):
        x, y = pos
        penalty = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.obstacle_grid[nx, ny] == 1:
                        penalty += self.obstacle_penalty
        return penalty

    def heuristic(self, a, b):
        return self.heuristic_weight * (abs(a[0] - b[0]) + abs(a[1] - b[1]))

    def _is_straight_line(self, a, b):
        x0, y0 = a
        x1, y1 = b
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        step_x = 1 if x1 > x0 else -1
        step_y = 1 if y1 > y0 else -1
        err = dx - dy
        while x != x1 or y != y1:
            if self.obstacle_grid[x, y] == 1:
                return False
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += step_x
            if e2 < dx:
                err += dx
                y += step_y
        return True

    def smooth_path(self, path):
        if len(path) < 3 or not self.path_smoothing:
            return path
        smoothed = [path[0]]
        i = 0
        max_i = len(path) - 1
        while i < len(path):
            found = False
            for j in range(min(i + 20, max_i), i, -1):
                if self._is_straight_line(path[i], path[j]):
                    smoothed.append(path[j])
                    i = j
                    found = True
                    break
            if not found:
                i += 1
                if i <= max_i:
                    smoothed.append(path[i])
        return smoothed

    def get_path(self, start, goal):
        if (start[0] < 0 or start[0] >= self.grid_size or
                start[1] < 0 or start[1] >= self.grid_size or
                goal[0] < 0 or goal[0] >= self.grid_size or
                goal[1] < 0 or goal[1] >= self.grid_size):
            return [start]
        if self.obstacle_grid[goal[0], goal[1]] == 1:
            return [start]
        open_set = PriorityQueue()
        open_set.put((0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self.heuristic(start, goal)}

        while not open_set.empty():
            current_f, current = open_set.get()
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return self.smooth_path(path[::-1])
            for dx, dy in self.directions:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < self.grid_size and 0 <= neighbor[1] < self.grid_size:
                    if self.obstacle_grid[neighbor[0], neighbor[1]] == 1:
                        continue
                    tentative_g_score = g_score[current] + 1 + self._obstacle_penalty(neighbor)
                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = g_score[neighbor] + self.heuristic(neighbor, goal)
                        open_set.put((f_score[neighbor], neighbor))
        return [start]


# ====================== 2. 匈牙利算法任务分配 ======================
def hungarian_task_assignment(robots_pos, task_grid, obstacle_grid):
    """
    匈牙利算法实现机器人-任务分配
    task_type:1巡检任务; 2施肥任务
    """
    task_positions = []
    for x in range(task_grid.shape[0]):
        for y in range(task_grid.shape[1]):
            task_type = task_grid[x, y]
            if task_type in [1, 2]:
                task_positions.append((x, y, task_type))
    if len(task_positions) == 0:
        return [None] * len(robots_pos)

    n_robots = len(robots_pos)
    n_tasks = len(task_positions)
    cost_matrix = np.zeros((n_robots, n_tasks))

    for i, (rx, ry) in enumerate(robots_pos):
        for j, (tx, ty, t_type) in enumerate(task_positions):
            distance = abs(rx - tx) + abs(ry - ty)
            priority = 1.5 if t_type == 2 else 1
            obstacle_penalty = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = tx + dx, ty + dy
                    if 0 <= nx < obstacle_grid.shape[0] and 0 <= ny < obstacle_grid.shape[1]:
                        if obstacle_grid[nx, ny] == 1:
                            obstacle_penalty += 0.5
            cost_matrix[i, j] = (distance / priority) + obstacle_penalty

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    assigned_tasks = [None] * n_robots
    for robot_idx, task_idx in zip(row_ind, col_ind):
        if robot_idx < n_robots and task_idx < n_tasks:
            assigned_tasks[robot_idx] = task_positions[task_idx]
    return assigned_tasks


# ====================== 3. 农业机器人仿真环境 ======================
class AgriRobotEnv:
    def __init__(self, grid_size=20, num_robots=3):
        self.grid_size = grid_size
        self.num_robots = num_robots
        self.robots_pos = []
        self.task_grid = np.zeros((grid_size, grid_size))
        self.obstacle_grid = np.zeros((grid_size, grid_size))
        self.assigned_tasks = []
        self.done = False
        self.step_count = 0
        self.max_steps = 1500
        self._reset_env()

    def _reset_env(self):
        print("🔄 重置20×20环境中...")
        self.obstacle_grid = np.zeros((self.grid_size, self.grid_size))
        obstacle_num = random.randint(30, 40)
        for _ in range(obstacle_num):
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            self.obstacle_grid[x, y] = 1

        self.task_grid = np.zeros((self.grid_size, self.grid_size))
        task_num = random.randint(80, 100)
        available_positions = []
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.obstacle_grid[x, y] == 0:
                    available_positions.append((x, y))
        if len(available_positions) < task_num:
            task_num = len(available_positions)
            print(f"⚠️ 可用位置不足，任务数调整为 {task_num}")
        selected_positions = random.sample(available_positions, task_num)
        for (x, y) in selected_positions:
            self.task_grid[x, y] = random.choice([1, 2])

        self.robots_pos = []
        while len(self.robots_pos) < self.num_robots:
            x = random.randint(0, self.grid_size - 1)
            y = random.randint(0, self.grid_size - 1)
            if self.obstacle_grid[x, y] == 0 and (x, y) not in self.robots_pos:
                self.robots_pos.append((x, y))

        self.assigned_tasks = hungarian_task_assignment(self.robots_pos, self.task_grid, self.obstacle_grid)
        self.done = False
        self.step_count = 0
        self.prev_dist = [float('inf')] * self.num_robots
        self.prev_completed = 0
        print(f"✅ 环境重置完成 | 任务数: {task_num} | 障碍物数: {obstacle_num} | 机器人位置: {self.robots_pos}")
        return self._get_obs()

    def _get_obs(self):
        obs = []
        for i in range(self.num_robots):
            x, y = self.robots_pos[i]
            pos_obs = np.array([x / self.grid_size, y / self.grid_size])
            local_grid = np.zeros((3, 3, 2))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                        local_grid[dx + 1, dy + 1, 0] = self.obstacle_grid[nx, ny]
                        local_grid[dx + 1, dy + 1, 1] = self.task_grid[nx, ny] / 2
            task_obs = np.zeros(2)
            if self.assigned_tasks[i] is not None:
                tx, ty, _ = self.assigned_tasks[i]
                task_obs[0] = (tx - x) / self.grid_size
                task_obs[1] = (ty - y) / self.grid_size
            remain_inspect = np.sum(self.task_grid == 1) / 100
            remain_fertilize = np.sum(self.task_grid == 2) / 100
            progress_obs = np.array([remain_inspect, remain_fertilize])
            robot_obs = np.concatenate([pos_obs, local_grid.flatten(), task_obs, progress_obs])
            obs.append(robot_obs)
        return np.array(obs)

    def _get_global_state(self):
        robot_pos_flat = np.array(self.robots_pos).flatten() / self.grid_size
        task_grid_flat = self.task_grid.flatten() / 3
        obstacle_grid_flat = self.obstacle_grid.flatten()
        assigned_tasks_flat = []
        for task in self.assigned_tasks:
            if task is None:
                assigned_tasks_flat.extend([0, 0, 0])
            else:
                assigned_tasks_flat.extend([task[0] / self.grid_size, task[1] / self.grid_size, task[2] / 2])
        global_state = np.concatenate([robot_pos_flat, task_grid_flat, obstacle_grid_flat, assigned_tasks_flat])
        return global_state

    def _get_reward(self):
        reward = np.zeros(self.num_robots)
        global_reward = 0
        total_completed = np.sum(self.task_grid == 3)
        new_completed = total_completed - self.prev_completed
        self.prev_completed = total_completed
        global_reward += total_completed * 1.5
        global_reward += new_completed * 8

        for i in range(self.num_robots):
            x, y = self.robots_pos[i]
            if self.assigned_tasks[i] is not None:
                tx, ty, _ = self.assigned_tasks[i]
                curr_dist = abs(x - tx) + abs(y - ty)
                if self.prev_dist[i] > curr_dist:
                    reward[i] += 0.35 * (self.prev_dist[i] - curr_dist)
                elif self.prev_dist[i] < curr_dist:
                    reward[i] -= 0.12 * (curr_dist - self.prev_dist[i])
                self.prev_dist[i] = curr_dist

        for i in range(self.num_robots):
            x, y = self.robots_pos[i]
            if self.assigned_tasks[i] is not None:
                tx, ty, _ = self.assigned_tasks[i]
                dist = abs(x - tx) + abs(y - ty)
                reward[i] += max(0, 1.5 - np.tanh(dist * 0.1))

        for i in range(self.num_robots):
            x, y = self.robots_pos[i]
            if self.obstacle_grid[x, y] == 0:
                obstacle_dist = min(
                    [abs(x - ox) + abs(y - oy) for ox, oy in np.argwhere(self.obstacle_grid == 1)] + [20])
                reward[i] += 0.15 * max(0, obstacle_dist - 1)
            else:
                reward[i] -= 1.0

        for i in range(self.num_robots):
            for j in range(i + 1, self.num_robots):
                if self.robots_pos[i] == self.robots_pos[j]:
                    reward[i] -= 0.8
                    reward[j] -= 0.8

        reward += global_reward / self.num_robots
        reward -= 0.005

        if np.sum((self.task_grid == 1) | (self.task_grid == 2)) == 0:
            reward += 8
            self.done = True
        if self.step_count >= self.max_steps:
            self.done = True
            reward -= 2

        reward = np.tanh(reward / 3) * 3
        reward += 0.1
        reward = np.clip(reward, -3, 10)
        return reward

    def step(self, actions):
        self.step_count += 1
        astar_solver = WeightedAStar(self.grid_size, self.obstacle_grid)
        for i in range(self.num_robots):
            x, y = self.robots_pos[i]
            action = actions[i]
            target_pos = (x, y)
            if self.assigned_tasks[i] is not None:
                tx, ty, t_type = self.assigned_tasks[i]
                goal = (tx, ty)
                if action in [0, 1, 2, 3]:
                    path = astar_solver.get_path((x, y), goal)
                    if len(path) > 1:
                        target_pos = path[1]
                else:
                    if action == 4 and self.task_grid[x, y] == 1:
                        self.task_grid[x, y] = 3
                        self.assigned_tasks = hungarian_task_assignment(self.robots_pos, self.task_grid,
                                                                        self.obstacle_grid)
                        self.prev_dist = [float('inf')] * self.num_robots
                    elif action == 5 and self.task_grid[x, y] == 2:
                        self.task_grid[x, y] = 3
                        self.assigned_tasks = hungarian_task_assignment(self.robots_pos, self.task_grid,
                                                                        self.obstacle_grid)
                        self.prev_dist = [float('inf')] * self.num_robots

            if (0 <= target_pos[0] < self.grid_size and
                    0 <= target_pos[1] < self.grid_size and
                    self.obstacle_grid[target_pos[0], target_pos[1]] == 0):
                self.robots_pos[i] = target_pos

        obs = self._get_obs()
        global_state = self._get_global_state()
        reward = self._get_reward()
        done = self.done
        info = {"completed_tasks": np.sum(self.task_grid == 3), "assigned_tasks": self.assigned_tasks}
        return obs, global_state, reward, done, info


# ====================== 4. MAPPO 网络结构 ======================
class AttentionActor(nn.Module):
    """带多头自注意力的Actor网络"""
    def __init__(self, obs_dim, action_dim, n_agents, hidden_dim=64):
        super().__init__()
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=1, batch_first=True)
        self.dropout = nn.Dropout(0.03)
        self.policy_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs):
        obs_emb = self.obs_encoder(obs)
        attn_out, _ = self.attention(obs_emb, obs_emb, obs_emb)
        attn_out = self.dropout(attn_out)
        logits = self.policy_head(attn_out + obs_emb)
        return F.softmax(logits, dim=-1)


class Critic(nn.Module):
    """全局Critic网络，输入全局状态"""
    def __init__(self, global_state_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, global_state):
        return self.net(global_state)


# ====================== 5. MAPPO训练器 ======================
Transition = namedtuple('Transition',
                        ['obs', 'global_state', 'actions', 'rewards', 'next_obs', 'next_global_state', 'dones',
                         'log_probs'])


class MAPPOTrainer:
    def __init__(self, obs_dim, global_state_dim, action_dim, num_agents,
                 lr_actor=2e-5, lr_critic=8e-5, gamma=0.995, clip_eps=0.12,
                 batch_size=128, buffer_size=40000, update_epochs=10, save_path="./models"):
        self.num_agents = num_agents
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.batch_size = batch_size
        self.update_epochs = update_epochs
        self.save_path = save_path
        self.gae_lambda = 0.98
        os.makedirs(save_path, exist_ok=True)

        self.actor = AttentionActor(obs_dim, action_dim, num_agents)
        self.critic = Critic(global_state_dim)
        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=lr_actor, weight_decay=1e-5, eps=1e-8)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=lr_critic, weight_decay=1e-5, eps=1e-8)
        self.buffer = deque(maxlen=buffer_size)

    def store_transition(self, transition):
        self.buffer.append(transition)

    def get_action(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action_probs = self.actor(obs_tensor).squeeze(0)
        action_probs = action_probs.clamp(1e-8, 1.0 - 1e-8)
        action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
        action_dist = torch.distributions.Categorical(action_probs)
        actions = action_dist.sample()
        log_probs = action_dist.log_prob(actions)
        return actions.numpy(), log_probs.detach().numpy()

    def compute_gae(self, rewards, values, next_values, dones):
        gae = 0
        advantages = []
        rewards = np.array(rewards, dtype=np.float32)
        values = np.array(values, dtype=np.float32)
        next_values = np.array(next_values, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
            delta = np.clip(delta, -3, 3)
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        returns = np.array(advantages) + values
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std
        advantages = np.clip(advantages, -2, 2)
        returns = np.clip(returns, -30, 30)
        return advantages, returns

    def train(self):
        if len(self.buffer) < self.batch_size:
            return 0.0, 0.0
        batch = random.sample(self.buffer, self.batch_size)
        obs_batch = torch.tensor(np.array([t.obs for t in batch]), dtype=torch.float32)
        global_state_batch = torch.tensor(np.array([t.global_state for t in batch]), dtype=torch.float32)
        actions_batch = torch.tensor(np.array([t.actions for t in batch]), dtype=torch.long)
        rewards_batch = torch.tensor(np.array([t.rewards for t in batch]), dtype=torch.float32)
        next_global_state_batch = torch.tensor(np.array([t.next_global_state for t in batch]), dtype=torch.float32)
        dones_batch = torch.tensor(np.array([t.dones for t in batch]), dtype=torch.float32)
        old_log_probs_batch = torch.tensor(np.array([t.log_probs for t in batch]), dtype=torch.float32)

        values = self.critic(global_state_batch).squeeze().detach().numpy()
        next_values = self.critic(next_global_state_batch).squeeze().detach().numpy()
        rewards_mean = rewards_batch.mean(dim=1).numpy()

        advantages, returns = self.compute_gae(rewards_mean, values, next_values, dones_batch.numpy())
        advantages = torch.tensor(advantages, dtype=torch.float32)
        returns = torch.tensor(returns, dtype=torch.float32)

        actor_loss_total = 0.0
        critic_loss_total = 0.0
        for _ in range(self.update_epochs):
            action_probs = self.actor(obs_batch)
            action_probs = action_probs.clamp(1e-8, 1.0 - 1e-8)
            action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True)
            action_dist = torch.distributions.Categorical(action_probs)
            new_log_probs = action_dist.log_prob(actions_batch)

            ratio = torch.exp(new_log_probs - old_log_probs_batch)
            ratio = torch.clamp(ratio, 0.7, 1.3)
            surr1 = ratio * advantages.unsqueeze(1)
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages.unsqueeze(1)
            actor_loss = -torch.mean(torch.min(surr1, surr2))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.2)
            self.actor_optimizer.step()

            values_pred = self.critic(global_state_batch).squeeze()
            critic_loss = F.mse_loss(values_pred, returns)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.2)
            self.critic_optimizer.step()

            actor_loss_total += actor_loss.item()
            critic_loss_total += critic_loss.item()

        avg_actor_loss = actor_loss_total / self.update_epochs
        avg_critic_loss = critic_loss_total / self.update_epochs
        return avg_actor_loss, avg_critic_loss

    def save_best_model(self, episode, total_reward):
        save_dict = {
            'episode': episode,
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'best_reward': total_reward
        }
        model_path = os.path.join(self.save_path, "agri_mappo_best.pth")
        torch.save(save_dict, model_path)
        print(f"✅ 保存最优模型（奖励：{total_reward:.2f}）")

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location=torch.device('cpu'))
        self.actor.load_state_dict(ckpt['actor_state_dict'])
        self.critic.load_state_dict(ckpt['critic_state_dict'])
        print(f"✅ 加载模型完成，episode:{ckpt['episode']}, best reward:{ckpt['best_reward']:.2f}")


# ====================== 绘图工具 ======================
def plot_training_curves(episodes, total_rewards, actor_losses, critic_losses, completed_tasks, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('农业多机器人MAPPO+匈牙利算法+加权A*训练曲线（20×20栅格）', fontsize=16, fontweight='bold')
    axes[0, 0].plot(episodes, total_rewards, color='#2E86AB', linewidth=2, label='总奖励', alpha=0.7)
    window_size = 20
    if len(total_rewards) >= window_size:
        smoothed_rewards = np.convolve(total_rewards, np.ones(window_size) / window_size, mode='valid')
        smoothed_episodes = episodes[window_size - 1:]
        axes[0, 0].plot(smoothed_episodes, smoothed_rewards, color='#A23B72', linewidth=2.5, label='20轮移动平均')
    axes[0, 0].set_title('每轮总奖励', fontsize=12)
    axes[0, 0].set_xlabel('训练轮数')
    axes[0, 0].set_ylabel('总奖励')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(episodes, actor_losses, color='#F18F01', linewidth=2)
    axes[0, 1].set_title('Actor损失', fontsize=12)
    axes[0, 1].set_xlabel('训练轮数')
    axes[0, 1].set_ylabel('损失值')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(episodes, critic_losses, color='#C73E1D', linewidth=2)
    axes[1, 0].set_title('Critic损失', fontsize=12)
    axes[1, 0].set_xlabel('训练轮数')
    axes[1, 0].set_ylabel('损失值')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(episodes, completed_tasks, color='#59CD90', linewidth=2)
    axes[1, 1].set_title('每轮完成任务数', fontsize=12)
    axes[1, 1].set_xlabel('训练轮数')
    axes[1, 1].set_ylabel('完成任务数')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "training_curves_20x20.png"), dpi=200, bbox_inches='tight')
    plt.show()


# ====================== 可视化仿真器 ======================
class GridMapSimulator:
    def __init__(self, env, grid_size=20):
        self.env = env
        self.grid_size = grid_size
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.robot_colors = ['#FF3333', '#33FF33', '#3333FF']
        self.task_colors = {1: '#FFCC00', 2: '#FF6600', 3: '#00CC00'}
        self.astar = WeightedAStar(grid_size, env.obstacle_grid)

    def render(self, paths=None, episode=0, step=0, total_reward=0):
        self.ax.clear()
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(range(self.grid_size))
        self.ax.set_yticks(range(self.grid_size))
        self.ax.grid(True, color='black', linewidth=0.3)
        self.ax.set_title(
            f'农业多机器人栅格地图仿真（20×20）| Episode:{episode} Step:{step} | Total Reward:{total_reward:.1f}',
            fontsize=12, fontweight='bold')

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.env.obstacle_grid[x, y] == 1:
                    self.ax.fill_between([y - 0.5, y + 0.5], [x - 0.5, x - 0.5], [x + 0.5, x + 0.5], color='black')

        for x in range(self.grid_size):
            for y in range(self.grid_size):
                t = self.env.task_grid[x, y]
                if t in self.task_colors:
                    self.ax.fill_between([y - 0.5, y + 0.5], [x - 0.5, x - 0.5], [x + 0.5, x + 0.5],
                                         color=self.task_colors[t], alpha=0.6)
        if paths is not None:
            for i, path in enumerate(paths):
                if path and len(path) > 1:
                    path_x = [p[0] for p in path]
                    path_y = [p[1] for p in path]
                    self.ax.plot(path_y, path_x, color=self.robot_colors[i], linewidth=2, alpha=0.6, linestyle='--')

        for i, (x, y) in enumerate(self.env.robots_pos):
            self.ax.scatter(y, x, s=200, c=self.robot_colors[i], edgecolors='white', linewidth=2, zorder=5)

        for i, task in enumerate(self.env.assigned_tasks):
            if task is not None:
                rx, ry = self.env.robots_pos[i]
                tx, ty, _ = task
                self.ax.plot([ry, ty], [rx, tx], color=self.robot_colors[i], linewidth=1.5, alpha=0.8)
        plt.pause(0.05)

    def close(self):
        plt.close()


def test_grid_simulation(env, trainer, max_steps=1000):
    simulator = GridMapSimulator(env, env.grid_size)
    obs = env._reset_env()
    total_reward = 0
    step = 0
    done = False
    print("\n🚀 开始20×20栅格地图仿真演示...")
    while not done and step < max_steps:
        actions, _ = trainer.get_action(obs)
        paths = []
        for i in range(env.num_robots):
            rp = env.robots_pos[i]
            t = env.assigned_tasks[i]
            if t:
                p = simulator.astar.get_path(rp, (t[0], t[1]))
                paths.append(p)
            else:
                paths.append([])
        next_obs, _, reward, done, info = env.step(actions)
        total_reward += np.sum(reward)
        obs = next_obs
        step += 1
        simulator.render(paths=paths, episode=0, step=step, total_reward=total_reward)
    simulator.close()
    print(f"✅ 仿真完成！总奖励：{total_reward:.2f} 完成任务：{info['completed_tasks']}")


# ====================== 主入口 ======================
if __name__ == "__main__":
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    GRID_SIZE = 20
    NUM_ROBOTS = 3
    OBS_DIM = 24
    GLOBAL_STATE_DIM = NUM_ROBOTS * 2 + GRID_SIZE * GRID_SIZE + GRID_SIZE * GRID_SIZE + NUM_ROBOTS * 3
    N_ACTIONS = 6
    SAVE_PATH = "./agri_mappo_result_20x20"
    os.makedirs(SAVE_PATH, exist_ok=True)

    env = AgriRobotEnv(grid_size=GRID_SIZE, num_robots=NUM_ROBOTS)
    trainer = MAPPOTrainer(
        obs_dim=OBS_DIM,
        global_state_dim=GLOBAL_STATE_DIM,
        action_dim=N_ACTIONS,
        num_agents=NUM_ROBOTS,
        save_path=SAVE_PATH
    )

    EPISODES = 800
    EPS_START = 0.6
    EPS_END = 0.02
    best_reward = -np.inf
    log_episodes = []
    log_total_rewards = []
    log_actor_losses = []
    log_critic_losses = []
    log_completed_tasks = []

    print("=" * 80)
    print("开始训练：MAPPO + 匈牙利分配 + 加权A* | 20×20栅格 | 任务数量 80~100")
    print(f"总轮数：{EPISODES} | 机器人：{NUM_ROBOTS} | 网格：{GRID_SIZE}×{GRID_SIZE}")
    print("=" * 80)

    for episode in range(EPISODES):
        obs = env._reset_env()
        global_state = env._get_global_state()
        total_reward = 0
        eps = EPS_END + (EPS_START - EPS_END) * 0.5 * (1 + np.cos(episode / EPISODES * np.pi))
        while not env.done:
            if random.random() > eps:
                actions, log_probs = trainer.get_action(obs)
            else:
                actions = np.random.randint(0, N_ACTIONS, size=NUM_ROBOTS)
                log_probs = np.zeros(NUM_ROBOTS)
            next_obs, next_global_state, reward, done, info = env.step(actions)
            total_reward += np.sum(reward)
            trainer.store_transition(
                Transition(obs, global_state, actions, reward, next_obs, next_global_state, done, log_probs)
            )
            actor_loss, critic_loss = trainer.train()
            obs = next_obs
            global_state = next_global_state

        log_episodes.append(episode + 1)
        log_total_rewards.append(total_reward)
        log_actor_losses.append(actor_loss)
        log_critic_losses.append(critic_loss)
        log_completed_tasks.append(info['completed_tasks'])

        if len(log_total_rewards) >= 15:
            recent = np.mean(log_total_rewards[-15:])
            if recent > best_reward:
                best_reward = recent
                trainer.save_best_model(episode + 1, recent)
        else:
            if total_reward > best_reward:
                best_reward = total_reward
                trainer.save_best_model(episode + 1, total_reward)

        if (episode + 1) % 40 == 0:
            avg_r = np.mean(log_total_rewards[-40:])
            avg_al = np.mean(log_actor_losses[-40:])
            avg_cl = np.mean(log_critic_losses[-40:])
            avg_cmp = np.mean(log_completed_tasks[-40:])
            print(f"Episode {episode + 1}/{EPISODES} | Avg Reward: {avg_r:.2f} | "
                  f"Actor Loss: {avg_al:.4f} | Critic Loss: {avg_cl:.4f} | "
                  f"Avg Completed: {avg_cmp:.1f} | Best Reward: {best_reward:.2f} | EPS: {eps:.4f}")

    plot_training_curves(log_episodes, log_total_rewards, log_actor_losses, log_critic_losses, log_completed_tasks,
                         SAVE_PATH)

    # 取消下面注释即可加载最优模型运行可视化仿真
    # checkpoint = torch.load(os.path.join(SAVE_PATH, "agri_mappo_best.pth"))
    # trainer.actor.load_state_dict(checkpoint['actor_state_dict'])
    # trainer.critic.load_state_dict(checkpoint['critic_state_dict'])
    # test_grid_simulation(env, trainer)
