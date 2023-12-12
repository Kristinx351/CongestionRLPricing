# only train agentQ

import torch
import torch.nn.functional as F
from torch.optim import Adam
from .model import AgentQNet, LocalPNet
from .utils import soft_update, hard_update
from .generate_graph import generate_data 
import logging
import os
import numpy as np


class AgentWise:
    def __init__(self, action_space, num_agent, world, args):
        self.no_train_idx = world.no_train_idx
        self.train_idx = world.train_idx
        self.action_space = action_space
        self.edge_index = world.edge_index
        self.world = world
        self.num_agent = num_agent
        self.graph_num_agent = args.batch_size * num_agent
        self.num_train = len(self.train_idx)
        self.gamma = args.gamma
        self.alpha = args.alpha
        self.target_update_interval = args.target_update_interval
        self.tau = args.tau
        self.local_critic = AgentQNet()
        self.local_critic_optim = Adam(self.local_critic.parameters(), lr=args.lr)
        self.local_critic_target = AgentQNet()
        self.policy = LocalPNet()
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        hard_update(self.local_critic_target, self.local_critic)
        
         
    def select_action(self, state):
        state = torch.FloatTensor(state)
        action, _ = self.policy.get_log_prob(state, self.edge_index, with_logprob=False)
        action = action.detach().numpy()
        action[self.no_train_idx] = 0
        return action
    
    def update_parameters(self, memory, batch_size, updates):
        batch = memory.sample(batch_size=batch_size)
        state_batch = batch['state']
        next_state_batch = batch['next_state']
        action_batch = batch['action']
        reward_batch = batch['reward']
        done_batch = batch['done']

        state_tensor = torch.FloatTensor(state_batch).view(self.graph_num_agent, -1)
        next_state_tensor = torch.FloatTensor(next_state_batch).view(self.graph_num_agent, -1)
        action_tensor = torch.FloatTensor(action_batch).view(self.graph_num_agent, -1)
        reward_tensor = torch.FloatTensor(reward_batch).view(self.graph_num_agent, -1)
        mask_tensor = 1 - torch.FloatTensor(done_batch).view(self.graph_num_agent, -1)

        # 图生成的a'
        with torch.no_grad():
            # compute Q'
            next_action, next_log_pi = self.policy.get_log_prob(next_state_tensor, self.large_edge_index)  # shape:(batch_size*num_agent, 1)
            next_agent_Q1, next_agent_Q2 = self.local_critic_target(next_state_tensor, next_action)
            min_agentq_next_target = torch.min(next_agent_Q1, next_agent_Q2) - self.alpha * next_log_pi
            next_agentq_value = reward_tensor + self.gamma * mask_tensor * min_agentq_next_target
        # compute Q
        agent_qf1, agent_qf2 = self.local_critic(state_tensor, action_tensor)
        # 更新localQ参数
        agent_qf1_loss = F.mse_loss(
            agent_qf1.view(-1, self.num_agent)[:, self.train_idx], next_agentq_value.view(-1, self.num_agent)[:, self.train_idx]
        )  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        agent_qf2_loss = F.mse_loss(
            agent_qf2.view(-1, self.num_agent)[:, self.train_idx], next_agentq_value.view(-1, self.num_agent)[:, self.train_idx]
        )  
        agent_qf_loss = agent_qf1_loss + agent_qf2_loss
        self.local_critic_optim.zero_grad()
        agent_qf_loss.backward()
        self.local_critic_optim.step()

        # 更新policy网络
        predict_action, predict_log_pi = self.policy.get_log_prob(state_tensor, self.large_edge_index)
        # 生成current_policy做出的action和state的图
        agent_q1, agent_q2 = self.local_critic(state_tensor, predict_action)
        qf1_pi = agent_q1.view(-1, self.num_agent)
        qf2_pi = agent_q2.view(-1, self.num_agent)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        predict_log_pi = predict_log_pi.view(-1, self.num_agent)
        policy_loss = ((self.alpha * predict_log_pi[:, self.train_idx]) - min_qf_pi[:, self.train_idx]).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        if updates % self.target_update_interval == 0:
            print("soft update")
            soft_update(self.local_critic_target, self.local_critic, self.tau)
            soft_update(self.graph_critic_target, self.graph_critic, self.tau)

        return agent_qf_loss.item(), policy_loss.item()

