import numpy as np
import torch
import torch.nn as nn
from utils.util import get_gard_norm, huber_loss, mse_loss
from utils.valuenorm import ValueNorm
from algorithms.utils.util import check

class R_IPPO():
    """
    Trainer class for Independent PPO to update policies.
    Each agent has its own policy and critic network, trained independently.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_IPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self,
                 args,
                 policy,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        
        # IPPO specific: number of agents
        self.num_agents = args.num_agents

        assert (self._use_popart and self._use_valuenorm) == False, (
            "self._use_popart and self._use_valuenorm can not be set True simultaneously")

        # IPPO: Independent value normalizers for each agent
        if self._use_popart:
            self.value_normalizer = [self.policy.critic.v_out for _ in range(self.num_agents)]
        elif self._use_valuenorm:
            self.value_normalizer = [ValueNorm(1, device=self.device) for _ in range(self.num_agents)]
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch, agent_id=0):
        """
        Calculate value function loss for a specific agent.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value predictions from data batch.
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead.
        :param agent_id: (int) agent index for independent value normalization.

        :return value_loss: (torch.Tensor) value function loss.
        """
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        
        # IPPO: Use agent-specific value normalizer
        if self._use_popart or self._use_valuenorm:
            if self.value_normalizer is not None:
                self.value_normalizer[agent_id].update(return_batch)
                error_clipped = self.value_normalizer[agent_id].normalize(return_batch) - value_pred_clipped
                error_original = self.value_normalizer[agent_id].normalize(return_batch) - values
            else:
                error_clipped = return_batch - value_pred_clipped
                error_original = return_batch - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def ppo_update_single_agent(self, sample, agent_id, update_actor=True):
        """
        Update actor and critic networks for a single agent independently.
        :param sample: (Tuple) contains data batch for single agent.
        :param agent_id: (int) agent index.
        :param update_actor: (bool) whether to update actor network.

        :return: training statistics for this agent
        """
        # Extract single agent data from batch
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)

        # IPPO: Use only individual agent's observation for critic (not shared observation)
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(obs_batch,  # Use obs_batch instead of share_obs_batch
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              actions_batch,
                                                                              masks_batch,
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        
        # Actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        # IPPO: Independent actor optimizer for each agent
        self.policy.actor_optimizer[agent_id].zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor[agent_id].parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor[agent_id].parameters())

        self.policy.actor_optimizer[agent_id].step()

        # Critic update with agent-specific value normalizer
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch, agent_id)

        # IPPO: Independent critic optimizer for each agent
        self.policy.critic_optimizer[agent_id].zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic[agent_id].parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic[agent_id].parameters())

        self.policy.critic_optimizer[agent_id].step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights

    def ppo_update(self, sample, update_actor=True):
        """
        Update all agents independently.
        :param sample: (Tuple) contains data batch with which to update networks.
        :param update_actor: (bool) whether to update actor network.

        :return: aggregated training statistics
        """
        # Initialize aggregated losses
        total_value_loss = 0
        total_critic_grad_norm = 0
        total_policy_loss = 0
        total_dist_entropy = 0
        total_actor_grad_norm = 0
        total_imp_weights = 0

        # IPPO: Update each agent independently
        for agent_id in range(self.num_agents):
            # Extract agent-specific sample
            agent_sample = self._extract_agent_sample(sample, agent_id)
            
            value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights = \
                self.ppo_update_single_agent(agent_sample, agent_id, update_actor)
            
            # Aggregate losses
            total_value_loss += value_loss
            total_critic_grad_norm += critic_grad_norm
            total_policy_loss += policy_loss
            total_dist_entropy += dist_entropy
            total_actor_grad_norm += actor_grad_norm
            total_imp_weights += imp_weights.mean()

        # Return averaged losses
        return (total_value_loss / self.num_agents,
                total_critic_grad_norm / self.num_agents,
                total_policy_loss / self.num_agents,
                total_dist_entropy / self.num_agents,
                total_actor_grad_norm / self.num_agents,
                total_imp_weights / self.num_agents)

    def _extract_agent_sample(self, sample, agent_id):
        """
        Extract data sample for a specific agent.
        :param sample: (Tuple) full batch sample
        :param agent_id: (int) agent index
        :return: agent-specific sample
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch = sample

        # Extract agent-specific data
        agent_obs_batch = obs_batch[:, agent_id]
        agent_share_obs_batch = share_obs_batch[:, agent_id] if share_obs_batch is not None else None
        agent_rnn_states_batch = rnn_states_batch[:, agent_id] if rnn_states_batch is not None else None
        agent_rnn_states_critic_batch = rnn_states_critic_batch[:, agent_id] if rnn_states_critic_batch is not None else None
        agent_actions_batch = actions_batch[:, agent_id]
        agent_value_preds_batch = value_preds_batch[:, agent_id]
        agent_return_batch = return_batch[:, agent_id]
        agent_masks_batch = masks_batch[:, agent_id] if masks_batch is not None else None
        agent_active_masks_batch = active_masks_batch[:, agent_id] if active_masks_batch is not None else None
        agent_old_action_log_probs_batch = old_action_log_probs_batch[:, agent_id]
        agent_adv_targ = adv_targ[:, agent_id]
        agent_available_actions_batch = available_actions_batch[:, agent_id] if available_actions_batch is not None else None

        return (agent_share_obs_batch, agent_obs_batch, agent_rnn_states_batch, agent_rnn_states_critic_batch,
                agent_actions_batch, agent_value_preds_batch, agent_return_batch, agent_masks_batch,
                agent_active_masks_batch, agent_old_action_log_probs_batch, agent_adv_targ, agent_available_actions_batch)

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD for all agents independently.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update.
        """
        # IPPO: Calculate advantages independently for each agent
        train_info = {}
        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0

        # IPPO: Process each agent independently
        for agent_id in range(self.num_agents):
            if self._use_popart or self._use_valuenorm:
                if self.value_normalizer is not None:
                    advantages = buffer.returns[:-1, agent_id] - self.value_normalizer[agent_id].denormalize(buffer.value_preds[:-1, agent_id])
                else:
                    advantages = buffer.returns[:-1, agent_id] - buffer.value_preds[:-1, agent_id]
            else:
                advantages = buffer.returns[:-1, agent_id] - buffer.value_preds[:-1, agent_id]
            
            advantages_copy = advantages.copy()
            if hasattr(buffer, 'active_masks') and buffer.active_masks is not None:
                advantages_copy[buffer.active_masks[:-1, agent_id] == 0.0] = np.nan
            
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

            # Training loop for this agent
            for _ in range(self.ppo_epoch):
                if self._use_recurrent_policy:
                    data_generator = buffer.recurrent_generator_single_agent(advantages, self.num_mini_batch, 
                                                                           self.data_chunk_length, agent_id)
                elif self._use_naive_recurrent:
                    data_generator = buffer.naive_recurrent_generator_single_agent(advantages, self.num_mini_batch, agent_id)
                else:
                    data_generator = buffer.feed_forward_generator_single_agent(advantages, self.num_mini_batch, agent_id)

                for sample in data_generator:
                    value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                        = self.ppo_update_single_agent(sample, agent_id, update_actor)

                    train_info['value_loss'] += value_loss.item()
                    train_info['policy_loss'] += policy_loss.item()
                    train_info['dist_entropy'] += dist_entropy.item()
                    train_info['actor_grad_norm'] += actor_grad_norm
                    train_info['critic_grad_norm'] += critic_grad_norm
                    train_info['ratio'] += imp_weights.mean()

        num_updates = self.ppo_epoch * self.num_mini_batch * self.num_agents

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        """Set networks to training mode."""
        for agent_id in range(self.num_agents):
            self.policy.actor[agent_id].train()
            self.policy.critic[agent_id].train()

    def prep_rollout(self):
        """Set networks to evaluation mode."""
        for agent_id in range(self.num_agents):
            self.policy.actor[agent_id].eval()
            self.policy.critic[agent_id].eval()