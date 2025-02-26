import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from verl.common.distributed_utils import get_rank, get_world_size
from verl.rl.ppo.actor_critic import Actor, Critic
from verl.rl.ppo.rollout import Rollout
from verl.rl.ppo.reference import Reference


class PPOAlgorithm(nn.Module):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        
        # Create actor
        self.actor = Actor(cfg.actor_rollout_ref.actor)
        
        # Create critic
        self.critic = Critic(cfg.critic)
        
        # Create rollout
        self.rollout = Rollout(cfg.actor_rollout_ref.rollout)
        
        # Create reference
        self.reference = Reference(cfg.actor_rollout_ref.ref)
        
        # Create optimizer
        self.actor_optimizer = torch.optim.AdamW(
            self.actor.parameters(),
            lr=cfg.actor_rollout_ref.actor.optim.lr,
            weight_decay=cfg.actor_rollout_ref.actor.optim.weight_decay if hasattr(cfg.actor_rollout_ref.actor.optim, "weight_decay") else 0.0,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=cfg.critic.optim.lr,
            weight_decay=cfg.critic.optim.weight_decay if hasattr(cfg.critic.optim, "weight_decay") else 0.0,
        )
        
        # Create KL controller
        self.kl_ctl = AdaptiveKLController(
            init_kl_coef=cfg.algorithm.kl_ctrl.kl_coef,
            target=cfg.algorithm.kl_ctrl.target,
            horizon=cfg.algorithm.kl_ctrl.horizon,
        )
        
        # Create PPO parameters
        self.clip_param = cfg.algorithm.clip_param
        self.value_clip_param = cfg.algorithm.value_clip_param
        self.ppo_epochs = cfg.algorithm.ppo_epochs
        self.max_grad_norm = cfg.algorithm.max_grad_norm
        
    def train_step(self, batch):
        # Generate responses
        rollouts = self.rollout(
            prompts=batch["prompt"],
            actor=self.actor,
        )
        
        # Compute rewards
        rewards = self.compute_rewards(
            prompts=batch["prompt"],
            responses=rollouts["response"],
            reference_responses=batch["response"],
        )
        
        # Compute advantages
        values = self.critic(
            prompts=batch["prompt"],
            responses=rollouts["response"],
        )
        advantages = rewards - values
        
        # PPO update
        actor_loss, critic_loss, kl_div = self.ppo_update(
            prompts=batch["prompt"],
            responses=rollouts["response"],
            log_probs=rollouts["log_prob"],
            values=values,
            advantages=advantages,
            rewards=rewards,
        )
        
        # Update KL controller
        self.kl_ctl.update(kl_div.mean().item())
        
        # Return metrics
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "kl_div": kl_div.mean().item(),
            "kl_coef": self.kl_ctl.value,
            "reward": rewards.mean().item(),
        }
    
    def validation_step(self, batch):
        # Generate responses
        with torch.no_grad():
            rollouts = self.rollout(
                prompts=batch["prompt"],
                actor=self.actor,
            )
            
            # Compute rewards
            rewards = self.compute_rewards(
                prompts=batch["prompt"],
                responses=rollouts["response"],
                reference_responses=batch["response"],
            )
            
            # Compute values
            values = self.critic(
                prompts=batch["prompt"],
                responses=rollouts["response"],
            )
        
        # Return metrics
        return {
            "reward": rewards.mean().item(),
            "value": values.mean().item(),
        }
    
    def validation_epoch_end(self, outputs):
        # Aggregate metrics
        reward = torch.tensor([x["reward"] for x in outputs]).mean().item()
        value = torch.tensor([x["value"] for x in outputs]).mean().item()
        
        # Return metrics
        return {
            "reward": reward,
            "value": value,
        }
    
    def compute_rewards(self, prompts, responses, reference_responses):
        # Compute rewards using the reference model
        rewards = self.reference(
            prompts=prompts,
            responses=responses,
            reference_responses=reference_responses,
        )
        return rewards
    
    def ppo_update(self, prompts, responses, log_probs, values, advantages, rewards):
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        actor_loss = 0
        critic_loss = 0
        kl_div = 0
        
        for _ in range(self.ppo_epochs):
            # Compute new log probs and values
            new_log_probs = self.actor(
                prompts=prompts,
                responses=responses,
            )
            new_values = self.critic(
                prompts=prompts,
                responses=responses,
            )
            
            # Compute ratio
            ratio = torch.exp(new_log_probs - log_probs)
            
            # Compute surrogate loss
            surrogate1 = ratio * advantages
            surrogate2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
            actor_loss = -torch.min(surrogate1, surrogate2).mean()
            
            # Compute value loss
            value_pred_clipped = values + torch.clamp(new_values - values, -self.value_clip_param, self.value_clip_param)
            value_loss1 = F.mse_loss(new_values, rewards)
            value_loss2 = F.mse_loss(value_pred_clipped, rewards)
            critic_loss = torch.max(value_loss1, value_loss2).mean()
            
            # Compute KL divergence
            kl = new_log_probs - log_probs
            kl_div = kl.mean()
            
            # Compute total loss
            loss = actor_loss + 0.5 * critic_loss + self.kl_ctl.value * kl_div
            
            # Backward
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss.backward()
            
            # Clip grad norm
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            # Optimize
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        
        return actor_loss, critic_loss, kl_div
    
    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "kl_ctl": self.kl_ctl.state_dict(),
        }
    
    def load_state_dict(self, state_dict):
        self.actor.load_state_dict(state_dict["actor"])
        self.critic.load_state_dict(state_dict["critic"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state_dict["critic_optimizer"])
        self.kl_ctl.load_state_dict(state_dict["kl_ctl"])


class AdaptiveKLController:
    def __init__(self, init_kl_coef, target, horizon):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon
    
    def update(self, current):
        error = current - self.target
        self.value = self.value * torch.exp(error / self.horizon)
        return self.value
    
    def state_dict(self):
        return {
            "value": self.value,
            "target": self.target,
            "horizon": self.horizon,
        }
    
    def load_state_dict(self, state_dict):
        self.value = state_dict["value"]
        self.target = state_dict["target"]
        self.horizon = state_dict["horizon"] 