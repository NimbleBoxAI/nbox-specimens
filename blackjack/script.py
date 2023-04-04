# Author: Till Zemann
# License: MIT License

from __future__ import annotations

from collections import defaultdict
from cloudpickle import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch
from tqdm import trange

import gymnasium as gym

from nbox import Project

class BlackjackAgent:
  def __init__(
    self,
    learning_rate: float,
    initial_epsilon: float,
    epsilon_decay: float,
    final_epsilon: float,
    env,
    discount_factor: float = 0.95,
  ):
    """Initialize a Reinforcement Learning agent with an empty dictionary
    of state-action values (q_values), a learning rate and an epsilon.

    Args:
      learning_rate: The learning rate
      initial_epsilon: The initial epsilon value
      epsilon_decay: The decay for epsilon
      final_epsilon: The final epsilon value
      discount_factor: The discount factor for computing the Q-value
    """
    self.env = env

    self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

    self.lr = learning_rate
    self.discount_factor = discount_factor

    self.epsilon = initial_epsilon
    self.epsilon_decay = epsilon_decay
    self.final_epsilon = final_epsilon

    self.training_error = []

  def get_action(self, obs: tuple[int, int, bool]) -> int:
    """
    Returns the best action with probability (1 - epsilon)
    otherwise a random action with probability epsilon to ensure exploration.
    """
    # with probability epsilon return a random action to explore the environment
    if np.random.random() < self.epsilon:
      return self.env.action_space.sample()

    # with probability (1 - epsilon) act greedily (exploit)
    else:
      return int(np.argmax(self.q_values[obs]))

  def update(
    self,
    obs: tuple[int, int, bool],
    action: int,
    reward: float,
    terminated: bool,
    next_obs: tuple[int, int, bool],
  ):
    """Updates the Q-value of an action."""
    future_q_value = (not terminated) * np.max(self.q_values[next_obs])
    temporal_difference = (
      reward + self.discount_factor * future_q_value - self.q_values[obs][action]
    )

    self.q_values[obs][action] = (
      self.q_values[obs][action] + self.lr * temporal_difference
    )
    self.training_error.append(temporal_difference)

  def decay_epsilon(self):
    self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


def create_grids(agent, usable_ace=False):
  """Create value and policy grid given an agent."""
  # convert our state-action values to state values
  # and build a policy dictionary that maps observations to actions
  state_value = defaultdict(float)
  policy = defaultdict(int)
  for obs, action_values in agent.q_values.items():
    state_value[obs] = float(np.max(action_values))
    policy[obs] = int(np.argmax(action_values))

  player_count, dealer_count = np.meshgrid(
    # players count, dealers face-up card
    np.arange(12, 22),
    np.arange(1, 11),
  )

  # create the value grid for plotting
  value = np.apply_along_axis(
    lambda obs: state_value[(obs[0], obs[1], usable_ace)],
    axis=2,
    arr=np.dstack([player_count, dealer_count]),
  )
  value_grid = player_count, dealer_count, value

  # create the policy grid for plotting
  policy_grid = np.apply_along_axis(
    lambda obs: policy[(obs[0], obs[1], usable_ace)],
    axis=2,
    arr=np.dstack([player_count, dealer_count]),
  )
  return value_grid, policy_grid


def create_plots(value_grid, policy_grid, title: str):
  """Creates a plot using a value and policy grid."""
  # create a new figure with 2 subplots (left: state values, right: policy)
  player_count, dealer_count, value = value_grid
  fig = plt.figure(figsize=plt.figaspect(0.4))
  fig.suptitle(title, fontsize=16)

  # plot the state values
  ax1 = fig.add_subplot(1, 2, 1, projection="3d")
  ax1.plot_surface(
    player_count,
    dealer_count,
    value,
    rstride=1,
    cstride=1,
    cmap="viridis",
    edgecolor="none",
  )
  plt.xticks(range(12, 22), range(12, 22))
  plt.yticks(range(1, 11), ["A"] + list(range(2, 11)))
  ax1.set_title(f"State values: {title}")
  ax1.set_xlabel("Player sum")
  ax1.set_ylabel("Dealer showing")
  ax1.zaxis.set_rotate_label(False)
  ax1.set_zlabel("Value", fontsize=14, rotation=90)
  ax1.view_init(20, 220)

  # plot the policy
  fig.add_subplot(1, 2, 2)
  ax2 = sns.heatmap(policy_grid, linewidth=0, annot=True, cmap="Accent_r", cbar=False)
  ax2.set_title(f"Policy: {title}")
  ax2.set_xlabel("Player sum")
  ax2.set_ylabel("Dealer showing")
  ax2.set_xticklabels(range(12, 22))
  ax2.set_yticklabels(["A"] + list(range(2, 11)), fontsize=12)

  # add a legend
  legend_elements = [
    Patch(facecolor="lightgreen", edgecolor="black", label="Hit"),
    Patch(facecolor="grey", edgecolor="black", label="Stick"),
  ]
  ax2.legend(handles=legend_elements, bbox_to_anchor=(1.3, 1))
  return fig



def main(
  learning_rate: float = 0.01,
  n_episodes: int = 100_000,
  start_epsilon: float = 1.0,
  final_epsilon: float = 0.1,
  log_every: int = 100,
):
  p = Project()
  tracker = p.get_exp_tracker()

  # Let's start by creating the blackjack environment.
  # Note: We are going to follow the rules from Sutton & Barto.
  # Other versions of the game can be found below for you to experiment.
  env = gym.make("Blackjack-v1", sab=True)

  # create an agent
  agent = BlackjackAgent(
    learning_rate = learning_rate,
    initial_epsilon = start_epsilon,
    epsilon_decay = start_epsilon / (n_episodes / 2), # reduce the exploration over time
    final_epsilon = final_epsilon,
    env = env,
  )

  env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=n_episodes)
  for ep in trange(n_episodes):
    obs, info = env.reset()
    done = False

    # play one episode
    while not done:
      action = agent.get_action(obs)
      next_obs, reward, terminated, truncated, info = env.step(action)

      # update the agent
      agent.update(obs, action, reward, terminated, next_obs)

      # update if the environment is done and the current obs
      done = terminated or truncated
      obs = next_obs

    agent.decay_epsilon()

    if ep and (ep+1) % log_every == 0:
      tracker.log({
        'episode': ep+1,
        'reward': float(np.array(env.return_queue).flatten()[-log_every:].mean()),
        'length': float(np.array(env.length_queue).flatten()[-log_every:].mean()),
        'td_error': float(np.mean(agent.training_error[-log_every:])),
        'epsilon': agent.epsilon,
        'lr': agent.lr,
        'discount_factor': agent.discount_factor,
      })

  # state values & policy with usable ace (ace counts as 11)
  value_grid, policy_grid = create_grids(agent, usable_ace=True)
  fig1 = create_plots(value_grid, policy_grid, title="With usable ace")
  fig1.savefig('./usable_ace.png')
  tracker.save_file('./usable_ace.png')

  # state values & policy without usable ace (ace counts as 1)
  value_grid, policy_grid = create_grids(agent, usable_ace=False)
  fig2 = create_plots(value_grid, policy_grid, title="Without usable ace")
  fig2.savefig('./no_usable_ace.png')
  tracker.save_file('./no_usable_ace.png')

  # save the agent
  with open('./agent.pkl', 'wb') as f:
    pickle.dump(agent, f)
  tracker.save_file('./agent.pkl')
