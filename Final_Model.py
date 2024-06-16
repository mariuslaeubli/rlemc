import datetime
import os
import random
import time
from collections import deque, namedtuple
from typing import Callable, Optional

import gym
import numpy as np
from tensorboardX import SummaryWriter
from absl import flags, app

import rle_assignment.env

FLAGS = flags.FLAGS

# Basic flags
flags.DEFINE_enum('mode', 'eval', ['train', 'eval'], 'Run mode.')
flags.DEFINE_string('logdir', './runs', 'Directory where all outputs are written to.')
flags.DEFINE_string('run_name', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'Run name.')
flags.DEFINE_bool('cuda', True, 'Whether to run the model on GPU or on CPU.')  # Not used without torch
flags.DEFINE_integer('seed', 42, 'Random seed.')

# Training flags
flags.DEFINE_integer('num_envs', 2, 'Number of parallel env processes.')
flags.DEFINE_integer('total_steps', 1000000, 'Total number of agent steps.')
flags.DEFINE_integer('checkpoint_freq', 100000, 'Frequency at which checkpoints are stored.')
flags.DEFINE_integer('logging_freq', 10000, 'Frequency at which logs are written.')
flags.DEFINE_integer('n_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('max_t', 5000, 'Maximum number of time steps per episode.')

# DQN specific flags
flags.DEFINE_float('gamma', 0.99, 'Discount factor for future rewards.')
flags.DEFINE_float('epsilon_start', 1.0, 'Starting value of epsilon for epsilon-greedy policy.')
flags.DEFINE_float('epsilon_min', 0.01, 'Minimum value of epsilon.')
flags.DEFINE_float('epsilon_decay', 0.995, 'Decay rate of epsilon per episode.')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate for the optimizer.')
flags.DEFINE_integer('update_every', 4, 'How often to update the network.')
flags.DEFINE_integer('target_update_freq', 1000, 'How often to update the target network.')
flags.DEFINE_float('alpha', 0.6, 'Alpha parameter for prioritized experience replay.')
flags.DEFINE_float('beta_start', 0.4, 'Starting value of beta for prioritized experience replay.')
flags.DEFINE_float('beta_frames', 100000, 'Number of frames over which beta is annealed.')
flags.DEFINE_integer('batch_size', 64, 'Batch size for experience replay.')

# Evaluation flags
flags.DEFINE_integer('eval_num_episodes', 30, 'Number of eval episodes.')
flags.DEFINE_bool('eval_render', False, 'Render env during eval.')
flags.DEFINE_integer('eval_seed', 42, 'Eval seed.')
flags.DEFINE_string('model_path', None, 'Path to the saved model for evaluation.')

def make_env_fn(seed: int, render_human: bool = False, video_folder: Optional[str] = None) -> Callable[[], gym.Env]:
    """Returns a pickleable callable to create an env instance."""
    def env_fn():
        env = rle_assignment.env.make_env(render_human, video_folder)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return env_fn

def initialize_weights(input_size, output_size):
    """Initialize weights for a fully connected layer with Xavier initialization."""
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))

class DuelingDQN:
    def __init__(self, input_shape, n_actions):
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)
        self.hidden_size = 512
        self.n_actions = n_actions

        self.weights1 = initialize_weights(self.input_size, 512)
        self.bias1 = np.zeros((1, 512))
        self.weights2 = initialize_weights(512, 512)
        self.bias2 = np.zeros((1, 512))
        self.value_weights = initialize_weights(512, 1)
        self.value_bias = np.zeros((1, 1))
        self.advantage_weights = initialize_weights(512, n_actions)
        self.advantage_bias = np.zeros((1, n_actions))

    def forward(self, x):
        x = x.flatten().reshape(-1, self.input_size) / 255.0
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = np.maximum(0, self.z2)  # ReLU activation
        self.value = np.dot(self.a2, self.value_weights) + self.value_bias
        self.advantage = np.dot(self.a2, self.advantage_weights) + self.advantage_bias
        qvals = self.value + (self.advantage - np.mean(self.advantage, axis=1, keepdims=True))
        return qvals

    def predict(self, x):
        return self.forward(x)

    def update_weights(self, grads, lr):
        self.weights1 -= lr * grads['dW1']
        self.bias1 -= lr * grads['db1']
        self.weights2 -= lr * grads['dW2']
        self.bias2 -= lr * grads['db2']
        self.value_weights -= lr * grads['dVW']
        self.value_bias -= lr * grads['dbV']
        self.advantage_weights -= lr * grads['dAW']
        self.advantage_bias -= lr * grads['dbA']

    def get_weights(self):
        return {
            'weights1': self.weights1,
            'bias1': self.bias1,
            'weights2': self.weights2,
            'bias2': self.bias2,
            'value_weights': self.value_weights,
            'value_bias': self.value_bias,
            'advantage_weights': self.advantage_weights,
            'advantage_bias': self.advantage_bias
        }

    def set_weights(self, weights):
        self.weights1 = weights['weights1']
        self.bias1 = weights['bias1']
        self.weights2 = weights['weights2']
        self.bias2 = weights['bias2']
        self.value_weights = weights['value_weights']
        self.value_bias = weights['value_bias']
        self.advantage_weights = weights['advantage_weights']
        self.advantage_bias = weights['advantage_bias']

def compute_grads(network, x, y_true):
    m = x.shape[0]
    
    # Forward pass
    y_pred = network.forward(x)
    
    # Compute loss gradient
    dz = (y_pred - y_true) / m

    # Advantage stream gradients
    dAW = np.dot(network.a2.T, dz)
    dbA = np.sum(dz, axis=0, keepdims=True)
    da2_adv = np.dot(dz, network.advantage_weights.T)

    # Value stream gradients
    dz_value = dz.mean(axis=1, keepdims=True)
    dVW = np.dot(network.a2.T, dz_value)
    dbV = np.sum(dz_value, axis=0, keepdims=True)
    da2_val = np.dot(dz_value, network.value_weights.T)

    # Combine gradients
    dz2 = (da2_adv + da2_val) * (network.z2 > 0)
    dW2 = np.dot(network.a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, network.weights2.T)
    dz1 = da1 * (network.z1 > 0)
    dW1 = np.dot(x.flatten().reshape(-1, network.input_size).T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    grads = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2,
        'dVW': dVW,
        'dbV': dbV,
        'dAW': dAW,
        'dbA': dbA
    }

    return grads

def clip_gradients(grads, max_grad_norm):
    """Clip gradients to avoid explosion."""
    for key in grads:
        norm = np.linalg.norm(grads[key])
        if norm > max_grad_norm:
            grads[key] = grads[key] * (max_grad_norm / norm)
    return grads

def sample_action(network, state, epsilon, action_size):
    """Sample action using epsilon-greedy strategy."""
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    q_values = network.predict(state)
    return np.argmax(q_values[0])

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.buffer = []
        self.experience = namedtuple('Experience', 
                                     field_names=['state', 'action', 'reward', 'next_state', 'done'])

    def add(self, state, action, reward, next_state, done):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(self.experience(state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = self.experience(state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs = prios ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = PrioritizedReplayBuffer(capacity=100000, alpha=FLAGS.alpha)
        self.gamma = FLAGS.gamma
        self.epsilon = FLAGS.epsilon_start
        self.epsilon_min = FLAGS.epsilon_min
        self.epsilon_decay = FLAGS.epsilon_decay
        self.learning_rate = FLAGS.learning_rate
        self.update_every = FLAGS.update_every
        self.target_update_freq = FLAGS.target_update_freq
        self.batch_size = FLAGS.batch_size 

        self.qnetwork_local = DuelingDQN(state_shape, action_size)
        self.qnetwork_target = DuelingDQN(state_shape, action_size)
        self.t_step = 0
        self.beta = FLAGS.beta_start
        self.beta_increment = (1.0 - FLAGS.beta_start) / FLAGS.beta_frames

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step += 1
        if self.t_step % self.update_every == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample(self.batch_size, self.beta)
                self.learn(experiences)
            # Update target network
            if self.t_step % self.target_update_freq == 0:
                self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        return sample_action(self.qnetwork_local, state, self.epsilon, self.action_size)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, weights, indices = experiences

        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-5)

        # Get max predicted Q values (for next states) from target model
        best_actions = np.argmax(self.qnetwork_local.predict(next_states), axis=1)
        Q_targets_next = self.qnetwork_target.predict(next_states)[np.arange(self.batch_size), best_actions]

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        mask = np.zeros((self.batch_size, self.action_size))
        mask[np.arange(self.batch_size), actions] = 1
        Q_expected = np.sum(self.qnetwork_local.predict(states) * mask, axis=1)

        # Compute loss
        loss = np.mean(weights * (Q_expected - Q_targets) ** 2)
        if not np.isfinite(loss):
            print("Loss is not finite, skipping this batch.")
            return

        # Compute gradients
        grads = compute_grads(self.qnetwork_local, states, Q_targets.reshape(-1, 1))
        
        # Clip gradients
        grads = clip_gradients(grads, max_grad_norm=1.0)

        # Update weights
        self.qnetwork_local.update_weights(grads, self.learning_rate)

        # Update priorities
        td_errors = np.abs(Q_expected - Q_targets)
        self.memory.update_priorities(indices, td_errors)

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.beta = min(1.0, self.beta + self.beta_increment)

    def save(self, filepath):
        weights = self.qnetwork_local.get_weights()
        np.savez(filepath, **weights)

    def load(self, filepath):
        data = np.load(filepath)
        weights = {key: data[key] for key in data.files}
        self.qnetwork_local.set_weights(weights)
        self.qnetwork_target.set_weights(weights)

def main(argv):
    del argv  
    
    seed = FLAGS.seed
    logdir = os.path.join(FLAGS.logdir, FLAGS.run_name)
    os.makedirs(logdir, exist_ok=True)
    
    if FLAGS.mode == 'train':
        env_fn = make_env_fn(seed)
        env = env_fn()
        state_shape = env.observation_space.shape
        action_size = env.action_space.n
        agent = DQNAgent(state_shape, action_size)
        writer = SummaryWriter(logdir)

        for episode in range(FLAGS.n_episodes):
            state = env.reset()

            if len(state.shape) == 2:  # Change if grayscale not used
                state = np.expand_dims(state, axis=0)  
            score = 0
            for t in range(FLAGS.max_t):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)


                if len(next_state.shape) == 2:  # Change if grayscale not used
                    next_state = np.expand_dims(next_state, axis=0) 

                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            print(f"Episode {episode + 1}\tScore: {score}")
            writer.add_scalar('Score', score, episode)

            # Save model checkpoint
            if (episode + 1) % FLAGS.checkpoint_freq == 0:
                agent.save(os.path.join(logdir, f'model_{episode + 1}.npz'))

        # Save the final model
        agent.save(os.path.join(logdir, 'model_final.npz'))

        env.close()
        writer.close()
    
    elif FLAGS.mode == 'eval':
        env_fn = make_env_fn(FLAGS.eval_seed, FLAGS.eval_render)
        env = env_fn()
        state_shape = env.observation_space.shape
        action_size = env.action_space.n
        agent = DQNAgent(state_shape, action_size)
        
        # Load the agent's model if model_path is provided
        if FLAGS.model_path:
            agent.load(FLAGS.model_path)
        else:
            agent.load(os.path.join(logdir, 'model_final.npz'))

        for episode in range(FLAGS.eval_num_episodes):
            state = env.reset()
            if len(state.shape) == 2: # Change if grayscale not used
                state = np.expand_dims(state, axis=0)  
            score = 0
            done = False
            while not done:
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                if len(next_state.shape) == 2:  # Change if grayscale not used
                    next_state = np.expand_dims(next_state, axis=0)  
                state = next_state
                score += reward
                if FLAGS.eval_render:
                    env.render()
            print(f"Eval Episode {episode + 1}\tScore: {score}")

if __name__ == "__main__":
    app.run(main)
