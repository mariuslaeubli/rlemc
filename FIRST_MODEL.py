import datetime
import os
import random
import time
from collections import deque
from typing import Callable, Optional

import gym
import gym.wrappers.frame_stack
import numpy as np
from tensorboardX import SummaryWriter
from absl import flags, app

import rle_assignment.env

FLAGS = flags.FLAGS

# Basic flags
flags.DEFINE_enum('mode', 'train', ['train', 'eval'], 'Run mode.')
flags.DEFINE_string('logdir', './runs', 'Directory where all outputs are written to.')
flags.DEFINE_string('run_name', datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'), 'Run name.')
flags.DEFINE_bool('cuda', True, 'Whether to run the model on gpu or on cpu.') # cannot really be used in this code since i am not allowed to import other libraries like torch
flags.DEFINE_integer('seed', 42, 'Random seed.')

# Training flags
flags.DEFINE_integer('num_envs', 4, 'Number of parallel env processes.')
flags.DEFINE_integer('total_steps', 1_000_000, 'Total number of agent steps.')
flags.DEFINE_integer('checkpoint_freq', 100_000, 'Frequency at which checkpoints are stored.')
flags.DEFINE_integer('logging_freq', 10_000, 'Frequency at which logs are written.')
flags.DEFINE_integer('n_episodes', 1000, 'Number of episodes.')
flags.DEFINE_integer('max_t', 1200, 'Maximum number of time steps per episode.')

# DQN specific flags
flags.DEFINE_float('gamma', 0.99, 'Discount factor for future rewards.')
flags.DEFINE_float('epsilon_start', 0.5, 'Starting value of epsilon for epsilon-greedy policy.')
flags.DEFINE_float('epsilon_min', 0.03, 'Minimum value of epsilon.')
flags.DEFINE_float('epsilon_decay', 0.995, 'Decay rate of epsilon per episode.')
flags.DEFINE_float('learning_rate', 0.005, 'Learning rate for the optimizer.') 
flags.DEFINE_integer('update_every', 4, 'How often to update the network.')
flags.DEFINE_integer('target_update_freq', 10, 'How often to update the target network.') 


# Evaluation flags
flags.DEFINE_integer('eval_num_episodes', 30, 'Number of eval episodes.')
flags.DEFINE_bool('eval_render', False, 'Render env during eval.')
flags.DEFINE_integer('eval_seed', 42, 'Eval seed.')
flags.DEFINE_string('model_path', None, 'Path to the saved model for evaluation.')

def make_env_fn(seed: int, render_human: bool = False, video_folder: Optional[str] = None) -> Callable[[], gym.Env]:
    """Returns a pickleable callable to create an env instance."""
    def env_fn():
        env = rle_assignment.env.make_env(render_human, video_folder)

        # Maybe add other gym.wrappers
        enc = gym.wrappers.frame_stack.FrameStack(env, 4)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return env_fn

def initialize_weights(input_size, output_size):
    """Initialize weights for a fully connected layer with Xavier initialization."""
    limit = np.sqrt(6 / (input_size + output_size))
    return np.random.uniform(-limit, limit, (input_size, output_size))

def compute_loss(y_true, y_pred):
    """Compute mean squared error loss."""
    loss = np.mean((y_true - y_pred) ** 2)
    return loss if np.isfinite(loss) else np.inf

def clip_gradients(grads, max_grad_norm):
    """Clip gradients to avoid explosion."""
    for key in grads:
        norm = np.linalg.norm(grads[key])
        if norm > max_grad_norm:
            grads[key] = grads[key] * (max_grad_norm / norm)
    return grads

class SimpleNN:
    def __init__(self, input_shape, hidden_size, output_size):
        self.input_shape = input_shape
        self.input_size = np.prod(input_shape)  # Flattened input size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights1 = initialize_weights(self.input_size, hidden_size)
        self.bias1 = np.zeros((1, hidden_size))
        self.weights2 = initialize_weights(hidden_size, hidden_size)
        self.bias2 = np.zeros((1, hidden_size))
        self.weights3 = initialize_weights(hidden_size, output_size)
        self.bias3 = np.zeros((1, output_size))

    def forward(self, x):
        x = x.flatten().reshape(-1, self.input_size)  # Flatten the input
        self.z1 = np.dot(x, self.weights1) + self.bias1
        self.a1 = np.maximum(0, self.z1)  # ReLU activation
        self.z2 = np.dot(self.a1, self.weights2) + self.bias2
        self.a2 = np.maximum(0, self.z2)  # ReLU activation
        self.z3 = np.dot(self.a2, self.weights3) + self.bias3
        return self.z3

    def predict(self, x):
        return self.forward(x)

    def update_weights(self, grads, lr):
        self.weights1 -= lr * grads['dW1']
        self.bias1 -= lr * grads['db1']
        self.weights2 -= lr * grads['dW2']
        self.bias2 -= lr * grads['db2']
        self.weights3 -= lr * grads['dW3']
        self.bias3 -= lr * grads['db3']

    def get_weights(self):
        return {
            'weights1': self.weights1,
            'bias1': self.bias1,
            'weights2': self.weights2,
            'bias2': self.bias2,
            'weights3': self.weights3,
            'bias3': self.bias3
        }

    def set_weights(self, weights):
        self.weights1 = weights['weights1']
        self.bias1 = weights['bias1']
        self.weights2 = weights['weights2']
        self.bias2 = weights['bias2']
        self.weights3 = weights['weights3']
        self.bias3 = weights['bias3']

def compute_grads(network, x, y_true):
    m = x.shape[0]
    
    # Forward pass
    y_pred = network.forward(x)
    
    # Compute loss gradient
    dz3 = (y_pred - y_true) / m
    dW3 = np.dot(network.a2.T, dz3)
    db3 = np.sum(dz3, axis=0, keepdims=True)

    da2 = np.dot(dz3, network.weights3.T)
    dz2 = da2 * (network.z2 > 0)
    dW2 = np.dot(network.a1.T, dz2)
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = np.dot(dz2, network.weights2.T)
    dz1 = da1 * (network.z1 > 0)
    dW1 = np.dot(x.flatten().reshape(-1, network.input_size).T, dz1)
    db1 = np.sum(dz1, axis=0, keepdims=True)

    grads = {
        'dW3': dW3,
        'db3': db3,
        'dW2': dW2,
        'db2': db2,
        'dW1': dW1,
        'db1': db1
    }
    
    # Clip gradients
    grads = clip_gradients(grads, max_grad_norm=1.0)

    return grads

def sample_action(network, state, epsilon, action_size):
    """Sample action using epsilon-greedy strategy."""
    if np.random.rand() <= epsilon:
        return random.randrange(action_size)
    q_values = network.predict(state)
    return np.argmax(q_values[0])

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = np.vstack([e[0] for e in experiences])
        actions = np.array([e[1] for e in experiences]).astype(np.int32)
        rewards = np.array([e[2] for e in experiences]).astype(np.float32)
        next_states = np.vstack([e[3] for e in experiences])
        dones = np.array([e[4] for e in experiences]).astype(np.uint8)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    """Interacts with and learns from the environment."""
    def __init__(self, state_shape, action_size):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = ReplayBuffer(buffer_size=100000, batch_size=64)
        self.gamma = FLAGS.gamma
        self.epsilon = FLAGS.epsilon_start
        self.epsilon_min = FLAGS.epsilon_min
        self.epsilon_decay = FLAGS.epsilon_decay
        self.learning_rate = FLAGS.learning_rate
        self.update_every = FLAGS.update_every
        self.target_update_freq = FLAGS.target_update_freq  # New target update frequency

        self.qnetwork_local = SimpleNN(state_shape, 64, action_size)
        self.qnetwork_target = SimpleNN(state_shape, 64, action_size)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:
            if len(self.memory) > self.memory.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences)
            # Update target network
            if self.t_step % self.target_update_freq == 0:
                self.qnetwork_target.set_weights(self.qnetwork_local.get_weights())

    def act(self, state):
        state = np.expand_dims(state, axis=0)
        return sample_action(self.qnetwork_local, state, self.epsilon, self.action_size)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Normalize rewards
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-5)

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target.predict(next_states).max(axis=1)

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        mask = np.zeros((self.memory.batch_size, self.action_size))
        mask[np.arange(self.memory.batch_size), actions] = 1
        Q_expected = np.sum(self.qnetwork_local.predict(states) * mask, axis=1)

        # Compute loss
        loss = compute_loss(Q_expected, Q_targets)
        if not np.isfinite(loss):
            print("Loss is not finite, skipping this batch.")
            return

        # Compute gradients
        grads = compute_grads(self.qnetwork_local, states, Q_targets.reshape(-1, 1))

        # Update weights
        self.qnetwork_local.update_weights(grads, self.learning_rate)

        # Update epsilon
        self.update_epsilon()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters."""
        local_weights = local_model.get_weights()
        target_weights = target_model.get_weights()
        for key in local_weights:
            target_weights[key] = tau * local_weights[key] + (1.0 - tau) * target_weights[key]
        target_model.set_weights(target_weights)

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filepath):
        weights = self.qnetwork_local.get_weights()
        np.savez(filepath, **weights)

    def load(self, filepath):
        data = np.load(filepath)
        weights = {key: data[key] for key in data.files}
        self.qnetwork_local.set_weights(weights)
        self.qnetwork_target.set_weights(weights)

# Main script remains the same
def main(argv):
    del argv  # Unused.
    
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
            score = 0
            for t in range(FLAGS.max_t):
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            print(f"Episode {episode + 1}\tScore: {score}")
            writer.add_scalar('Score', score, episode)
            writer.add_scalar('Epsilon', agent.epsilon, episode)
            writer.add_scalar('Steps', t, episode)

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
            score = 0
            done = False
            while not done:
                action = agent.act(state)
                state, reward, done, _ = env.step(action)
                score += reward
                if FLAGS.eval_render:
                    env.render()
            print(f"Eval Episode {episode + 1}\tScore: {score}")

if __name__ == "__main__":
    app.run(main)