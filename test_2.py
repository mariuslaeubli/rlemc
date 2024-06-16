import gym
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory

env = gym.make('SpaceInvaders-v0')
height, width, channels = env.observation_space.shape
actions = env.action_space.n
print(height, width, channels, actions)
meanings = env.unwrapped.get_action_meanings()
print(meanings)

EPISODES = 1000
SHOW_EVERY = 100

''''
for episode in range(EPISODES):
    if episode % SHOW_EVERY == 0:
        print(episode)
        render = True
    else:
        render = False
    if render:
        env = gym.make('ALE/SpaceInvaders-v5', render_mode='human')
    else:
        env = gym.make('ALE/SpaceInvaders-v5')
    env.reset()
    state = env.reset()
    done = False
    score = 0
    while not done:
        action = random.choice(list(range(actions)))
        next_state, reward, termination, truncation, info = env.step(action)
        done = termination or truncation
        score += reward
        state = next_state
    print(f'Episode: {episode}, Score: {score}')
env.close()'''

# Deep Q Learning
def build_model(height, width, channels, actions):
    model = Sequential()
    model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    model.compile(loss='mse', optimizer=Adam())
    return model

model = build_model(height, width, channels, actions)

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                   enable_dueling_network=True, dueling_type='avg',
                   nb_actions=actions, nb_steps_warmup=1000)
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-4))
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

scores = dqn.test(env, nb_episodes=10, visualize=True)
print(np.mean(scores.history['episode_reward']))

dqn.save_weights('savedweights/dqn_weights.h5f', overwrite=True)