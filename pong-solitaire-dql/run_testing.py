
from keras.models import load_model

nn_model = load_model('agent_model.h5')
print(nn_model)

import gym
import gym_pong_solitaire
import numpy as np

env = gym.make('PongSolitaire-v0')

current_state = env.reset()
current_state = np.reshape(current_state, [1,5])

done = False
total_reward = 0

time = 0
while not done:
    prediction = nn_model.predict(current_state)
    action = np.argmax(prediction)
    
    next_state, reward, done, _ = env.step(action)
    env.render()

    next_state = np.reshape(next_state, [1,5])
    
    time += 1
    print("Time: " + str(time) + " | Reward: " + str(reward) + " | Total reward: " + str(total_reward))

    current_state = next_state
    total_reward += reward # La puntuación final es el número de veces que se consigue rebotar la pelota
    
print("Total score: " + str(total_reward))