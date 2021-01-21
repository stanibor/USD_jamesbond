import gym
import time
import numpy as np
import cv2

FPS = 120

if __name__ == '__main__':
    env_name = 'Jamesbond-v0'
    env = gym.make(env_name)
    env.reset()
    print(env.observation_space)
    print(env.action_space)

    done = False

    cv2.namedWindow('game', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('game', 600, 600)

    cv2.namedWindow('robot', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('robot', 600, 600)

    while not done:
        action = env.action_space.sample()
        step, reward, done, info = env.step(action)


        blue = step[38:-20, 8:, :]/255
        blue = blue.dot([0.1, 0.8, 0.1])
        # print(blue.shape)
        small = cv2.resize(blue, (76, 76))
        cv2.imshow('game', blue)
        cv2.imshow('robot', small)
        cv2.waitKey(int(1000/FPS))