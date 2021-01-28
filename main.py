import gym
import cv2
from frame_processing import FramePreprocessor

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

    preprocessor = FramePreprocessor(paddings=(34, -20, 7, -1), dsize=(84, 84), color_weights=(0.1, 0.8, 0.1))

    while not done:
        action = env.action_space.sample()
        step, reward, done, info = env.step(action)

        frame = step
        robot = preprocessor.process(frame)
        cv2.imshow('game', frame)
        cv2.imshow('robot', robot)
        cv2.waitKey(int(1000/FPS))