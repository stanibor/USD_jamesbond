import numpy as np
import cv2 as cv

class FramePreprocessor:
    def __init__(self, paddings=(0, 0, 0, 0), dsize=(84, 84), color_weights=(0.33, 0.33, 0.33)):
        self.paddings = paddings
        self.dsize = dsize
        self.color_weights = color_weights

    def process(self, frame):
        frame = frame[self.paddings[0]:self.paddings[1], self.paddings[2]:self.paddings[3], :]
        frame = cv.resize(frame, self.dsize)
        frame = frame.dot(self.color_weights)  # Gray
        frame = frame.astype(np.float32)
        frame *= (1.0 / 255.0)  # Levels
        # frame = np.reshape(frame, [1, 42, 42])
        return frame

