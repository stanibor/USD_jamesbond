import numpy as np
import cv2 as cv

class FramePreprocessor:
    def __init__(self, paddings=(0, 0, 0, 0), size=(84, 84), color_weights=(0.33, 0.33, 0.33)):
        self.paddings = paddings
        self.size = size
        self.color_weights = np.array(color_weights)

    def process(self, frame):
        out = frame[self.paddings[0]:-(self.paddings[1]+1), self.paddings[1]:-(self.paddings[3]+1), :] * (1.0 / 255.0)
        out = out.dot(self.color_weights)
        out = cv.resize(out, self.size)
        return out

