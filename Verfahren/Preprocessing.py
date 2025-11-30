import cv2
import numpy as np


class Preprocessing:
    def __init__(self):
        self.img_original = None
        self.img_gray = None
        self.img_binary = None

    def load_image(self, filepath):
        self.img_original = cv2.imread(filepath)
        if self.img_original is None:
            raise FileNotFoundError(f"Image not found at path: {filepath}")
    
    def img_to_gray(self):
        self.img_gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        
    def img_to_binary(self):
        _, self.img_binary = cv2.threshold(self.img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        self.img_binary = (self.img_binary > 0).astype(np.uint8)
        
    def preprocess(self, filepath):
        self.load_image(filepath)
        self.img_to_gray()
        self.img_to_binary()
        