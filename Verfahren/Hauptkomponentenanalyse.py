import cv2
import numpy as np
import matplotlib.pyplot as plt

from Verfahren.Preprocessing import Preprocessing

class PCA(Preprocessing):
    def __init__(self):
        super().__init__()
        self.contour = None
        self.major_axis = None
        self.object_center = None
        self.orientation_deg = None
        self.mean = None
        
    def find_largest_contour(self):
        contours, _ = cv2.findContours(self.img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.contour = max(contours, key=cv2.contourArea)
        
    def principal_component_analysis(self):
        pts = self.contour.reshape(-1, 2).astype(np.float32)
        self.mean = np.mean(pts, axis=0)
        self.object_center = int(self.mean[0]), int(self.mean[1])
        pts_centered = pts - self.mean
        cov = np.cov(pts_centered.T)
        e_vals, e_vecs = np.linalg.eig(cov)

        self.major_axis = e_vecs[:, np.argmax(e_vals)]
     
    def visualize(self):
        cx, cy = self.object_center
        cv2.circle(self.img_original, (cx, cy), 12, (0, 0, 255), -1)

        length = 600
        p1 = (int(cx - self.major_axis[0] * length), int(cy - self.major_axis[1] * length))
        p2 = (int(cx + self.major_axis[0] * length), int(cy + self.major_axis[1] * length))
        cv2.line(self.img_original, p1, p2, (0, 0, 255), 3)

        plt.figure(figsize=(8, 6))
        plt.imshow(self.img_original)
        plt.axis("off")
        plt.title("Objektorientierung (PCA)")
        plt.show()
        
    
    def run(self, filepath):
        self.preprocess(filepath)
        self.find_largest_contour()
        self.principal_component_analysis()
        self.visualize()