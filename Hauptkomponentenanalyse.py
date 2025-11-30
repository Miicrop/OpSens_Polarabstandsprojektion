import cv2
import numpy as np
import matplotlib.pyplot as plt

class PCA:
    def __init__(self):
        self.img_original = None
        self.img_gray = None
        self.img_binary = None
        self.img_distance = None
        self.contour = None
        self.major_axis = None
        self.object_center = None
        self.orientation_deg = None
        self.mean = None
        
        
    def load_image(self, filepath):
        self.img_original = cv2.imread("OpSens_Polarabstandsprojektion/"+filepath)
    
    def img_to_gray(self):
        self.img_gray = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
        
    def img_to_binary(self):
        _, self.img_binary = cv2.threshold(self.img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
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
        
        
        
if __name__ == "__main__":
    pca = PCA()
    pca.load_image("Ei2.png")
    pca.img_to_gray()
    pca.img_to_binary()
    pca.find_largest_contour()
    pca.principal_component_analysis()
    pca.visualize()
