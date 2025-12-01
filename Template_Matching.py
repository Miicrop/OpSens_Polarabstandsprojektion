import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

class TemplateRotationMatcher:
    
    def __init__(self):
        self.best_angle = 0.0
        self.min_difference = float('inf')
        
    def load_and_preprocess(self, filepath):
        img = cv.imread(filepath)
        if img is None: raise ValueError(f"Bild nicht gefunden: {filepath}")
        
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        blur = cv.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv.threshold(blur, 100, 255, cv.THRESH_BINARY_INV)
        
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        if not contours: raise ValueError("Keine Kontur gefunden.")
        main_contour = max(contours, key=cv.contourArea)
        
        M = cv.moments(main_contour)
        if M["m00"] == 0: cx, cy = 0, 0
        else: cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            
        return img, binary, (cx, cy)

    def center_object(self, binary_img, center, canvas_size=(800, 800)):
        cx, cy = center
        h, w = binary_img.shape
        
        tx = canvas_size[0]//2 - cx
        ty = canvas_size[1]//2 - cy
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        
        centered = cv.warpAffine(binary_img, M, canvas_size)
        return centered

    def rotate_image(self, image, angle):
        h, w = image.shape
        center = (w // 2, h // 2)
        
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(image, M, (w, h))
        return rotated

    def calculate_best_angle(self, canvas_ref, canvas_meas):
        best_angle = 0.0
        min_difference = float('inf')
        
        scores = []
        angles = np.arange(-180, 180, 0.5) 
        
        for angle in angles:
            rotated_meas = self.rotate_image(canvas_meas, angle)

            diff = cv.bitwise_xor(canvas_ref, rotated_meas)
            score = np.sum(diff) 
            
            scores.append(score)
            
            if score < min_difference:
                min_difference = score
                best_angle = -angle 
        if best_angle < 0:
            best_angle += 360 
        return best_angle, min_difference

    def find_best_rotation_multiple(self, ref_path, measure_paths):

        print(f"Lade Referenz: {ref_path}")
        img_ref_raw, bin_ref, com_ref = self.load_and_preprocess(ref_path)
        canvas_ref = self.center_object(bin_ref, com_ref)
        
        results = {}
        successful_measurements = []
        
        for measure_path in measure_paths:
            print(f"\nVerarbeite Messung: {measure_path}")
            
            try:
                img_meas_raw, bin_meas, com_meas = self.load_and_preprocess(measure_path)
                canvas_meas = self.center_object(bin_meas, com_meas)
                
                print("Starte Matching (0,5°-Schritte)")
                best_angle, min_difference = self.calculate_best_angle(canvas_ref, canvas_meas)
                
                print(f"Beste Matching bei {best_angle:.2f}° (Score: {min_difference:.0f})")
                
                results[measure_path] = {
                    "angle": best_angle,
                    "difference": min_difference,
                }
                successful_measurements.append((img_meas_raw, com_meas, best_angle, measure_path))
                
            except ValueError as e:
                print(f"FEHLER bei {measure_path}: {e}")
                results[measure_path] = {"angle": None, "difference": None, "error": str(e)}
                continue
            
        self.visualize_all_results(img_ref_raw, com_ref, successful_measurements)
            
        return results

    def visualize_all_results(self, img_ref, com_ref, successful_measurements):
        
        num_plots = 1 + len(successful_measurements) 
        cols = 3 
        rows = int(np.ceil(num_plots / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        
        if rows * cols == 1:
            axes = np.array([axes])
        else:
            axes = axes.flatten()
            
        length = 150 
        ax = axes[0]
        ax.imshow(cv.cvtColor(img_ref, cv.COLOR_BGR2RGB))
        ax.plot(com_ref[0], com_ref[1], 'ro', markersize=8)
        ax.arrow(com_ref[0], com_ref[1], length, 0, color='red', width=5, head_width=20)
        ax.set_title("Referenz (0°)")
        ax.axis('off')

        for i, (img_meas, com_meas, angle, path) in enumerate(successful_measurements, start=1):
            ax = axes[i]
            
            ax.imshow(cv.cvtColor(img_meas, cv.COLOR_BGR2RGB))
            ax.plot(com_meas[0], com_meas[1], 'ro', markersize=8)
            
            angle_rad = np.radians(angle)
            dx = length * np.cos(angle_rad)
            dy = length * np.sin(angle_rad) * -1 
            
            ax.arrow(com_meas[0], com_meas[1], dx, dy, color='blue', width=5, head_width=20)
            
            filename = path.split('/')[-1]
            ax.set_title(f"{filename}\n({angle:.2f}°)")
            ax.axis('off')

        for i in range(num_plots, rows * cols):
            fig.delaxes(axes[i])
            
        fig.suptitle("Ergebnis der Rotationserkennung", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

if __name__ == "__main__":
    REF_FILE = "coffee_filter_0deg.jpeg"
    MEAS_FILES = [
        "coffee_filter_1deg.jpeg", 
        "coffee_filter_3.png",
        "coffee_filter_4.png",
        "coffee_filter_5.jpeg"
    ]
    matcher = TemplateRotationMatcher()
    
    all_results = matcher.find_best_rotation_multiple(REF_FILE, MEAS_FILES)
    '''
    print("\n\n--- Zusammenfassung der Ergebnisse ---")
    print(f"| {'Dateiname':<30} | {'Winkel':>7} | {'Score (Diff)':>10} |")
    print("-" * 55)
    for file, res in all_results.items():
        if res.get("angle") is not None:
            print(f"| {file:<30} | {res['angle']:>7.2f}° | {res['difference']:>10.0f} |")
        else:
            print(f"| {file:<30} | {'FEHLER':>7} | {res.get('error', 'Unbekannt'):<10} |")
    
    '''
