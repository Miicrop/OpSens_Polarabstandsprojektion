import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button

from Verfahren.Preprocessing import Preprocessing

class PolarDistanceProjection(Preprocessing):
    def __init__(self):
        super().__init__()
        self.img_distance = None
        self.object_center = None
        self.thetas = None
        self.radii = None
        self.dominant_frequency = None
        self.phase = None
        self.orientation_deg = None
 
        
    def calculate_distance_transform(self):
        self.img_distance = cv2.distanceTransform(self.img_binary, cv2.DIST_L2, 5)
 
        
    def find_center(self):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.img_distance)
        self.object_center = np.array([int(max_loc[0]), int(max_loc[1])], dtype=np.int32)
  
        
    def polar_projection(self, n_angles=360, max_radius=2000):
        h, w = self.img_binary.shape
        self.thetas = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
        self.radii = np.zeros(n_angles)

        for i, theta in enumerate(self.thetas):
            dx = np.cos(theta)
            dy = np.sin(theta)

            for r in range(0, max_radius):
                x = int(self.object_center[0] + r*dx)
                y = int(self.object_center[1] + r*dy)

                if x < 0 or x >= w or y < 0 or y >= h:
                    self.radii[i] = r
                    break

                if self.img_binary[y, x] == 0:
                    self.radii[i] = r
                    break
 
                
    def smooth_radii(self, kernel_size=7):
        self.radii = np.convolve(self.radii, np.ones(kernel_size)/kernel_size, mode='same')
  
        
    def analyze_frequencies(self, methdod="fft"):
        if methdod == "peak":
            idx = np.argmax(self.radii)
            angle = np.rad2deg(self.thetas[idx]) % 360

            self.dominant_frequency = None
            self.phase = None
            self.orientation_deg = angle
            return
        
        r = self.radii - np.mean(self.radii)
        F = np.fft.fft(r)

        absolute_frequency_amplitudes = np.abs(F[:len(F)//2])
        absolute_frequency_amplitudes[0] = 0

        self.dominant_frequency = np.argmax(absolute_frequency_amplitudes)
        self.phase = np.angle(F[self.dominant_frequency])
        orientation = -self.phase / self.dominant_frequency
        self.orientation_deg = np.rad2deg(orientation) % 360
  
        
    def print_results(self):
        print(f"Dominante Frequenz k = {self.dominant_frequency}")
        print(f"Geschätzte Orientierung = {self.orientation_deg:.2f}°")
        
        
    def visualize(self):
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 4, figure=fig)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title("Original Image")
        ax1.imshow(cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB))

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title("Binary Image")
        ax2.imshow(self.img_binary, cmap='gray')
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title("Distance Transform")
        ax3.imshow(self.img_distance, cmap='gray')
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[0, 3])
        ax4.set_title("Distance Transform")
        ax4.imshow(self.img_distance, cmap='jet')
        ax4.axis('off')

        ax5 = fig.add_subplot(gs[1, 0])
        ax5.set_title("Object Center")
        img_centered = self.img_original.copy()
        cv2.circle(img_centered, tuple(self.object_center), 20, (0, 0, 255), -1)
        cv2.putText(
            img_centered,
            f"Center: {self.object_center[0]}, {self.object_center[1]}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            4.0, (0, 0, 255), 4
        )
        ax5.imshow(cv2.cvtColor(img_centered, cv2.COLOR_BGR2RGB))
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[1, 1])
        ax6.set_title("Orientation Line")
        img_oriented = self.img_original.copy()
        length = 300
        cx, cy = self.object_center
        if self.dominant_frequency is None:
            angle_rad = np.deg2rad(self.orientation_deg)
        else:
            angle_rad = -self.phase / self.dominant_frequency
        x2 = int(cx + length * np.cos(angle_rad))
        y2 = int(cy + length * np.sin(angle_rad))
        cv2.arrowedLine(img_oriented, (cx, cy), (x2, y2), (0, 0, 255), 20)
        cv2.putText(
            img_oriented,
            f"Orientation: {self.orientation_deg:.1f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            4.0, (0, 0, 255), 4
        )
        ax6.imshow(cv2.cvtColor(img_oriented, cv2.COLOR_BGR2RGB))
        ax6.axis('off')

        ax7 = fig.add_subplot(gs[1, 2:4])
        ax7.set_title("Polar Distance Projection")
        ax7.plot(np.rad2deg(self.thetas), self.radii)
        ax7.axvline(self.orientation_deg, color='r', linewidth=2)
        ax7.set_xlabel("Angle (degrees)")
        ax7.set_ylabel("Radius")
        ax7.grid()

        plt.show()
   
        
    def animate_live(self, interval=1):

        if self.radii is None or self.thetas is None:
            raise ValueError("Bitte vorher polar_projection() ausführen.")

        img_rgb = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
        img_bin = self.img_binary

        h, w = img_bin.shape
        cx, cy = self.object_center

        thetas = self.thetas
        radii = self.radii.copy()

        max_r = np.max(radii) * 1.1

        fig, (ax_img, ax_plot) = plt.subplots(1, 2, figsize=(16, 7))
        plt.subplots_adjust(bottom=0.2)  # Platz für Buttons

        ax_img.set_title("Sampling")
        ax_img.imshow(img_rgb)
        ax_img.plot(cx, cy, "ro")
        ax_img.axis("off")

        line_beam, = ax_img.plot([], [], "g-", linewidth=3)
        dot_end, = ax_img.plot([], [], "bo", markersize=10)

        ax_plot.set_title("Polar Distance Projection")
        ax_plot.set_xlim(0, 360)
        ax_plot.set_ylim(0, max_r)
        ax_plot.set_xlabel("Angle [°]")
        ax_plot.set_ylabel("Radius")
        ax_plot.grid()

        line_live, = ax_plot.plot([], [], "r-")

        anim_running = False
        frame = 0

        def update(_):
            nonlocal frame

            if not anim_running:
                return line_beam, dot_end, line_live

            if frame >= len(thetas):
                return line_beam, dot_end, line_live

            theta = thetas[frame]
            r = radii[frame]

            dx, dy = np.cos(theta), np.sin(theta)
            x2 = cx + r * dx
            y2 = cy + r * dy

            line_beam.set_data([cx, x2], [cy, y2])
            dot_end.set_data([x2], [y2])

            angles_deg = np.rad2deg(thetas[:frame+1])
            line_live.set_data(angles_deg, radii[:frame+1])

            frame += 1
            return line_beam, dot_end, line_live

        anim = FuncAnimation(
            fig,
            update,
            interval=interval,
            blit=True
        )

        ax_button_start = plt.axes([0.3, 0.05, 0.15, 0.07])
        btn_start = Button(ax_button_start, 'Start / Pause')

        ax_button_reset = plt.axes([0.55, 0.05, 0.15, 0.07])
        btn_reset = Button(ax_button_reset, 'Restart')

        def start_pause(event):
            nonlocal anim_running
            anim_running = not anim_running

        def restart(event):
            nonlocal frame, anim_running
            frame = 0
            anim_running = False
            line_beam.set_data([], [])
            dot_end.set_data([], [])
            line_live.set_data([], [])
            fig.canvas.draw_idle()

        btn_start.on_clicked(start_pause)
        btn_reset.on_clicked(restart)

        plt.show()
    
        
    def run(self, img_path):
        self.preprocess(img_path)
        self.calculate_distance_transform()
        self.find_center()
        self.polar_projection()
        self.smooth_radii()
        self.analyze_frequencies("peak")
        self.print_results()
        self.visualize()
        self.animate_live()