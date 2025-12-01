import cv2
import numpy as np
import matplotlib.pyplot as plt


from Verfahren.Preprocessing import Preprocessing

class CircleIntersection(Preprocessing):
    def __init__(self):
        super().__init__()
        self.img_distance = None
        self.object_center = None
    
    
    def calculate_distance_transform(self):
        self.img_distance = cv2.distanceTransform(self.img_binary, cv2.DIST_L2, 5)
 
        
    def find_center(self):
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(self.img_distance)
        self.object_center = np.array([int(max_loc[0]), int(max_loc[1])], dtype=np.int32)
        
    
    def find_intersections(self, circle_radius, n_samples=2000, min_segment_len=10):
        """
        Find intersections where the circle overlaps foreground (white) pixels.
        Returns a list of segments [(start_pt, end_pt, mid_pt, indices), ...]
        and stores self.intersection_segments and self.intersection_midpoints.
        """
        if self.object_center is None:
            raise ValueError("Object center not calculated yet.")

        cx, cy = int(self.object_center[0]), int(self.object_center[1])
        h, w = self.img_binary.shape

        thetas = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        coords = np.zeros((n_samples, 2), dtype=int)
        hits = np.zeros(n_samples, dtype=bool)

        for i, theta in enumerate(thetas):
            x = int(round(cx + circle_radius * np.cos(theta)))
            y = int(round(cy + circle_radius * np.sin(theta)))
            coords[i] = (x, y)
            if 0 <= x < w and 0 <= y < h:
                hits[i] = (self.img_binary[y, x] == 1)
            else:
                hits[i] = False

        # Find contiguous true segments on circular array
        segments = []
        i = 0
        visited = 0
        N = n_samples
        while visited < N:
            if hits[i]:
                start = i
                while hits[i]:
                    i = (i + 1) % N
                    visited += 1
                    if visited >= N:
                        break
                end = (i - 1) % N
                segments.append((start, end))
            else:
                i = (i + 1) % N
                visited += 1

        # Merge wrap-around segment if present
        if len(segments) >= 2 and segments[0][0] == 0 and segments[-1][1] == N-1:
            new_start = segments[-1][0]
            new_end = segments[0][1]
            segments = segments[1:-1]
            segments.insert(0, (new_start, new_end))

        # Build detailed segments and filter tiny ones
        detailed = []
        for (s,e) in segments:
            if e >= s:
                indices = np.arange(s, e+1)
            else:
                indices = np.concatenate((np.arange(s, N), np.arange(0, e+1)))
            if len(indices) < min_segment_len:
                continue
            pts = coords[indices]
            start_pt = tuple(pts[0])
            end_pt = tuple(pts[-1])
            mid_pt = tuple(pts[len(pts)//2])
            detailed.append({
                "start_idx": s,
                "end_idx": e,
                "start_pt": start_pt,
                "end_pt": end_pt,
                "mid_pt": mid_pt,
                "indices": indices
            })

        # Save results
        self.intersection_segments = detailed
        self.intersection_midpoints = [tuple(seg["mid_pt"]) for seg in detailed]

        return detailed

    
    def calculate_orientation(self):
        C = np.array(self.object_center, dtype=float)
        
        mids = np.array(self.intersection_midpoints, dtype=float)
        M = mids.mean(axis=0)

        vec = M - C
        angle = np.rad2deg(np.arctan2(vec[1], vec[0])) % 360

        self.orientation_angle = angle
        self.orientation_vector_endpoint = (int(M[0]), int(M[1]))

        return angle
    
        
    def visualize(self, circle_radius):
        if self.object_center is None:
            raise ValueError("Object center not calculated yet.")
        if not hasattr(self, "intersection_segments"):
            raise ValueError("Intersections not computed yet.")

        cx, cy = int(self.object_center[0]), int(self.object_center[1])
        vis = self.img_original.copy()
        
        cv2.circle(vis, (cx, cy), circle_radius, (0,255,0), 16)
        cv2.circle(vis, (cx, cy), 20, (0,0,255), -1)

        for seg in self.intersection_segments:
            cv2.circle(vis, seg["start_pt"], 20, (0,0,255), -1)
            cv2.circle(vis, seg["end_pt"], 20, (0,0,255), -1)
            cv2.circle(vis, seg["mid_pt"], 20, (0,0,255), -1)
            cv2.arrowedLine(vis, (cx, cy), seg["mid_pt"], (0,255,255), 10, tipLength=0.15)

        if len(self.intersection_midpoints) > 0:
            mpx = int(np.mean([p[0] for p in self.intersection_midpoints]))
            mpy = int(np.mean([p[1] for p in self.intersection_midpoints]))
            cv2.circle(vis, (mpx, mpy), 20, (128,0,200), -1)
            cv2.arrowedLine(vis, (cx, cy), (mpx, mpy), (128,0,200), 10, tipLength=0.15)
            
            text = f"Orientation: {self.orientation_angle:.1f}"
            cv2.putText(vis, text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0,0,255), 4, cv2.LINE_AA)

        plt.figure(figsize=(8,8))
        plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

    
    def run(self, filepath, circle_radius=1000):
        self.preprocess(filepath)
        self.calculate_distance_transform()
        self.find_center()
        self.find_intersections(circle_radius)
        orientation = self.calculate_orientation()
        print(f"Calculated Orientation: {orientation} degrees")
        self.visualize(circle_radius)