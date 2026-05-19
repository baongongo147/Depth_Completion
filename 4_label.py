import cv2
import numpy as np
import os
import json
import glob

import common

# --- CONFIGURATION ---
INPUT_DIR = "lidar_image_pairs"
OUTPUT_DIR = "lidar_image_pair_labeled"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


# class Labeler:
#     def __init__(self):
#         self.image_pts = []
#         self.radar_pts = [] # Will store (rx, ry) pixels
#         self.img_w = 0
#         self.combined_view = None
#         self.cx, self.cy = 0, 0 # Radar center

#     def mouse_callback(self, event, x, y, flags, param):
#         if event == cv2.EVENT_LBUTTONDOWN:
#             if x < self.img_w:
#                 if len(self.image_pts) < 3:
#                     self.image_pts.append((x, y))
#             else:
#                 if len(self.radar_pts) < 3:
#                     self.radar_pts.append((x - self.img_w, y))
#             self.redraw()

#     def redraw(self):
#         temp_view = self.combined_view.copy()
#         for i, pt in enumerate(self.image_pts):
#             cv2.circle(temp_view, pt, 5, (0, 255, 0), -1)
#         for i, pt in enumerate(self.radar_pts):
#             rpt = (pt[0] + self.img_w, pt[1])
#             cv2.circle(temp_view, rpt, 5, (0, 0, 255), -1)
#         cv2.imshow("Labeling Tool", temp_view)


#GLOBAL STATE
lidar_data_pts = []
lidar_pts = []

image = None
image_pts = []

def draw_lidar(lidar_points_world_space, opt_lidar_pts = None):
    radar_view = np.zeros((common.LIDAR_VIEW_SIZE[0], common.LIDAR_VIEW_SIZE[1], 3), dtype=np.uint8)
    radar_ruler_color = (50, 50, 50)
    cx, cy = common.LIDAR_VIEW_SIZE[0] // 2, common.LIDAR_VIEW_SIZE[1] // 2
    cv2.line(radar_view, (0, cy), (common.LIDAR_VIEW_SIZE[0], cy), radar_ruler_color, 1)
    cv2.line(radar_view, (cx, 0), (cx, common.LIDAR_VIEW_SIZE[1]), radar_ruler_color, 1)
    cv2.putText(radar_view, "-z", (0, cy), cv2.FONT_HERSHEY_COMPLEX, 1.0,  radar_ruler_color)
    cv2.putText(radar_view, "+z", (common.LIDAR_VIEW_SIZE[0] - 40, cy), cv2.FONT_HERSHEY_COMPLEX, 1.0,  radar_ruler_color)

    cv2.putText(radar_view, "+x", (cx, 40), cv2.FONT_HERSHEY_COMPLEX, 1.0,  radar_ruler_color)
    cv2.putText(radar_view, "-x", (cx, common.LIDAR_VIEW_SIZE[1]), cv2.FONT_HERSHEY_COMPLEX, 1.0,  radar_ruler_color)
    
    for pt in lidar_points_world_space:
        lx, lz = pt[0], pt[2]
        u, v = common.lidarWorldToLidarView(lx / 1000.0, lz / 1000.0)
        if 0 <= u < common.LIDAR_VIEW_SIZE[0] and 0 <= v < common.LIDAR_VIEW_SIZE[1]:
            text = f"{lx:.2f} | {lz:.2f}"
            #cv2.putText(radar_view, text, (u, v), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255))
            cv2.circle(radar_view, (u, v), 1, (255, 255, 255), -1)

    if opt_lidar_pts != None:
        for pt in opt_lidar_pts:
            lx, lz = pt[0], pt[2] #Millimeters
            u, v = common.lidarWorldToLidarView(lx / 1000.0, lz / 1000.0)
            if 0 <= u < common.LIDAR_VIEW_SIZE[0] and 0 <= v < common.LIDAR_VIEW_SIZE[1]:
                text = f"{lx:.2f} | {lz:.2f}"
                cv2.circle(radar_view, (u, v), 2, (255, 0, 0), -1)


    cv2.imshow("Labeling Tool(Lidar)", radar_view)

def lidar_mouse_callback(event, x, y, flags, param):
    global lidar_data_pts
    global lidar_pts
    if event == cv2.EVENT_LBUTTONDOWN:
        #Convert lidar view pos to lidar world pos
        lx, lz = common.lidarViewToLidarWorld(x, y) 
        #Convert to millimeters
        lx *= 1000.0
        lz *= 1000.0
        print(f"Lidar points captured {lx} | {lz}")
        lidar_pts.append([lx, 0, lz])
        refresh(lidar_data_pts, image, lidar_pts, image_pts)
    pass

def image_mouse_callback(event, x, y, flags, param):
    global image
    global image_pts

    if event == cv2.EVENT_LBUTTONDOWN:
        image_pts.append([x, y])
        print(f"Image points captured {x} | {y}")
        refresh(lidar_data_pts, image, lidar_pts, image_pts)
    pass

def refresh(lidar_points_world_space, image, opt_lidar_pts = None, opt_image_pts = None):
    draw_lidar(lidar_points_world_space, opt_lidar_pts)
    image_temp = image.copy()

    if opt_image_pts != None:
        for pt in opt_image_pts:
            cv2.circle(image_temp, (pt[0], pt[1]), 2, (0, 255, 0), -1)
    cv2.imshow("Labeling Tool(Image)", image_temp)


def main():
    global lidar_data_pts
    global image
    global image_pts
    global lidar_pts

    cv2.namedWindow("Labeling Tool(Lidar)")
    cv2.namedWindow("Labeling Tool(Image)")
    cv2.setMouseCallback("Labeling Tool(Lidar)", lidar_mouse_callback)
    cv2.setMouseCallback("Labeling Tool(Image)", image_mouse_callback)

    img_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")))

    for img_path in img_files:
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        json_path = os.path.join(INPUT_DIR, f"{base_name}.json")
        if not os.path.exists(json_path): continue

        lidar_data = None
        with open(json_path, 'r') as f:
            lidar_data = json.load(f)
        if lidar_data == None:
            continue
        lidar_data_pts = lidar_data['lidar_points']

        #DRAW LIDAR
        draw_lidar(lidar_data_pts)

        #DRAW RGB
        image = cv2.imread(img_path)
        cv2.imshow("Labeling Tool(Image)", image)
        
        print(f"--- {base_name} ---")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                if len(image_pts) != len(lidar_pts):
                    print(f"Warning: So luong diem lidar != so luong diem image !")
                output_json = {}
                output_json["labels"] = {
                    "image_pixel_points": image_pts,
                    "lidar_3d_points_millimeters": lidar_pts
                }
                file_name = os.path.join(OUTPUT_DIR, f"{base_name}.json")
                print(f"Saved Label {file_name}")
                with open(file_name, 'w') as f:
                    json.dump(output_json, f, indent=4)
                image_pts = []
                lidar_pts = []
                break
            elif key == ord('r'):
                image_pts = []
                lidar_pts = []
                refresh(lidar_data_pts, image)
            elif key == ord('q'):
                return

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()