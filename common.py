
import math

CAMERA_BUFFER_SIZE = (512, 256)
LIDAR_VIEW_SIZE = (800, 800)
LIDAR_VIEW_SCALE = 300
CHECKERBOARD_SIZE = (14, 14) 
CHECKERBOARD_SQUARE_SIZE = 12.0  # mm



def calculate_fov(mtx):
    fx, fy = mtx[0, 0], mtx[1, 1]
    fov_h = 2 * math.atan(CAMERA_BUFFER_SIZE[0] / (2 * fx)) * (180 / math.pi)
    fov_v = 2 * math.atan(CAMERA_BUFFER_SIZE[1] / (2 * fy)) * (180 / math.pi)
    return fov_h, fov_v

#Input: angle(degree), distance(millimeters)
#Output: lx, lz of lidar(millimeters)
def angleDistanceToLidarXZ(angle, distance):
    angle_rad = math.radians(angle)

    lx : float = (distance) * math.sin(angle_rad) 
    lz : float = (distance) * math.cos(angle_rad)
    return lx , lz

def lidarWorldToLidarView(lx, lz):
    cx, cy = LIDAR_VIEW_SIZE[0] // 2, LIDAR_VIEW_SIZE[1] // 2
    u = int(cx + (lz * LIDAR_VIEW_SCALE)) 
    v = int(cy - (lx * LIDAR_VIEW_SCALE))
    return u, v

def lidarViewToLidarWorld(u, v):
    cx, cy = LIDAR_VIEW_SIZE[0] // 2, LIDAR_VIEW_SIZE[1] // 2
    lz = (u - cx) / LIDAR_VIEW_SCALE
    lx = (cy - v) / LIDAR_VIEW_SCALE
    return lx, lz

def imageViewToLidarWorld(u, v):
    lx = 0.0
    lz = 0.0


    return lx, lz