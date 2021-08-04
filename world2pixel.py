import numpy as np

def world2camera(worldcoors, camera_pos, camera_view, camera_up):
    l2wMatrix = np.zeros((4,4), dtype=np.float32)
    camera_pos = np.array(camera_pos)
    camera_view = np.array(camera_view)
    camera_up = np.array(camera_up)

    l2wMatrix[3, 0:3] = camera_pos[0:3]
    l2wMatrix[3, 3] = 1

    camera_view = camera_view / np.linalg.norm(camera_view)
    l2wMatrix[2, 0:3] = camera_view[0:3]

    x = np.cross(camera_view, camera_up)
    x = x / np.linalg.norm(x)
    l2wMatrix[0, 0:3] = x[0:3]

    y = np.cross(camera_view, x)
    y = y / np.linalg.norm(y)
    l2wMatrix[1, 0:3] = y[0:3]

    w2lMatrix = np.linalg.inv(l2wMatrix)
    worldcoors.append(1)
    worldcoors = np.array(worldcoors)
    localcoors = np.matmul(worldcoors, w2lMatrix)
    localcoorslist = list(localcoors)
    
    return localcoorslist[0:3]

def camera2pixel(cameracoors, camera_intrinsics):
    if cameracoors[2] < 0:
        x = -1
        y = -1
    else:
        x = round((cameracoors[0] * camera_intrinsics[0, 0]) / cameracoors[2] + camera_intrinsics[0, 2])
        y = round((cameracoors[1] * camera_intrinsics[1, 1]) / cameracoors[2] + camera_intrinsics[1, 2])

    return [x, y]

def world2pixel(worldcoors, camera_pos, camera_view, camera_up, camera_intrinsics):
    cameracoors = world2camera(worldcoors, camera_pos, camera_view, camera_up)
    pixelcoors = camera2pixel(cameracoors, camera_intrinsics)

    return pixelcoors
