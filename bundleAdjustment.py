import os
import numpy as np
import json
from scipy.spatial.transform import Rotation as R
from scipy.sparse import lil_matrix
import matplotlib.pyplot as plt
import random
from scipy.optimize import least_squares
import time


def create_initial_estimates(correspondences, poses):
    
    # create mapping from image name to camera ind
    image_2_camera_ind = {
        'img_0' : 0,
        'img_1' : 1
    }

    # each pixel is an observation
    points_2d = np.empty((0, 2))
    # each observation is a observation of a 3D coord
    points_3d = np.empty((0, 3))
    # which camera took each observation
    camera_indices = np.empty((0, ), dtype=int)
    #points = correspondences.values()
    point_indices = np.empty((0, ), dtype=int)
    point_idx = 0 # this will simply increase in the loop. The way i've structured the json makes this easy.

    for point, images in correspondences.items():
        # create matrix of for an estimate of each 3d point position
        x = random.uniform(-10,10); y = random.uniform(0,10)
        z = random.uniform(20,30) 
        points_3d = np.concatenate((points_3d, [[x,y,z]]), axis=0)
        # for each of the images that point occurs add an observation
        for image in images:
            # id of that point (should just be 0,0,1,1,2,2)
            point_indices = np.concatenate((point_indices, [int(point_idx)]), axis=0)
            # now actually adding observation
            pixels = correspondences[point][image]
            points_2d = np.concatenate((points_2d, [pixels]), axis=0)
            camera_indices = np.concatenate((camera_indices, [int(image_2_camera_ind[image])]), axis=0)
            
        point_idx = point_idx + 1

    # sort out camera externals (different for each image)
    camera_params_ext = np.empty((0, 6))
    for camera, pose in poses.items():
        r = R.from_euler('zyx', [pose['heading'], 0, 0], degrees=True)
        rotation_vector = r.as_rotvec()
        translation = np.array([pose['y'], pose['z'], pose['x']])
        camera_param = np.concatenate((rotation_vector, translation))
        camera_params_ext = np.concatenate((camera_params_ext, [camera_param]), axis=0)
   
    # camera internals (same for each image)
    camera_params_int = np.array([4000,0,0])

    print('camera_params_ext: ', camera_params_ext.shape)
    print('camera_params_int: ', camera_params_int.shape)
    print('points_3d: ', points_3d.shape)
    print('camera_indices: ', camera_indices.shape)
    print('points_indices: ', point_indices.shape)
    print('points_2d: ', points_2d.shape)

    return camera_params_ext, camera_params_int, points_3d, camera_indices, point_indices, points_2d



def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    
    Rodrigues' rotation formula is used.
    """
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v



def project(points, camera_params_ext, camera_params_int):
    """Convert 3-D points to 2-D by projecting onto images."""
    points_proj = rotate(points, camera_params_ext[:, :3])
    points_proj += camera_params_ext[:, 3:6]
    points_proj = -points_proj[:, :2] / points_proj[:, 2, np.newaxis]
    f = camera_params_int[0]
    k1 = camera_params_int[1]
    k2 = camera_params_int[2]
    n = np.sum(points_proj**2, axis=1)
    r = 1 + k1 * n + k2 * n**2
    points_proj *= (r * f)[:, np.newaxis]
    return points_proj


def fun(params, n_cameras, n_points, camera_indices, point_indices, points_2d):
    """Compute residuals.
    
    `params` contains camera parameters and 3-D coordinates.
    """
    camera_params_ext = params[:n_cameras * 6].reshape((n_cameras, 6))
    camera_params_int = params[n_cameras * 6 : n_cameras * 6 + 3]
    points_3d = params[(n_cameras * 6 + 3):].reshape((n_points, 3))
    points_proj = project(points_3d[point_indices], camera_params_ext[camera_indices], camera_params_int)
    return (points_proj - points_2d).ravel()


def bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices):
    m = camera_indices.size * 2
    n = n_cameras * 9 + n_points * 3
    A = lil_matrix((m, n), dtype=int)

    i = np.arange(camera_indices.size)
    for s in range(9):
        A[2 * i, camera_indices * 9 + s] = 1
        A[2 * i + 1, camera_indices * 9 + s] = 1

    for s in range(3):
        A[2 * i, n_cameras * 9 + point_indices * 3 + s] = 1
        A[2 * i + 1, n_cameras * 9 + point_indices * 3 + s] = 1

    return A


def main():
    # load correspondences and poses
    with open('data/correspondences.json') as json_file:
        correspondences = json.load(json_file)
    print('loaded {} correspondences'.format(len(correspondences)))

    with open('data/poses.json') as json_file:
        poses = json.load(json_file)
    print('loaded {} camera poses'.format(len(poses)))
    
    camera_params_ext, camera_params_int, points_3d, camera_indices, point_indices, points_2d = create_initial_estimates(correspondences, poses)

    n_cameras = camera_params_ext.shape[0]
    n_points = points_3d.shape[0]

    n = (6 * n_cameras) + 3 + (3 * n_points)
    m = 2 * points_2d.shape[0]

    print("n_cameras: {}".format(n_cameras))
    print("n_points: {}".format(n_points))
    print("Total number of parameters: {}".format(n))
    print("Total number of residuals: {}".format(m))

    x0 = np.hstack((camera_params_ext.ravel(),  camera_params_int.ravel(), points_3d.ravel()))
    f0 = fun(x0, n_cameras, n_points, camera_indices, point_indices, points_2d)
    #plt.plot(f0)
    #plt.show()

    #A = bundle_adjustment_sparsity(n_cameras, n_points, camera_indices, point_indices)
    t0 = time.time()
    res = least_squares(fun, x0, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                        args=(n_cameras, n_points, camera_indices, point_indices, points_2d))
    t1 = time.time()
    print("Optimization took {0:.0f} seconds".format(t1 - t0))
    plt.plot(res.fun)
    plt.show()



if __name__ == "__main__":
    main()