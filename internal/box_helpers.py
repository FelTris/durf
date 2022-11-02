
import jax
import jax.numpy as jnp

from internal import math



def ray_box_intersection_inv(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected
    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary
    Args:
        ray_o: Origin of the ray in each box frame, [rays, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified
    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = jnp.ones_like(ray_o) # tf.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = jnp.ones_like(ray_o) * -1. # tf.constant([1., 1., 1.])

    inv_d = jnp.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = jnp.maximum(t_min, t_max)
    t1 = jnp.minimum(t_min, t_max)

    t_near = jnp.minimum(jnp.minimum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = jnp.maximum(jnp.maximum(t1[..., 0], t1[..., 1]), t1[..., 2])

    # Check if rays are inside boxes
    intersection_map = jnp.where(t_near > t_far, 1, 0)

    #return intersection_map, t_far, t_near
    # Check that boxes are in front of the ray origin
    positive_far = jnp.where(t_near * intersection_map > 0, 1, 0)

    #intersection_map = (intersection_map[0][positive_far], intersection_map[1][positive_far])
    intersection_map = intersection_map * positive_far

    if not intersection_map.shape[0] == 0:
        z_ray_in = t_near * intersection_map
        z_ray_out = t_far * intersection_map
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def ray_box_intersection(ray_o, ray_d, aabb_min=None, aabb_max=None):
    """Returns 1-D intersection point along each ray if a ray-box intersection is detected
    If box frames are scaled to vertices between [-1., -1., -1.] and [1., 1., 1.] aabbb is not necessary
    Args:
        ray_o: Origin of the ray in each box frame, [rays, 3]
        ray_d: Unit direction of each ray in each box frame, [rays, 3]
        (aabb_min): Vertex of a 3D bounding box, [-1., -1., -1.] if not specified
        (aabb_max): Vertex of a 3D bounding box, [1., 1., 1.] if not specified
    Returns:
        z_ray_in:
        z_ray_out:
        intersection_map: Maps intersection values in z to their ray-box intersection
    """
    # Source: https://medium.com/@bromanz/another-view-on-the-classic-ray-aabb-intersection-algorithm-for-bvh-traversal-41125138b525
    # https://gamedev.stackexchange.com/questions/18436/most-efficient-aabb-vs-ray-collision-algorithms
    if aabb_min is None:
        aabb_min = jnp.ones_like(ray_o) * -1. # tf.constant([-1., -1., -1.])
    if aabb_max is None:
        aabb_max = jnp.ones_like(ray_o) # tf.constant([1., 1., 1.])

    inv_d = jnp.reciprocal(ray_d)

    t_min = (aabb_min - ray_o) * inv_d
    t_max = (aabb_max - ray_o) * inv_d

    t0 = jnp.minimum(t_min, t_max)
    t1 = jnp.maximum(t_min, t_max)

    t_near = jnp.maximum(jnp.maximum(t0[..., 0], t0[..., 1]), t0[..., 2])
    t_far = jnp.minimum(jnp.minimum(t1[..., 0], t1[..., 1]), t1[..., 2])

    # Check if rays are inside boxes
    intersection_map = jnp.where(t_far > t_near, 1, 0)

    #return intersection_map, t_far, t_near
    # Check that boxes are in front of the ray origin
    positive_far = jnp.where(t_far * intersection_map > 0, 1, 0)

    #intersection_map = (intersection_map[0][positive_far], intersection_map[1][positive_far])
    intersection_map = intersection_map * positive_far

    if not intersection_map.shape[0] == 0:
        z_ray_in = t_near * intersection_map
        z_ray_out = t_far * intersection_map
    else:
        return None, None, None

    return z_ray_in, z_ray_out, intersection_map


def rotate_yaw(p, yaw):
    """Rotates p with yaw in the given coord frame with y being the relevant axis and pointing downwards
    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        yaw: Rotation angle, given in radians, not degrees!!
    Returns:
        p: Rotated points [N_pts, N_frames, N_samples, 3]
    """
    # p of size [batch_rays, n_obj, samples, xyz]
    if len(p.shape) < 4:
        p = p[..., None, :]

    c_y = jnp.cos(yaw)[..., None]
    s_y = jnp.sin(yaw)[..., None]

    p_x = c_y * p[..., 0] + s_y * p[..., 2]
    p_y = p[..., 1]
    p_z = -1.0 * s_y * p[..., 0] + c_y * p[..., 2]

    return jnp.concatenate([p_x[..., None], p_y[..., None], p_z[..., None]], axis=-1)


def aa2matrix_old(angles):
    """This implements going from axis-angle to a rotation matrix representation
        Args:
            angles: rotation around each axis [x, y, z] in radians
        Returns:
            R: 3x3 rotation matrix
    """
    x, y, z = angles[:3]
    skew_r = jnp.array([[0., -z, y],
                        [z, 0., -x],
                        [-y, x, 0.]])
    angles_norm = math.safe_norm(angles) + 1e-12
    R = jnp.eye(3) + (jnp.sin(angles_norm) / angles_norm) * skew_r + \
        ((1 - jnp.cos(angles_norm)) / angles_norm**2) * math.matmul(skew_r, skew_r)
    return R


def aa2matrix(angles):
    """This implements going from axis-angle to a rotation matrix representation
        Args:
            angles: rotation around each axis [x, y, z] in radians [N_frames, 3]
        Returns:
            R: 3x3 rotation matrix [N_frames, 3, 3]
    """
    n_frames = angles.shape[0]
    zero = jnp.zeros_like(angles[:,0])
    skew_v0 = jnp.concatenate([zero[..., None], -angles[:,2:3], angles[:,1:2]], axis=-1)
    skew_v1 = jnp.concatenate([angles[:,2:3], zero[..., None], -angles[:,0:1]], axis=-1)
    skew_v2 = jnp.concatenate([-angles[:,1:2], angles[:,0:1], zero[..., None]], axis=-1)
    skew_r = jnp.concatenate([skew_v0[..., None, :],
                              skew_v1[..., None, :],
                              skew_v2[..., None, :]], axis=-2)
    angles_norm = math.safe_norm(angles) + 1e-12
    eye = jnp.broadcast_to(jnp.eye(3), [n_frames, 3, 3])
    R = eye + (jnp.sin(angles_norm) / angles_norm)[..., None] * skew_r + \
        ((1 - jnp.cos(angles_norm)) / angles_norm**2)[..., None] * math.matmul(skew_r, skew_r)
    return R


def rotate_matrix(p, m):
    """ Rotate p with matrix m in given coord frame
    Args:
        p: 3D points in a given frame [N_pts, N_frames, 3]/[N_pts, N_frames, N_samples, 3]
        m: 3x3 rotation matrix, should be shaped like [N_pts, N_frames, 3, 3]
    Returns:
        p: 3D point rotated by m with the shape [N_pts, N_frames, N_samples, 3]
    """
    if len(p.shape) < 4:
        p = p[..., None, :]
    p = math.matmul(m[..., None, :, :], p[..., None]).reshape(p.shape)
    return p


def scale_frames(p, sc_factor, inverse=False):
    """Scales points given in N_frames in each dimension [xyz] for each frame or rescales for inverse==True
    Args:
        p: Points given in N_frames frames [N_points, N_frames, N_samples, 3]
        sc_factor: Scaling factor for new frame [N_points, N_frames, 3]
        inverse: Inverse scaling if true, bool
    Returns:
        p_scaled: Points given in N_frames rescaled frames [N_points, N_frames, N_samples, 3]
    """
    # Take 150% of bbox to include shadows etc.
    dim = jnp.array([1., 1., 1.]) * sc_factor
    # dim = tf.constant([0.1, 0.1, 0.1]) * sc_factor

    half_dim = dim / 2
    scaling_factor = (1 / (dim + 1e-9))[:, :, None, :]

    if not inverse:
        p_scaled = scaling_factor * p
    else:
        p_scaled = (1/scaling_factor) * p

    return p_scaled


def world2object(pts, dirs, pose, theta_y, dim=None, inverse=False):
    """Transform points given in world frame into N_obj object frames
    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim
    Args:
        pts: N_pts times 3D points given in world frame, [N_pts, 3]
        dirs: Corresponding 3D directions given in world frame, [N_pts, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        inverse: if true pts and dirs should be given in object frame and are transofmed back into world frame, bool
            For inverse: pts, [N_pts, N_obj, 3]; dirs, [N_pts, N_obj, 3]
    Returns:
        pts_w: 3d points transformed into object frame (world frame for inverse task)
        dir_w: unit - 3d directions transformed into object frame (world frame for inverse task)
    """

    #  Prepare args if just one sample per ray-object or world frame only
    if len(pts.shape) == 3:
        # [batch_rays, n_obj, samples, xyz]
        n_sample_per_ray = pts.shape[1]

        pose = jnp.repeat(pose, n_sample_per_ray, axis=0)
        theta_y = jnp.repeat(theta_y, n_sample_per_ray, axis=0)
        if dim is not None:
            dim = jnp.repeat(dim, n_sample_per_ray, axis=0)
        if len(dirs.shape) == 2:
            dirs = jnp.repeat(dirs, n_sample_per_ray, axis=0)

        pts = jnp.reshape(pts, [-1, 3])

    # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    #y_shift = (tf.constant([0., -1., 0.])[tf.newaxis, :] if inverse else
    #           tf.constant([0., -1., 0.])[tf.newaxis, tf.newaxis, :]) * \
    #          (dim[..., 1] / 2)[..., tf.newaxis]
    pose_w = pose #+ y_shift

    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_yaw(-pose_w, theta_y)

    if not inverse:
        N_obj = theta_y.shape[1]
        pts_w = jnp.repeat(pts[:, None, ...], N_obj, axis=1)
        dirs_w = jnp.repeat(dirs[:, None, ...], N_obj, axis=1)

        # Rotate coordinate axis
        pts_o = rotate_yaw(pts_w, theta_y) + t_w_o
        dirs_o = rotate_yaw(dirs_w, theta_y)

        # Scale rays_o_v and rays_d_v for box [[-1.,1], [-1.,1], [-1.,1]]
        if dim is not None:
            pts_o = scale_frames(pts_o, dim)
            dirs_o = scale_frames(dirs_o, dim)

        # Normalize direction
        dirs_o = dirs_o / jnp.linalg.norm(dirs_o, axis=3)[..., None, :]
        return [pts_o.squeeze(axis=-2), dirs_o.squeeze(axis=-2)]

    else:
        pts_o = pts[None, :, None, :]
        dirs_o = dirs
        if dim is not None:
            pts_o = scale_frames(pts_o, dim[None, ...], inverse=True)
            if dirs is not None:
                dirs_o = scale_frames(dirs_o, dim, inverse=True)

        pts_o = pts_o - t_w_o
        pts_w = rotate_yaw(pts_o, -theta_y)[0, :]

        if dirs is not None:
            dirs_w = rotate_yaw(dirs_o, -theta_y)
            # Normalize direction
            dirs_w = dirs_w / math.safe_norm(dirs_w, axis=-1)[..., None, :]
        else:
            dirs_w = None

        return [pts_w, dirs_w]


def world2object_rpy(pts, dirs, pose, rot, dim=None, inverse=False):
    """Transform points given in world frame into N_obj object frames
    Object frames are scaled to [[-1.,1], [-1.,1], [-1.,1]] inside the 3D bounding box given by dim
    Args:
        pts: N_pts times 3D points given in world frame, [N_pts, 3]
        dirs: Corresponding 3D directions given in world frame, [N_pts, 3]
        pose: object position given in world frame, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        theta_y: Yaw of objects around world y axis, [N_pts, N_obj]/if inverse: [N_pts]
        dim: Object bounding box dimensions, [N_pts, N_obj, 3]/if inverse: [N_pts, 3]
        inverse: if true pts and dirs should be given in object frame and are transofmed back into world frame, bool
            For inverse: pts, [N_pts, N_obj, 3]; dirs, [N_pts, N_obj, 3]
    Returns:
        pts_w: 3d points transformed into object frame (world frame for inverse task)
        dir_w: unit - 3d directions transformed into object frame (world frame for inverse task)
    """

    #  Prepare args if just one sample per ray-object or world frame only
    if len(pts.shape) == 3:
        # [batch_rays, n_obj, samples, xyz]
        n_sample_per_ray = pts.shape[1]

        pose = jnp.repeat(pose, n_sample_per_ray, axis=0)
        theta_y = jnp.repeat(rot, n_sample_per_ray, axis=0)
        if dim is not None:
            dim = jnp.repeat(dim, n_sample_per_ray, axis=0)
        if len(dirs.shape) == 2:
            dirs = jnp.repeat(dirs, n_sample_per_ray, axis=0)

        pts = jnp.reshape(pts, [-1, 3])

    # Shift the object reference point to the middle of the bbox (vkitti2 specific)
    #y_shift = (tf.constant([0., -1., 0.])[tf.newaxis, :] if inverse else
    #           tf.constant([0., -1., 0.])[tf.newaxis, tf.newaxis, :]) * \
    #          (dim[..., 1] / 2)[..., tf.newaxis]
    pose_w = pose #+ y_shift

    # Describes the origin of the world system w in the object system o
    t_w_o = rotate_matrix(-pose_w, rot)

    if not inverse:
        N_obj = rot.shape[1]
        pts_w = jnp.repeat(pts[:, None, ...], N_obj, axis=1)
        dirs_w = jnp.repeat(dirs[:, None, ...], N_obj, axis=1)

        # Rotate coordinate axis
        pts_o = rotate_matrix(pts_w, rot) + t_w_o
        dirs_o = rotate_matrix(dirs_w, rot)

        # Scale rays_o_v and rays_d_v for box [[-1.,1], [-1.,1], [-1.,1]]
        if dim is not None:
            pts_o = scale_frames(pts_o, dim)
            dirs_o = scale_frames(dirs_o, dim)

        # Normalize direction
        dirs_o = dirs_o / jnp.linalg.norm(dirs_o, axis=3)[..., None, :]
        return [pts_o.squeeze(axis=-2), dirs_o.squeeze(axis=-2)]

    else:
        pts_o = pts[None, :, None, :]
        dirs_o = dirs
        if dim is not None:
            pts_o = scale_frames(pts_o, dim[None, ...], inverse=True)
            if dirs is not None:
                dirs_o = scale_frames(dirs_o, dim, inverse=True)

        pts_o = pts_o - t_w_o
        pts_w = rotate_yaw(pts_o, -theta_y)[0, :]

        if dirs is not None:
            dirs_w = rotate_yaw(dirs_o, -theta_y)
            # Normalize direction
            dirs_w = dirs_w / math.safe_norm(dirs_w, axis=-1)[..., None, :]
        else:
            dirs_w = None

        return [pts_w, dirs_w]
