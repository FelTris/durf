import json
import os
import random
from os import path
import queue
import threading
import cv2
import jax
import numpy as np
from PIL import Image
from internal import utils
from natsort import natsorted
from scipy.spatial.transform import Rotation as R



def get_dataset(split, train_dir, config):
    return dataset_dict[config.dataset_loader](split, train_dir, config)


def convert_to_ndc(origins, directions, focal, w, h, near=1.):
    """Convert a set of rays to NDC coordinates."""
    # Shift ray origins to near plane
    t = -(near + origins[..., 2]) / directions[..., 2]
    origins = origins + t[..., None] * directions

    dx, dy, dz = tuple(np.moveaxis(directions, -1, 0))
    ox, oy, oz = tuple(np.moveaxis(origins, -1, 0))

    # Projection
    o0 = -((2 * focal) / w) * (ox / oz)
    o1 = -((2 * focal) / h) * (oy / oz)
    o2 = 1 + 2 * near / oz

    d0 = -((2 * focal) / w) * (dx / dz - ox / oz)
    d1 = -((2 * focal) / h) * (dy / dz - oy / oz)
    d2 = -2 * near / oz

    origins = np.stack([o0, o1, o2], -1)
    directions = np.stack([d0, d1, d2], -1)
    return origins, directions


class Dataset(threading.Thread):
    """Dataset Base Class."""

    def __init__(self, split, data_dir, config):
        super(Dataset, self).__init__()
        self.queue = queue.Queue(3)  # Set prefetch buffer to 3 batches.
        self.daemon = True
        self.split = split
        self.data_dir = data_dir
        self.near = config.near
        self.far = config.far
        if split == 'train':
            self._train_init(config)
        elif split == 'test':
            self._test_init(config)
        elif split == 'render':
            self._test_init(config)
        else:
            raise ValueError(
                'the split argument should be either \'train\' or \'test\', set'
                'to {} here.'.format(split))
        self.batch_size = config.batch_size // jax.host_count()
        self.batching = config.batching
        self.c2f_steps = config.c2f_steps
        self.render_path = config.render_path
        self.start()

    def __iter__(self):
        return self

    def __next__(self):
        """Get the next training batch or test example.

        Returns:
          batch: dict, has 'pixels' and 'rays'.
        """
        x = self.queue.get()
        if self.split == 'train':
            return utils.shard(x)
        else:
            return utils.to_device(self._next_test())

    def peek(self):
        """Peek at the next training batch or test example without dequeuing it.

        Returns:
          batch: dict, has 'pixels' and 'rays'.
        """
        x = self.queue.queue[0].copy()  # Make a copy of the front of the queue.
        if self.split == 'train':
            return utils.shard(x)
        else:
            return utils.to_device(x)

    def run(self):
        if self.split == 'train':
            next_func = self._next_train
        else:
            next_func = self._next_test
        while True:
            self.queue.put(next_func())

    @property
    def size(self):
        return self.n_examples

    def _train_init(self, config):
        """Initialize training."""
        self._load_renderings(config)
        self._generate_rays()

        if config.batching == 'all_images':
            # flatten the ray and image dimension together.
            self.images = self.images.reshape([-1, 3])
            self.rays = utils.namedtuple_map(lambda r: r.reshape([-1, r.shape[-1]]),
                                             self.rays)
        elif config.batching == 'single_image':
            self.images = self.images.reshape([-1, self.resolution, 3])
            self.rays = utils.namedtuple_map(
                lambda r: r.reshape([-1, self.resolution, r.shape[-1]]), self.rays)
        else:
            raise NotImplementedError(
                f'{config.batching} batching strategy is not implemented.')

    def _test_init(self, config):
        self._load_renderings(config)
        self._generate_rays()
        self.it = 0

    def _next_train(self):
        """Sample next training batch."""

        if self.batching == 'all_images':
            ray_indices = np.random.randint(0, self.rays[0].shape[0],
                                            (self.batch_size,))
            batch_pixels = self.images[ray_indices]
            batch_rays = utils.namedtuple_map(lambda r: r[ray_indices], self.rays)
        elif self.batching == 'single_image':
            image_index = np.random.randint(0, self.n_examples, ())
            ray_indices = np.random.randint(0, self.rays[0][0].shape[0],
                                            (self.batch_size,))
            batch_pixels = self.images[image_index][ray_indices]
            batch_rays = utils.namedtuple_map(lambda r: r[image_index][ray_indices],
                                              self.rays)
        else:
            raise NotImplementedError(
                f'{self.batching} batching strategy is not implemented.')

        return {'pixels': batch_pixels, 'rays': batch_rays}

    def _next_test(self):
        """Sample next test example."""
        idx = self.it
        self.it = (self.it + 1) % self.n_examples

        if self.render_path:
            return {'rays': utils.namedtuple_map(lambda r: r[idx], self.render_rays)}
        else:
            return {
                'pixels': self.images[idx],
                'rays': utils.namedtuple_map(lambda r: r[idx], self.rays)
            }

    # TODO(bydeng): Swap this function with a more flexible camera model.
    def _generate_rays(self):
        """Generating rays for all images."""
        x, y = np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
            np.arange(self.w, dtype=np.float32),  # X-Axis (columns)
            np.arange(self.h, dtype=np.float32),  # Y-Axis (rows)
            indexing='xy')
        camera_dirs = np.stack(
            [(x - self.w * 0.5 + 0.5) / self.focal,
             -(y - self.h * 0.5 + 0.5) / self.focal, -np.ones_like(x)],
            axis=-1)

        directions = ((camera_dirs[None, ..., None, :] *
                       self.camtoworlds[:, None, None, :3, :3]).sum(axis=-1))
        origins = np.broadcast_to(self.camtoworlds[:, None, None, :3, -1],
                                  directions.shape)
        viewdirs = directions / np.linalg.norm(directions, axis=-1, keepdims=True)

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = np.sqrt(
            np.sum((directions[:, :-1, :, :] - directions[:, 1:, :, :]) ** 2, -1))
        dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.

        radii = dx[..., None] * 2 / np.sqrt(12)

        ones = np.ones_like(origins[..., :1])
        self.rays = utils.Rays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=ones * self.near,
            far=ones * self.far)


class Carla(Dataset):
    """CARLA Dataset."""

    def _train_init(self, config):
        """Initialize training."""
        np.random.seed(20201473)  # TODO: do i need to put this here or is in train enough?
        self._load_renderings(config)
        self._generate_rays_multi(self.w, self.h, self.focal, factor=4)
        self._generate_rays_multi(self.w_8, self.h_8, self.focal_8, factor=8)
        self._generate_rays_multi(self.w_12, self.h_12, self.focal_12, factor=12)
        self._generate_rays_multi(self.w_16, self.h_16, self.focal_16, factor=16)
        self.train_it = 0

        if config.batching == 'all_images':
            def flatten(x):
                # Always flatten out the height x width dimensions
                x = [y.reshape([-1, y.shape[-1]]) for y in x]
                if config.batching == 'all_images':
                    # If global batching, also concatenate all data into one list
                    x = np.concatenate(x, axis=0)
                return x

            self.images = flatten(self.images)
            self.depth = flatten(self.depth)
            self.sky_mask = flatten(self.sky_mask)
            self.rays = utils.namedtuple_map(flatten, self.rays)
            print(self.images.shape, self.depth.shape, self.sky_mask.shape)
        elif config.batching == 'single_image' or config.batching == 'single_image_consecutive':
            for i in range(len(self.images)):
                self.images[i] = self.images[i].reshape((int(self.resolution[i]), 3))
                self.depth[i] = self.depth[i].reshape((int(self.resolution[i]), 1))
                self.sky_mask[i] = self.sky_mask[i].reshape((int(self.resolution[i]), 1))
                for j in range(len(self.rays)):
                    self.rays[j][i] = self.rays[j][i].reshape((int(self.resolution[i]), -1))
        elif config.batching == 'timestep':

            def flatten_time(x):
                # flatten batches along timestep
                xt = []
                for y in x:
                    if y.shape[-1] == 4:
                        xt.append(y.reshape([-1, y.shape[-1], y.shape[-1]]))
                    else:
                        xt.append(y.reshape([-1, y.shape[-1]]))
                x = xt
                _, ind = np.unique(self.timesteps, return_counts=True)
                out = []
                for i in range(len(ind)):
                    if i == 0:
                        out.append(np.concatenate(x[:ind[i]], axis=0))
                        ind[i+1] += ind[i]
                    else:
                        out.append(np.concatenate(x[ind[i-1]:ind[i]]))
                        if i+1 < len(ind):
                            ind[i+1] += ind[i]
                return out

            for key in self.images:
                self.images[key] = flatten_time(self.images[key])
                self.depth[key] = flatten_time(self.depth[key])
                self.sky_mask[key] = flatten_time(self.sky_mask[key])
            self.masks2d = flatten_time(self.masks2d)
            #self.dyn_masks = flatten_time(self.dyn_masks)
            for keys in self.rays:
                self.rays[keys] = utils.namedtuple_map(flatten_time, self.rays[keys])

    def _next_train(self):
        """Sample next training batch."""

        if self.batching == 'all_images':
            ray_indices = np.random.randint(0, self.rays[0].shape[0],
                                            (self.batch_size,))
            batch_pixels = self.images[ray_indices]
            batch_depth = self.depth[ray_indices]
            batch_sky = self.sky_mask[ray_indices]
            batch_rays = utils.namedtuple_map(lambda r: r[ray_indices], self.rays)

            #ind_i = np.broadcast_to(ray_indices, (self.batch_size, self.batch_size))
            #ray_diff = np.abs(ind_i - ind_i.T)
            #neighbour_mask = (ray_diff == 1.0).astype(np.float32)

        elif self.batching == 'single_image':
            image_index = np.random.randint(0, self.n_examples, ())
            ray_indices = np.random.randint(0, self.rays[0][image_index].shape[0],
                                            (self.batch_size,))
            batch_pixels = self.images[image_index][ray_indices]
            batch_depth = self.depth[image_index][ray_indices]
            batch_sky = self.sky_mask[image_index][ray_indices]
            batch_rays = utils.namedtuple_map(lambda r: r[image_index][ray_indices],
                                              self.rays)

        elif self.batching == 'single_image_consecutive':
            image_index = np.random.randint(0, self.n_examples, ())
            ray_indices = np.random.randint(0, self.rays[0][image_index].shape[0] - self.batch_size)
            batch_pixels = self.images[image_index][ray_indices:ray_indices+self.batch_size]
            batch_depth = self.depth[image_index][ray_indices:ray_indices+self.batch_size]
            batch_sky = self.sky_mask[image_index][ray_indices:ray_indices+self.batch_size]
            batch_rays = utils.namedtuple_map(lambda r: r[image_index][ray_indices:ray_indices+self.batch_size],
                                              self.rays)

        elif self.batching == 'timestep':
            if self.train_it <= self.c2f_steps[0]:
                key = '16'
            elif self.c2f_steps[0] < self.train_it <= self.c2f_steps[1]:
                key = '12'
            elif self.c2f_steps[1] < self.train_it <= self.c2f_steps[2]:
                key = '8'
            elif self.c2f_steps[2] < self.train_it:
                key = '4'

            un, ind = np.unique(self.timesteps, return_counts=True)
            time_index = np.random.randint(0, len(un), ())
            ray_indices = np.random.randint(0, self.rays[key][0][time_index].shape[0],
                                            (self.batch_size,))
            batch_pixels = self.images[key][time_index][ray_indices]
            batch_depth = self.depth[key][time_index][ray_indices]
            batch_sky = self.sky_mask[key][time_index][ray_indices]

            self.train_it += 1

            # get 3D boxes and transformations for this timestep for every object and broadcast to batchsize
            cars = self.obj_ids
            cars = cars[cars != 0]
            batch_init = []
            if self.random_box:
                for i in range(len(un)):
                    batch_init.append(np.array([np.concatenate(self.box_pose[str(i+1)+'_'+str(c)+'_off'][:,None], axis=0) for c in cars]).reshape(-1, 6))
            else:
                for i in range(len(un)):
                    batch_init.append(np.array([np.concatenate(self.box_pose[str(i+1)+'_'+str(c)+'_center'][:,None], axis=0) for c in cars]).reshape(-1, 6))

            batch_init = np.array(batch_init).reshape(len(un), -1, 6)
            if len(batch_init.shape) < 3:
                batch_init = np.expand_dims(batch_init, 1)

            batch_target = np.array([np.concatenate(self.box_pose[str(time_index+1)+'_'+str(c)+'_center'][:, None], axis=0) for c in cars]).reshape(-1, 6)
            batch_box = np.array([np.concatenate(self.box_pose[str(time_index+1)+'_'+str(c)+'_off'][:, None], axis=0) for c in cars]).reshape(-1, 6)
            #batch_box = np.expand_dims(batch_box, 0)
            #batch_box = np.broadcast_to(batch_box, (self.batch_size, batch_box.shape[0], batch_box.shape[1], batch_box.shape[2]))
            batch_can = np.array([np.concatenate(self.box_pose[str(1)+'_'+str(c)+'_off'][:, None], axis=0) for c in cars]).reshape(-1, 6)
            #batch_can = np.expand_dims(batch_can, 0)
            batch_ext = np.array([np.concatenate(self.box_pose[str(time_index+1)+'_'+str(c)+'_ext'][...,None], axis=0) for c in cars]).reshape(-1, 3)
            #batch_ext = np.expand_dims(batch_ext, 0)
            #batch_ext = np.broadcast_to(batch_ext, (self.batch_size, batch_ext.shape[0], batch_ext.shape[1]))
            #batch_rel = np.array([np.concatenate(self.rel_poses[str(time_index+1)+'_'+str(c)+'_rel'], axis=0) for c in cars]).reshape(-1, 4, 4)
            #batch_rel = np.broadcast_to(batch_rel, (self.batch_size, batch_rel.shape[0], batch_rel.shape[1], batch_rel.shape[2]))

            batch_rays = utils.namedtuple_map(lambda r: r[time_index][ray_indices],
                                              self.rays[key])
        else:
            raise NotImplementedError(
                f'{self.batching} batching strategy is not implemented.')

        return {'pixels': batch_pixels, 'rays': batch_rays, 'depth': batch_depth, 'sky': batch_sky, 'box': batch_box,
                'ext': batch_ext, 'can': batch_can, 'ts': time_index, 'target': batch_target, 'init': batch_init}

    def _test_init(self, config):
        self._load_renderings(config)
        self._generate_rays_multi(self.w, self.h, self.focal, factor=4)
        self._generate_rays_multi(self.w_8, self.h_8, self.focal_8, factor=8)
        self._generate_rays_multi(self.w_12, self.h_12, self.focal_12, factor=12)
        self._generate_rays_multi(self.w_16, self.h_16, self.focal_16, factor=16)
        self.it = 0
        self.train_it = 0

    def _next_test(self):
        """Sample next test example."""
        idx = self.it
        self.it = (self.it + 1) % self.n_examples

        if self.train_it <= self.c2f_steps[0]:
            key = '16'
        elif self.c2f_steps[0] < self.train_it <= self.c2f_steps[1]:
            key = '12'
        elif self.c2f_steps[1] < self.train_it <= self.c2f_steps[2]:
            key = '8'
        elif self.c2f_steps[2] < self.train_it:
            key = '4'

        if self.render_path:
            return {'rays': utils.namedtuple_map(lambda r: r[idx], self.render_rays)}
        else:
            time_index = self.timesteps[idx]
            # get 3D boxes and transformations for this timestep for every object, shape is [N_obj, 4, 4]
            cars = self.obj_ids
            cars = cars[cars != 0]
            init = []
            for i in range(self.total_timesteps):
                init.append(np.array(
                    [np.concatenate(self.box_pose[str(i + 1) + '_' + str(c) + '_center'][:, None], axis=0) for c in
                     cars]).reshape(-1, 6))
            init = np.array(init).reshape(self.total_timesteps, -1, 6)
            box = np.array(
                [np.concatenate(self.box_pose[str(time_index) + '_' + str(c) + '_off'][:, None], axis=0) for c in
                 cars]).reshape(-1, 6)
            target = np.array(
                [np.concatenate(self.box_pose[str(time_index) + '_' + str(c) + '_center'][:, None], axis=0) for c in
                 cars]).reshape(-1, 6)
            can = np.array(
                [np.concatenate(self.box_pose[str(1) + '_' + str(c) + '_off'][:, None], axis=0) for c in
                 cars]).reshape(-1, 6)
            ext = np.array(
                [np.concatenate(self.box_pose[str(time_index) + '_' + str(c) + '_ext'][..., None], axis=0) for c in
                 cars]).reshape(-1, 3)
            rel = np.array(
                [np.concatenate(self.rel_poses[str(time_index) + '_' + str(c) + '_rel'], axis=0) for c in
                 cars]).reshape(-1, 4, 4)

            return {
                'pixels': self.images[key][idx],
                'rays': utils.namedtuple_map(lambda r: r[idx], self.rays[key]),
                'depth': self.depth[key][idx],
                'sky': self.sky_mask[key][idx],
                'box': box,
                'init': init,
                'ext': ext,
                'can': can,
                'ts': time_index-1,
                'target': target
            }

    def _load_renderings(self, config):
        """Load images from disk."""
        # Load images.
        imgdir_suffix = ''
        if config.factor > 0:
            imgdir_suffix = '_{}'.format(config.factor)
            factor = config.factor
        else:
            factor = 1
        imgdir = path.join(self.data_dir, 'images' + imgdir_suffix)
        if not utils.file_exists(imgdir):
            raise ValueError('Image folder {} does not exist.'.format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in natsorted(utils.listdir(imgdir))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        images = []
        for imgfile in imgfiles:
            with utils.open_file(imgfile, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                image = image[:,:,:3]  # remove a from rgba
                images.append(image)
        images = np.array(images)

        imgdir = path.join(self.data_dir, 'images_8')
        if not utils.file_exists(imgdir):
            raise ValueError('Image folder {} does not exist.'.format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in natsorted(utils.listdir(imgdir))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        images_8 = []
        for imgfile in imgfiles:
            with utils.open_file(imgfile, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                image = image[:, :, :3]  # remove a from rgba
                images_8.append(image)
        images_8 = np.array(images_8)

        imgdir = path.join(self.data_dir, 'images_12')
        if not utils.file_exists(imgdir):
            raise ValueError('Image folder {} does not exist.'.format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in natsorted(utils.listdir(imgdir))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        images_12 = []
        for imgfile in imgfiles:
            with utils.open_file(imgfile, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                image = image[:, :, :3]  # remove a from rgba
                images_12.append(image)
        images_12 = np.array(images_12)

        imgdir = path.join(self.data_dir, 'images_16')
        if not utils.file_exists(imgdir):
            raise ValueError('Image folder {} does not exist.'.format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in natsorted(utils.listdir(imgdir))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        images_16 = []
        for imgfile in imgfiles:
            with utils.open_file(imgfile, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                image = image[:, :, :3]  # remove a from rgba
                images_16.append(image)
        images_16 = np.array(images_16)

        # Load poses and bds.
        with utils.open_file(path.join(self.data_dir, 'poses_bounds.npy'),
                             'rb') as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, -2:].transpose([1, 0])
        if poses.shape[-1] != len(images):
            raise RuntimeError('Mismatch between imgs {} and poses {}'.format(
                len(images), poses.shape[-1]))

        # Load 3d bboxes
        with utils.open_file(path.join(self.data_dir, '3D_boxes.npy'),
                             'rb') as fp:
            masks3d = np.load(fp, allow_pickle=True).item()

        # extract 3d bbox center poses
        box_pose = []
        box_ext = []
        for key in masks3d:
            if 'center' in key:
                box_pose.append(masks3d[key])
            elif 'ext' in key:
                box_ext.append(masks3d[key])
        box_pose = np.array(box_pose)
        box_ext = np.array(box_ext)

        # Update poses according to downsampling.
        poses[:2, 4, :] = np.floor(poses[:2, 4, :] * 1. / factor)
        poses[2, 4, :] = poses[2, 4, :] * 1. / factor

        poses_8 = poses.copy()
        poses_8[:2, 4, :] = np.floor(poses_8[:2, 4, :] * 1. / 2)
        poses_8[2, 4, :] = poses_8[2, 4, :] * 1. / 2

        poses_12 = poses.copy()
        poses_12[:2, 4, :] = np.floor(poses_12[:2, 4, :] * 1. / 3)
        poses_12[2, 4, :] = poses_12[2, 4, :] * 1. / 3

        poses_16 = poses.copy()
        poses_16[:2, 4, :] = np.floor(poses_16[:2, 4, :] * 1. / 4)
        poses_16[2, 4, :] = poses_16[2, 4, :] * 1. / 4

        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        poses_8 = np.moveaxis(poses_8, -1, 0).astype(np.float32)
        poses_12 = np.moveaxis(poses_12, -1, 0).astype(np.float32)
        poses_16 = np.moveaxis(poses_16, -1, 0).astype(np.float32)

        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Correct rotation matrix ordering and move variable dim to axis 0.
        #poses = np.concatenate(
        #    [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        #poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        #bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        if config.centering:
            # Rescale according to a default bd factor.
            #scale = 1. / (bds.min() * .75)
            #bds *= scale

            # Recenter poses.
            poses, c2w = self._recenter_poses(poses)

            print(poses[0, :3, 3])
            poses[:, :3, 3] /= 5.0
            if config.random_box:
                self.random_box = config.random_box
                random_box = box_pose.copy()
                random_box[:, :3, 3] += np.random.uniform(-config.box_noise, config.box_noise, size=[box_pose.shape[0], 3])
                #random_box[:, 2, 3] += np.random.uniform(-1.0, 1.0, size=[box_pose.shape[0]])
                print(random_box[0])
                random_box = np.linalg.inv(c2w) @ random_box
                random_box[:, :3, 3] /= 5.0
                print(random_box[0])
            else:
                self.random_box = False
            box_pose = np.linalg.inv(c2w) @ box_pose
            box_pose[:, :3, 3] /= 5.0

            yaw = R.from_matrix(np.linalg.inv(box_pose[:, :3, :3]))  # take inverse of rotation matrix to go from world to object
            yaw = np.array(yaw.as_rotvec())


            if config.random_yaw and config.random_box:
                rand_yaw = yaw.copy()
                print(yaw)
                rand_yaw += (np.random.uniform(-config.yaw_noise, config.yaw_noise, size=yaw.shape) * (np.pi / 180.0))
                print(rand_yaw)
                rand_pose = np.concatenate([random_box[:, :3, 3], rand_yaw], axis=-1)
            elif config.random_box:
                rand_pose = np.concatenate([random_box[:, :3, 3], yaw], axis=-1)
            else:
                rand_pose = np.concatenate([box_pose[:, :3, 3], yaw], axis=-1)

            obbpose = np.concatenate([box_pose[:, :3, 3], yaw], axis=-1)
            box_ext /= 5.0
            print(poses[0, :3, 3])

            rel_pose = {}
            bpose = [k for k in masks3d if 'center' in k]
            print(bpose)
            for i, key in enumerate(bpose):
                if '1_' in key and 'center' in key:
                    can_pose = box_pose[i]
                    ts, car, _ = key.split('_')
                    rel_pose[ts + '_' + car + '_rel'] = np.eye(4)
                    masks3d[key] = obbpose[i]
                    masks3d[ts + '_' + car + '_off'] = rand_pose[i]
                    masks3d[ts + '_' + car + '_ext'] = box_ext[i]
                else:
                    ts, car, _ = key.split('_')
                    rel_pose[ts+'_'+car+'_rel'] = np.matmul(can_pose, np.linalg.inv(box_pose[i]))
                    masks3d[key] = obbpose[i]
                    masks3d[ts + '_' + car + '_off'] = rand_pose[i]
                    masks3d[ts + '_' + car + '_ext'] = box_ext[i]

        else:
            rel_pose = {}
            bpose = [k for k in masks3d if 'center' in k]
            for i, key in enumerate(bpose):
                if '1_' in key and 'center' in key:
                    can_pose = box_pose[i]
                    ts, car, _ = key.split('_')
                    rel_pose[ts + '_' + car + '_rel'] = np.eye(4)
                else:
                    ts, car, _ = key.split('_')
                    rel_pose[ts + '_' + car + '_rel'] = np.matmul(can_pose, np.linalg.inv(box_pose[i]))

        # Load depth
        with utils.open_file(path.join(self.data_dir, 'depth_images.npz'),
                             'rb') as fp:
            depth_list = np.load(fp, allow_pickle=True)['arr_0']
        if len(depth_list) != len(images):
            raise RuntimeError('Mismatch between imgs {} and depth {}'.format(
                len(images), len(depth_list)))

        with utils.open_file(path.join(self.data_dir, 'depth_images_8.npz'),
                             'rb') as fp:
            depth_list_8 = np.load(fp, allow_pickle=True)['arr_0']
        if len(depth_list) != len(images):
            raise RuntimeError('Mismatch between imgs {} and depth {}'.format(
                len(images), len(depth_list)))

        with utils.open_file(path.join(self.data_dir, 'depth_images_12.npz'),
                             'rb') as fp:
            depth_list_12 = np.load(fp, allow_pickle=True)['arr_0']
        if len(depth_list) != len(images):
            raise RuntimeError('Mismatch between imgs {} and depth {}'.format(
                len(images), len(depth_list)))

        with utils.open_file(path.join(self.data_dir, 'depth_images_16.npz'),
                             'rb') as fp:
            depth_list_16 = np.load(fp, allow_pickle=True)['arr_0']
        if len(depth_list) != len(images):
            raise RuntimeError('Mismatch between imgs {} and depth {}'.format(
                len(images), len(depth_list)))

        # Load sky mask
        with utils.open_file(path.join(self.data_dir, 'sky_masks.npz'),
                             'rb') as fp:
            sky_mask = np.load(fp, allow_pickle=True)['arr_0']
        if len(sky_mask) != len(images):
            raise RuntimeError('Mismatch between imgs {} and depth {}'.format(
                len(images), len(sky_mask)))

        with utils.open_file(path.join(self.data_dir, 'sky_masks_8.npz'),
                             'rb') as fp:
            sky_mask_8 = np.load(fp, allow_pickle=True)['arr_0']
        if len(sky_mask) != len(images):
            raise RuntimeError('Mismatch between imgs {} and depth {}'.format(
                len(images), len(sky_mask)))

        with utils.open_file(path.join(self.data_dir, 'sky_masks_12.npz'),
                             'rb') as fp:
            sky_mask_12 = np.load(fp, allow_pickle=True)['arr_0']
        if len(sky_mask) != len(images):
            raise RuntimeError('Mismatch between imgs {} and depth {}'.format(
                len(images), len(sky_mask)))

        with utils.open_file(path.join(self.data_dir, 'sky_masks_16.npz'),
                             'rb') as fp:
            sky_mask_16 = np.load(fp, allow_pickle=True)['arr_0']
        if len(sky_mask) != len(images):
            raise RuntimeError('Mismatch between imgs {} and depth {}'.format(
                len(images), len(sky_mask)))

        # Load 2d bboxes
        with utils.open_file(path.join(self.data_dir, '2D_boxes.npz'),
                             'rb') as fp:
            masks2d = np.load(fp, allow_pickle=True)['arr_0']
        if len(masks2d) != len(images):
            raise RuntimeError('Mismatch between imgs {} and masks2d {}'.format(
                len(images), len(masks2d)))

        # generate timesteps, each position in dataset has 5 images: FRONT, FRONT_LEFT, SIDE_LEFT, FRONT_RIGHT, SIDE_RIGHT
        timesteps = []
        for i in range(1, int(len(masks2d)/5) + 1):
            timesteps.append(np.array([i, i, i, i, i]))

        timesteps = np.asarray(timesteps).reshape(-1)
        total_timesteps = timesteps[-1]
        # Generate a spiral/spherical ray path for rendering videos.
        if config.spherify:
            #poses = self._generate_spherical_poses(poses, bds)
            self.spherify = True
        else:
            self.spherify = False
        if not config.spherify and self.split == 'test':
            self._generate_spiral_poses(poses, bds)
        """
        # generate per pixel / ray bbox data
        rel = []
        cent = []
        ext = []
        i = 1
        for j, m in enumerate(masks2d):
            if j > 0 and j % 5 == 0:
                i += 1
            u = np.unique(m)
            u = u[u > 0]
            for ids in u:
                mask = (m == ids)
                c = np.zeros((m.shape[0], m.shape[1], 4, 4))
                r = np.zeros((m.shape[0], m.shape[1], 4, 4))
                e = np.zeros((m.shape[0], m.shape[1], 3))
                c[mask] = masks3d[str(i) + '_' + str(ids) + '_center']
                e[mask] = masks3d[str(i) + '_' + str(ids) + '_ext']
                r[mask] = rel_pose[str(i) + '_' + str(ids) + '_rel']
                cent.append(c)
                ext.append(e)
                rel.append(r)
            if u.size == 0:
                c = np.zeros((m.shape[0], m.shape[1], 4, 4))
                r = np.zeros((m.shape[0], m.shape[1], 4, 4))
                e = np.zeros((m.shape[0], m.shape[1], 3))
                cent.append(c)
                ext.append(e)
                rel.append(r)

        rel = np.array(rel)
        ext = np.array(ext)
        cent = np.array(cent)"""

        # Select the split.
        i_test = np.arange(len(images))[::config.llffhold]
        #i_test = np.array([0, 20, 40])
        i_train = np.array(
            [i for i in np.arange(len(images))]) #if i not in i_test])
        if self.split == 'train':
            indices = i_train
        elif self.split == 'render':
            indices = i_train
        else:
            indices = i_test
        print(indices)
        images = images[indices]
        images_8 = images_8[indices]
        images_12 = images_12[indices]
        images_16 = images_16[indices]
        depth_list = depth_list[indices]
        depth_list_8 = depth_list_8[indices]
        depth_list_12 = depth_list_12[indices]
        depth_list_16 = depth_list_16[indices]
        sky_mask = sky_mask[indices]
        sky_mask_8 = sky_mask_8[indices]
        sky_mask_12 = sky_mask_12[indices]
        sky_mask_16 = sky_mask_16[indices]
        poses = poses[indices]
        poses_8 = poses_8[indices]
        poses_12 = poses_12[indices]
        poses_16 = poses_16[indices]
        timesteps = timesteps[indices]
        masks2d = masks2d[indices]
        print(timesteps)

        self.rel_poses = rel_pose
        self.box_pose = masks3d
        self.obj_ids = []
        for u in masks2d:
            un = np.unique(u)
            un = un[un!=0]
            for i in un:
                if i != 0 and i not in self.obj_ids:
                    self.obj_ids.append(i)
        self.obj_ids = np.array(self.obj_ids)
        self.masks2d = list(masks2d)
        self.timesteps = timesteps
        self.total_timesteps = total_timesteps
        self.images = {}
        self.images['4'] = list(images)
        self.images['8'] = list(images_8)
        self.images['12'] = list(images_12)
        self.images['16'] = list(images_16)
        self.depth = {}
        self.depth['4'] = list(depth_list)
        self.depth['8'] = list(depth_list_8)
        self.depth['12'] = list(depth_list_12)
        self.depth['16'] = list(depth_list_16)
        for key in self.depth:
            for elem in self.depth[key]:
                elem[elem>0.0] = elem[elem>0.0] / 5.0
            if config.batching == 'timestep':
                for i in range(len(self.depth[key])):
                    #if self.timesteps[i] > 1:
                    #    mask = (self.masks2d[i] > 0.0)
                    #    self.depth[i][mask] = 0.0
                    self.depth[key][i] = np.expand_dims(self.depth[key][i], -1)
                    self.masks2d[i] = np.expand_dims(self.masks2d[i], -1)
        self.sky_mask = {}
        self.sky_mask['4'] = list(sky_mask)
        self.sky_mask['8'] = list(sky_mask_8)
        self.sky_mask['12'] = list(sky_mask_12)
        self.sky_mask['16'] = list(sky_mask_16)
        # set skymask to large distance but not infinity
        for key in self.sky_mask:
            for elem in self.sky_mask[key]:
                elem[elem>0.0] = 0.995
            if config.batching == 'timestep':
                for i in range(len(self.sky_mask[key])):
                    self.sky_mask[key][i] = np.expand_dims(self.sky_mask[key][i], -1)
        self.camtoworlds = poses[:, :3, :4]

        # get 5 focal points, w, h, one for each image
        self.focal = poses[:, -1, -1]
        self.focal_8 = poses_8[:, -1, -1]
        self.focal_12 = poses_12[:, -1, -1]
        self.focal_16 = poses_16[:, -1, -1]
        self.h = poses[:, 0, -1]
        self.h_8 = poses_8[:, 0, -1]
        self.h_12 = poses_12[:, 0, -1]
        self.h_16 = poses_16[:, 0, -1]
        self.w = poses[:, 1, -1]
        self.w_8 = poses_8[:, 1, -1]
        self.w_12 = poses_12[:, 1, -1]
        self.w_16 = poses_16[:, 1, -1]

        self.rays = {}

        self.resolution = self.h * self.w
        self.resolution_8 = self.h_8 * self.w_8
        self.resolution_12 = self.h_12 * self.w_12
        self.resolution_16 = self.h_16 * self.w_16
        self.n_examples = len(self.images['4'])

    def _generate_rays_multi(self, w, h, focal, factor):
        """Generating rays for all images."""

        def res2grid(w, h):
            return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
                np.arange(w, dtype=np.float32),  # X-Axis (columns)
                np.arange(h, dtype=np.float32),  # Y-Axis (rows)
                indexing='xy')

        xy = [res2grid(w, h) for w, h in zip(w, h)]
        directions = []
        origins = []
        viewdirs = []
        for i in range(len(xy)):
            cam_dirs = np.stack(
                [(xy[i][0] - w[i] * 0.5) / focal[i],
                 -(xy[i][1] - h[i] * 0.5) / focal[i], -np.ones_like(xy[i][0])],
                axis=-1)

            directions.append(np.squeeze((cam_dirs[..., None, :] *
                                          self.camtoworlds[i, :3, :3]).sum(axis=-1)))

            origins.append(np.broadcast_to(self.camtoworlds[i, :3, -1], directions[i].shape))

            viewdirs.append(directions[i] / np.linalg.norm(directions[i], axis=-1, keepdims=True))

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

        ones = [np.ones_like(o[..., :1]) for o in origins]
        zeros = [np.zeros_like(o[..., :1]) for o in origins]
        near = [self.near * o for o in ones]
        far = [self.far * o for o in ones]
        timestep = [self.timesteps[i] * o for i, o in enumerate(ones)]
        print('origins:', len(origins))
        self.rays[str(factor)] = utils.BoxRays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=near,
            far=far,)

    def _generate_rays(self):
        """Generate normalized device coordinate rays for llff."""
        #if self.split == 'test':
            #n_render_poses = self.render_poses.shape[0]
            #self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds],
            #                                  axis=0)

        self._generate_rays_multi()
        if not self.spherify:
            ndc_o = []
            ndc_d = []
            r = []
            for i in range(len(self.rays.origins)):
                ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins[i],
                                                             self.rays.directions[i],
                                                             self.focal[i], self.w[i], self.h[i])
                ndc_o.append(ndc_origins)
                ndc_d.append(ndc_directions)
                print(ndc_o[i][0,0])
                print(ndc_d[i][0,0])

                mat = np.expand_dims(ndc_origins, 0)

                # Distance from each unit-norm direction vector to its x-axis neighbor.
                dx = np.sqrt(np.sum((mat[:, :-1, :, :] - mat[:, 1:, :, :]) ** 2, -1))
                dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

                dy = np.sqrt(np.sum((mat[:, :, :-1, :] - mat[:, :, 1:, :]) ** 2, -1))
                dy = np.concatenate([dy, dy[:, :, -2:-1]], 2)
                # Cut the distance in half, and then round it out so that it's
                # halfway between inscribed by / circumscribed about the pixel.
                radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)
                r.append(radii)

            ones = [np.ones_like(o[..., :1]) for o in ndc_o]
            near = [self.near * o for o in ones]
            far = [self.far * o for o in ones]
            self.rays = utils.Rays(
                origins=ndc_o,
                directions=ndc_d,
                viewdirs=self.rays.directions,
                radii=r,
                lossmult=ones,
                near=near,
                far=far)

        # Split poses from the dataset and generated poses
        #if self.split == 'test':
            #self.camtoworlds = self.camtoworlds[n_render_poses:]
            #split = [np.split(r, [n_render_poses], 0) for r in self.rays]
            #rint(len(split))
            #split0, split1 = zip(*split)
            #self.render_rays = utils.Rays(*split0)
            #self.rays = utils.Rays(*split1)
            #print(len(self.rays.origins))

    def normalize(self, v):
        """Normalize a vector."""
        return v / np.linalg.norm(v)

    def average_poses(self, poses):
        """
        Calculate the average pose, which is then used to center all poses
        using @center_poses. Its computation is as follows:
        1. Compute the center: the average of pose centers.
        2. Compute the z axis: the normalized average z axis.
        3. Compute axis y': the average y axis.
        4. Compute x' = y' cross product z, then normalize it as the x axis.
        5. Compute the y axis: z cross product x.

        Note that at step 3, we cannot directly use y' as y axis since it's
        not necessarily orthogonal to z axis. We need to pass from x to y.

        Inputs:
            poses: (N_images, 3, 4)

        Outputs:
            pose_avg: (3, 4) the average pose
        """
        # 1. Compute the center
        center = poses[..., 3].mean(0)  # (3)

        # 2. Compute the z axis
        z = self.normalize(poses[..., 2].mean(0))  # (3)

        # 3. Compute axis y' (no need to normalize as it's not the final output)
        y_ = poses[..., 1].mean(0)  # (3)

        # 4. Compute the x axis
        x = self.normalize(np.cross(y_, z))  # (3)

        # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
        y = np.cross(z, x)  # (3)

        pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

        return pose_avg

    def center_poses(self, poses):
        """
        Center the poses so that we can use NDC.
        See https://github.com/bmild/nerf/issues/34

        Inputs:
            poses: (N_images, 3, 4)

        Outputs:
            poses_centered: (N_images, 3, 4) the centered poses
            pose_avg: (3, 4) the average pose
        """

        pose_avg = self.average_poses(poses)  # (3, 4)
        pose_avg_homo = np.eye(4)
        pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
        # by simply adding 0, 0, 0, 1 as the last row
        last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
        poses_homo = \
            np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

        poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
        poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

        return poses_centered, np.linalg.inv(pose_avg_homo)

    def _recenter_poses(self, poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.], [1, 4])
        c2w = self._poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses, c2w

    def _poses_avg(self, poses):
        """Average poses according to the original NeRF code."""
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = self._normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
        return c2w

    def _viewmatrix(self, z, up, pos):
        """Construct lookat view matrix."""
        vec2 = self._normalize(z)
        vec1_avg = up
        vec0 = self._normalize(np.cross(vec1_avg, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def _normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def _generate_spiral_poses(self, poses, bds):
        """Generate a spiral path for rendering."""
        c2w = self._poses_avg(poses)
        # Get average pose.
        up = self._normalize(poses[:, :3, 1].sum(0))
        # Find a reasonable 'focus depth' for this dataset.
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path.
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        n_views = 120
        n_rots = 2
        # Generate poses for spiral path.
        render_poses = []
        rads = np.array(list(rads) + [1.])
        hwf = c2w_path[:, 4:5]
        zrate = .5
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
            c = np.dot(c2w[:3, :4], (np.array(
                [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
            z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
            render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
        self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]

    def _generate_spherical_poses(self, poses, bds):
        """Generate a 360 degree spherical path for rendering."""
        # pylint: disable=g-long-lambda
        p34_to_44 = lambda p: np.concatenate([
            p,
            np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
        ], 1)
        rays_d = poses[:, :3, 2:3]
        rays_o = poses[:, :3, 3:4]

        def min_line_dist(rays_o, rays_d):
            a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -a_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv(
                (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (poses[:, :3, 3] - center).mean(0)
        vec0 = self._normalize(up)
        vec1 = self._normalize(np.cross([.1, .2, .3], vec0))
        vec2 = self._normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = (
                np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
        sc = 1. / rad
        poses_reset[:, :3, 3] *= sc
        bds *= sc
        rad *= sc
        centroid = np.mean(poses_reset[:, :3, 3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad ** 2 - zh ** 2)
        new_poses = []

        for th in np.linspace(0., 2. * np.pi, 120):
            camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0, 0, -1.])
            vec2 = self._normalize(camorigin)
            vec0 = self._normalize(np.cross(vec2, up))
            vec1 = self._normalize(np.cross(vec2, vec0))
            pos = camorigin
            p = np.stack([vec0, vec1, vec2, pos], 1)
            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        new_poses = np.concatenate([
            new_poses,
            np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
        ], -1)
        poses_reset = np.concatenate([
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
        ], -1)
        if self.split == 'test':
            self.render_poses = new_poses[:, :3, :4]
        return poses_reset


class Waymo(Dataset):
    """Waymo Dataset."""

    def _train_init(self, config):
        """Initialize training."""
        np.random.seed(20201473)  # TODO: do i need to put this here or is in train enough?
        self._load_renderings(config)
        self._generate_rays()

        if config.batching == 'all_images':
            def flatten(x):
                # Always flatten out the height x width dimensions
                x = [y.reshape([-1, y.shape[-1]]) for y in x]
                if config.batching == 'all_images':
                    # If global batching, also concatenate all data into one list
                    x = np.concatenate(x, axis=0)
                return x

            self.images = flatten(self.images)
            self.depth = flatten(self.depth)
            self.sky_mask = flatten(self.sky_mask)
            self.rays = utils.namedtuple_map(flatten, self.rays)
            print(self.images.shape, self.depth.shape, self.sky_mask.shape)
        elif config.batching == 'single_image' or config.batching == 'single_image_consecutive':
            for i in range(len(self.images)):
                self.images[i] = self.images[i].reshape((int(self.resolution[i]), 3))
                self.depth[i] = self.depth[i].reshape((int(self.resolution[i]), 1))
                self.sky_mask[i] = self.sky_mask[i].reshape((int(self.resolution[i]), 1))
                for j in range(len(self.rays)):
                    self.rays[j][i] = self.rays[j][i].reshape((int(self.resolution[i]), -1))
        elif config.batching == 'timestep':

            def flatten_time(x):
                # flatten batches along timestep
                xt = []
                for y in x:
                    if y.shape[-1] == 4:
                        xt.append(y.reshape([-1, y.shape[-1], y.shape[-1]]))
                    else:
                        xt.append(y.reshape([-1, y.shape[-1]]))
                x = xt
                _, ind = np.unique(self.timesteps, return_counts=True)
                out = []
                for i in range(len(ind)):
                    if i == 0:
                        out.append(np.concatenate(x[:ind[i]], axis=0))
                        ind[i+1] += ind[i]
                    else:
                        out.append(np.concatenate(x[ind[i-1]:ind[i]]))
                        if i+1 < len(ind):
                            ind[i+1] += ind[i]
                return out

            self.images = flatten_time(self.images)
            self.depth = flatten_time(self.depth)
            self.sky_mask = flatten_time(self.sky_mask)
            self.masks2d = flatten_time(self.masks2d)
            #self.dyn_masks = flatten_time(self.dyn_masks)
            self.rays = utils.namedtuple_map(flatten_time, self.rays)

    def _next_train(self):
        """Sample next training batch."""

        if self.batching == 'all_images':
            ray_indices = np.random.randint(0, self.rays[0].shape[0],
                                            (self.batch_size,))
            batch_pixels = self.images[ray_indices]
            batch_depth = self.depth[ray_indices]
            batch_sky = self.sky_mask[ray_indices]
            batch_rays = utils.namedtuple_map(lambda r: r[ray_indices], self.rays)

            #ind_i = np.broadcast_to(ray_indices, (self.batch_size, self.batch_size))
            #ray_diff = np.abs(ind_i - ind_i.T)
            #neighbour_mask = (ray_diff == 1.0).astype(np.float32)

        elif self.batching == 'single_image':
            image_index = np.random.randint(0, self.n_examples, ())
            ray_indices = np.random.randint(0, self.rays[0][image_index].shape[0],
                                            (self.batch_size,))
            batch_pixels = self.images[image_index][ray_indices]
            batch_depth = self.depth[image_index][ray_indices]
            batch_sky = self.sky_mask[image_index][ray_indices]
            batch_rays = utils.namedtuple_map(lambda r: r[image_index][ray_indices],
                                              self.rays)

        elif self.batching == 'single_image_consecutive':
            image_index = np.random.randint(0, self.n_examples, ())
            ray_indices = np.random.randint(0, self.rays[0][image_index].shape[0] - self.batch_size)
            batch_pixels = self.images[image_index][ray_indices:ray_indices+self.batch_size]
            batch_depth = self.depth[image_index][ray_indices:ray_indices+self.batch_size]
            batch_sky = self.sky_mask[image_index][ray_indices:ray_indices+self.batch_size]
            batch_rays = utils.namedtuple_map(lambda r: r[image_index][ray_indices:ray_indices+self.batch_size],
                                              self.rays)

        elif self.batching == 'timestep':
            un, ind = np.unique(self.timesteps, return_counts=True)
            time_index = np.random.randint(0, len(un), ())
            ray_indices = np.random.randint(0, self.rays[0][time_index].shape[0],
                                            (self.batch_size,))
            batch_pixels = self.images[time_index][ray_indices]
            batch_depth = self.depth[time_index][ray_indices]
            batch_sky = self.sky_mask[time_index][ray_indices]

            # get 3D boxes and transformations for this timestep for every object and broadcast to batchsize
            cars = self.obj_ids
            cars = cars[cars != 0]
            batch_init = []
            if self.random_box:
                for i in range(len(un)):
                    batch_init.append(np.array([np.concatenate(self.box_pose[str(i+1)+'_'+str(c)+'_off'][:,None], axis=0) for c in cars]).reshape(-1, 6))
            else:
                for i in range(len(un)):
                    batch_init.append(np.array([np.concatenate(self.box_pose[str(i+1)+'_'+str(c)+'_center'][:,None], axis=0) for c in cars]).reshape(-1, 6))

            batch_init = np.array(batch_init).reshape(len(un), -1, 6)
            if len(batch_init.shape) < 3:
                batch_init = np.expand_dims(batch_init, 1)

            batch_target = np.array([np.concatenate(self.box_pose[str(time_index+1)+'_'+str(c)+'_center'][:, None], axis=0) for c in cars]).reshape(-1, 6)
            batch_box = np.array([np.concatenate(self.box_pose[str(time_index+1)+'_'+str(c)+'_off'][:, None], axis=0) for c in cars]).reshape(-1, 6)
            #batch_box = np.expand_dims(batch_box, 0)
            #batch_box = np.broadcast_to(batch_box, (self.batch_size, batch_box.shape[0], batch_box.shape[1], batch_box.shape[2]))
            batch_can = np.array([np.concatenate(self.box_pose[str(1)+'_'+str(c)+'_off'][:, None], axis=0) for c in cars]).reshape(-1, 6)
            #batch_can = np.expand_dims(batch_can, 0)
            batch_ext = np.array([np.concatenate(self.box_pose[str(time_index+1)+'_'+str(c)+'_ext'][...,None], axis=0) for c in cars]).reshape(-1, 3)
            #batch_ext = np.expand_dims(batch_ext, 0)
            #batch_ext = np.broadcast_to(batch_ext, (self.batch_size, batch_ext.shape[0], batch_ext.shape[1]))
            #batch_rel = np.array([np.concatenate(self.rel_poses[str(time_index+1)+'_'+str(c)+'_rel'], axis=0) for c in cars]).reshape(-1, 4, 4)
            #batch_rel = np.broadcast_to(batch_rel, (self.batch_size, batch_rel.shape[0], batch_rel.shape[1], batch_rel.shape[2]))

            batch_rays = utils.namedtuple_map(lambda r: r[time_index][ray_indices],
                                              self.rays)
        else:
            raise NotImplementedError(
                f'{self.batching} batching strategy is not implemented.')

        return {'pixels': batch_pixels, 'rays': batch_rays, 'depth': batch_depth, 'sky': batch_sky, 'box': batch_box,
                'ext': batch_ext, 'can': batch_can, 'ts': time_index, 'target': batch_target, 'init': batch_init}

    def _next_test(self):
        """Sample next test example."""
        idx = self.it
        self.it = (self.it + 1) % self.n_examples

        if self.render_path:
            return {'rays': utils.namedtuple_map(lambda r: r[idx], self.render_rays)}
        else:
            time_index = self.timesteps[idx]
            # get 3D boxes and transformations for this timestep for every object, shape is [N_obj, 4, 4]
            cars = self.obj_ids
            cars = cars[cars != 0]
            init = []
            for i in range(self.total_timesteps):
                init.append(np.array(
                    [np.concatenate(self.box_pose[str(i + 1) + '_' + str(c) + '_center'][:, None], axis=0) for c in
                     cars]).reshape(-1, 6))
            init = np.array(init).reshape(self.total_timesteps, -1, 6)
            box = np.array(
                [np.concatenate(self.box_pose[str(time_index) + '_' + str(c) + '_off'][:, None], axis=0) for c in
                 cars]).reshape(-1, 6)
            target = np.array(
                [np.concatenate(self.box_pose[str(time_index) + '_' + str(c) + '_center'][:, None], axis=0) for c in
                 cars]).reshape(-1, 6)
            can = np.array(
                [np.concatenate(self.box_pose[str(1) + '_' + str(c) + '_off'][:, None], axis=0) for c in
                 cars]).reshape(-1, 6)
            ext = np.array(
                [np.concatenate(self.box_pose[str(time_index) + '_' + str(c) + '_ext'][..., None], axis=0) for c in
                 cars]).reshape(-1, 3)
            rel = np.array(
                [np.concatenate(self.rel_poses[str(time_index) + '_' + str(c) + '_rel'], axis=0) for c in
                 cars]).reshape(-1, 4, 4)
            return {
                'pixels': self.images[idx],
                'rays': utils.namedtuple_map(lambda r: r[idx], self.rays),
                'depth': self.depth[idx],
                'sky': self.sky_mask[idx],
                'box': box,
                'init': init,
                'ext': ext,
                'can': can,
                'ts': time_index-1,
                'target': target
            }

    def _load_renderings(self, config):
        """Load images from disk."""
        # Load images.
        imgdir_suffix = ''
        if config.factor > 0:
            imgdir_suffix = '_{}'.format(config.factor)
            factor = config.factor
        else:
            factor = 1
        imgdir = path.join(self.data_dir, 'images' + imgdir_suffix)
        if not utils.file_exists(imgdir):
            raise ValueError('Image folder {} does not exist.'.format(imgdir))
        imgfiles = [
            path.join(imgdir, f)
            for f in natsorted(utils.listdir(imgdir))
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')
        ]
        images = []
        for imgfile in imgfiles:
            with utils.open_file(imgfile, 'rb') as imgin:
                image = np.array(Image.open(imgin), dtype=np.float32) / 255.
                image = image[:,:,:3]  # remove a from rgba
                images.append(image)
        images = np.array(images)

        # Load poses and bds.
        with utils.open_file(path.join(self.data_dir, 'poses_bounds.npy'),
                             'rb') as fp:
            poses_arr = np.load(fp)
        poses = poses_arr[:, :15].reshape([-1, 3, 5]).transpose([1, 2, 0])
        bds = poses_arr[:, 15:17].transpose([1, 0])
        principal_point = poses_arr[:, 17:]

        if poses.shape[-1] != len(images):
            raise RuntimeError('Mismatch between imgs {} and poses {}'.format(
                len(images), poses.shape[-1]))

        # Load 3d bboxes
        with utils.open_file(path.join(self.data_dir, '3D_boxes.npy'),
                             'rb') as fp:
            masks3d = np.load(fp, allow_pickle=True).item()

        # extract 3d bbox center poses
        box_pose = []
        box_ext = []
        for key in masks3d:
            if 'center' in key:
                box_pose.append(masks3d[key])
            elif 'ext' in key:
                box_ext.append(masks3d[key])
        box_pose = np.array(box_pose)
        box_ext = np.array(box_ext)

        # Update poses according to downsampling.
        poses[:2, 4, :] = np.floor(poses[:2, 4, :] * 1. / factor)
        poses[2, 4, :] = poses[2, 4, :] * 1. / factor
        principal_point = principal_point * 1. / factor

        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        # Correct rotation matrix ordering and move variable dim to axis 0.
        #poses = np.concatenate(
        #    [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
        #poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        #bds = np.moveaxis(bds, -1, 0).astype(np.float32)

        if config.centering:
            # Rescale according to a default bd factor.
            #scale = 1. / (bds.min() * .75)
            #bds *= scale

            # Recenter poses.
            poses, c2w = self._recenter_poses(poses)

            print(poses[0, :3, 3])
            poses[:, :3, 3] /= 5.0
            if config.random_box:
                self.random_box = config.random_box
                random_box = box_pose.copy()
                random_box[:, :3, 3] += np.random.uniform(-config.box_noise, config.box_noise, size=[box_pose.shape[0], 3])
                #random_box[:, 2, 3] += np.random.uniform(-1.0, 1.0, size=[box_pose.shape[0]])
                print(random_box[0])
                random_box = np.linalg.inv(c2w) @ random_box
                random_box[:, :3, 3] /= 5.0
                print(random_box[0])
            else:
                self.random_box = False
            box_pose = np.linalg.inv(c2w) @ box_pose
            box_pose[:, :3, 3] /= 5.0

            yaw = R.from_matrix(np.linalg.inv(box_pose[:, :3, :3]))  # take inverse of rotation matrix to go from world to object
            yaw = np.array(yaw.as_rotvec())


            if config.random_yaw and config.random_box:
                rand_yaw = yaw.copy()
                print(yaw)
                rand_yaw += (np.random.uniform(-config.yaw_noise, config.yaw_noise, size=yaw.shape) * (np.pi / 180.0))
                print(rand_yaw)
                rand_pose = np.concatenate([random_box[:, :3, 3], rand_yaw], axis=-1)
            elif config.random_box:
                rand_pose = np.concatenate([random_box[:, :3, 3], yaw], axis=-1)
            else:
                rand_pose = np.concatenate([box_pose[:, :3, 3], yaw], axis=-1)

            obbpose = np.concatenate([box_pose[:, :3, 3], yaw], axis=-1)
            box_ext /= 5.0
            print(poses[0, :3, 3])

            rel_pose = {}
            bpose = [k for k in masks3d if 'center' in k]
            print(bpose)
            for i, key in enumerate(bpose):
                if '1_' in key and 'center' in key:
                    can_pose = box_pose[i]
                    ts, car, _ = key.split('_')
                    rel_pose[ts + '_' + car + '_rel'] = np.eye(4)
                    masks3d[key] = obbpose[i]
                    masks3d[ts + '_' + car + '_off'] = rand_pose[i]
                    masks3d[ts + '_' + car + '_ext'] = box_ext[i]
                else:
                    ts, car, _ = key.split('_')
                    rel_pose[ts+'_'+car+'_rel'] = np.matmul(can_pose, np.linalg.inv(box_pose[i]))
                    masks3d[key] = obbpose[i]
                    masks3d[ts + '_' + car + '_off'] = rand_pose[i]
                    masks3d[ts + '_' + car + '_ext'] = box_ext[i]

        else:
            rel_pose = {}
            bpose = [k for k in masks3d if 'center' in k]
            for i, key in enumerate(bpose):
                if '1_' in key and 'center' in key:
                    can_pose = box_pose[i]
                    ts, car, _ = key.split('_')
                    rel_pose[ts + '_' + car + '_rel'] = np.eye(4)
                else:
                    ts, car, _ = key.split('_')
                    rel_pose[ts + '_' + car + '_rel'] = np.matmul(can_pose, np.linalg.inv(box_pose[i]))

        # Load depth
        with utils.open_file(path.join(self.data_dir, 'depth_images.npz'),
                             'rb') as fp:
            depth_list = np.load(fp, allow_pickle=True)['arr_0']
        if len(depth_list) != len(images):
            raise RuntimeError('Mismatch between imgs {} and depth {}'.format(
                len(images), len(depth_list)))

        # Load sky mask
        with utils.open_file(path.join(self.data_dir, 'sky_masks.npz'),
                             'rb') as fp:
            sky_mask = np.load(fp, allow_pickle=True)['arr_0']
        if len(sky_mask) != len(images):
            raise RuntimeError('Mismatch between imgs {} and depth {}'.format(
                len(images), len(sky_mask)))

        # Load 2d bboxes
        with utils.open_file(path.join(self.data_dir, '2D_boxes.npz'),
                             'rb') as fp:
            masks2d = np.load(fp, allow_pickle=True)['arr_0']
        if len(masks2d) != len(images):
            raise RuntimeError('Mismatch between imgs {} and masks2d {}'.format(
                len(images), len(masks2d)))

        # generate timesteps, each position in dataset has 5 images: FRONT, FRONT_LEFT, SIDE_LEFT, FRONT_RIGHT, SIDE_RIGHT
        timesteps = []
        for i in range(1, int(len(masks2d) / 5) + 1):
            timesteps.append(np.array([i, i, i, i, i]))

        timesteps = np.asarray(timesteps).reshape(-1)
        total_timesteps = timesteps[-1]
        # Generate a spiral/spherical ray path for rendering videos.
        if config.spherify:
            #poses = self._generate_spherical_poses(poses, bds)
            self.spherify = True
        else:
            self.spherify = False
        if not config.spherify and self.split == 'test':
            self._generate_spiral_poses(poses, bds)
        """
        # generate per pixel / ray bbox data
        rel = []
        cent = []
        ext = []
        i = 1
        for j, m in enumerate(masks2d):
            if j > 0 and j % 5 == 0:
                i += 1
            u = np.unique(m)
            u = u[u > 0]
            for ids in u:
                mask = (m == ids)
                c = np.zeros((m.shape[0], m.shape[1], 4, 4))
                r = np.zeros((m.shape[0], m.shape[1], 4, 4))
                e = np.zeros((m.shape[0], m.shape[1], 3))
                c[mask] = masks3d[str(i) + '_' + str(ids) + '_center']
                e[mask] = masks3d[str(i) + '_' + str(ids) + '_ext']
                r[mask] = rel_pose[str(i) + '_' + str(ids) + '_rel']
                cent.append(c)
                ext.append(e)
                rel.append(r)
            if u.size == 0:
                c = np.zeros((m.shape[0], m.shape[1], 4, 4))
                r = np.zeros((m.shape[0], m.shape[1], 4, 4))
                e = np.zeros((m.shape[0], m.shape[1], 3))
                cent.append(c)
                ext.append(e)
                rel.append(r)

        rel = np.array(rel)
        ext = np.array(ext)
        cent = np.array(cent)"""

        # Select the split.
        i_test = np.arange(len(images))[::config.llffhold]
        #i_test = np.array([0, 20, 40])
        i_train = np.array(
            [i for i in np.arange(len(images))]) #if i not in i_test])
        if self.split == 'train':
            indices = i_train
        elif self.split == 'render':
            indices = i_train
        else:
            indices = i_test
        print(indices)
        images = images[indices]
        depth_list = depth_list[indices]
        sky_mask = sky_mask[indices]
        poses = poses[indices]
        timesteps = timesteps[indices]
        masks2d = masks2d[indices]
        print(timesteps)

        self.rel_poses = rel_pose
        self.box_pose = masks3d
        self.obj_ids = []
        for u in masks2d:
            un = np.unique(u)
            un = un[un!=0]
            for i in un:
                if i != 0 and i not in self.obj_ids:
                    self.obj_ids.append(i)
        self.obj_ids = np.array([1])
        self.masks2d = list(masks2d)
        self.timesteps = timesteps
        self.total_timesteps = total_timesteps
        self.images = list(images)
        self.depth = list(depth_list)
        for elem in self.depth:
            elem[elem>0.0] = elem[elem>0.0] / 5.0
        print(self.depth[0][self.depth[0]>0.0])
        print(self.depth[0][self.depth[0]<0.0])
        print(np.max(self.depth[0]))
        print(np.min(self.depth[0][self.depth[0]>0.0]))

        if config.batching == 'timestep':
            for i in range(len(self.depth)):
                #if self.timesteps[i] > 1:
                #    mask = (self.masks2d[i] > 0.0)
                #    self.depth[i][mask] = 0.0
                self.depth[i] = np.expand_dims(self.depth[i], -1)
                self.masks2d[i] = np.expand_dims(self.masks2d[i], -1)
        self.sky_mask = list(sky_mask)
        # set skymask to large distance but not infinity
        for elem in self.sky_mask:
            elem[elem>0.0] = 0.995
        if config.batching == 'timestep':
            for i in range(len(self.sky_mask)):
                self.sky_mask[i] = np.expand_dims(self.sky_mask[i], -1)
        self.camtoworlds = poses[:, :3, :4]

        # get 5 focal points, w, h, one for each image
        self.focal = poses[:, -1, -1]
        self.h = poses[:, 0, -1]
        self.w = poses[:, 1, -1]
        self.principal_point = principal_point

        self.resolution = self.h * self.w
        self.n_examples = len(self.images)

    def _generate_rays_multi(self):
        """Generating rays for all images."""

        def res2grid(w, h):
            return np.meshgrid(  # pylint: disable=unbalanced-tuple-unpacking
                np.arange(w, dtype=np.float32),  # X-Axis (columns)
                np.arange(h, dtype=np.float32),  # Y-Axis (rows)
                indexing='xy')

        xy = [res2grid(w, h) for w, h in zip(self.w, self.h)]
        directions = []
        origins = []
        viewdirs = []
        for i in range(len(xy)):
            cam_dirs = np.stack(
                [(xy[i][0] - self.principal_point[i, 0]) / self.focal[i],
                 -(xy[i][1] - self.principal_point[i, 1]) / self.focal[i], -np.ones_like(xy[i][0])],
                axis=-1)

            directions.append(np.squeeze((cam_dirs[..., None, :] *
                                          self.camtoworlds[i, :3, :3]).sum(axis=-1)))

            origins.append(np.broadcast_to(self.camtoworlds[i, :3, -1], directions[i].shape))

            viewdirs.append(directions[i] / np.linalg.norm(directions[i], axis=-1, keepdims=True))

        # Distance from each unit-norm direction vector to its x-axis neighbor.
        dx = [
            np.sqrt(np.sum((v[:-1, :, :] - v[1:, :, :]) ** 2, -1)) for v in directions
        ]
        dx = [np.concatenate([v, v[-2:-1, :]], 0) for v in dx]
        # Cut the distance in half, and then round it out so that it's
        # halfway between inscribed by / circumscribed about the pixel.
        radii = [v[..., None] * 2 / np.sqrt(12) for v in dx]

        ones = [np.ones_like(o[..., :1]) for o in origins]
        zeros = [np.zeros_like(o[..., :1]) for o in origins]
        near = [self.near * o for o in ones]
        far = [self.far * o for o in ones]
        timestep = [self.timesteps[i] * o for i, o in enumerate(ones)]
        print('origins:', len(origins))
        self.rays = utils.BoxRays(
            origins=origins,
            directions=directions,
            viewdirs=viewdirs,
            radii=radii,
            lossmult=ones,
            near=near,
            far=far,
            object=self.masks2d)

    def _generate_rays(self):
        """Generate normalized device coordinate rays for llff."""
        #if self.split == 'test':
            #n_render_poses = self.render_poses.shape[0]
            #self.camtoworlds = np.concatenate([self.render_poses, self.camtoworlds],
            #                                  axis=0)

        self._generate_rays_multi()
        if not self.spherify:
            ndc_o = []
            ndc_d = []
            r = []
            for i in range(len(self.rays.origins)):
                ndc_origins, ndc_directions = convert_to_ndc(self.rays.origins[i],
                                                             self.rays.directions[i],
                                                             self.focal[i], self.w[i], self.h[i])
                ndc_o.append(ndc_origins)
                ndc_d.append(ndc_directions)
                print(ndc_o[i][0,0])
                print(ndc_d[i][0,0])

                mat = np.expand_dims(ndc_origins, 0)

                # Distance from each unit-norm direction vector to its x-axis neighbor.
                dx = np.sqrt(np.sum((mat[:, :-1, :, :] - mat[:, 1:, :, :]) ** 2, -1))
                dx = np.concatenate([dx, dx[:, -2:-1, :]], 1)

                dy = np.sqrt(np.sum((mat[:, :, :-1, :] - mat[:, :, 1:, :]) ** 2, -1))
                dy = np.concatenate([dy, dy[:, :, -2:-1]], 2)
                # Cut the distance in half, and then round it out so that it's
                # halfway between inscribed by / circumscribed about the pixel.
                radii = (0.5 * (dx + dy))[..., None] * 2 / np.sqrt(12)
                r.append(radii)

            ones = [np.ones_like(o[..., :1]) for o in ndc_o]
            near = [self.near * o for o in ones]
            far = [self.far * o for o in ones]
            self.rays = utils.Rays(
                origins=ndc_o,
                directions=ndc_d,
                viewdirs=self.rays.directions,
                radii=r,
                lossmult=ones,
                near=near,
                far=far)

        # Split poses from the dataset and generated poses
        #if self.split == 'test':
            #self.camtoworlds = self.camtoworlds[n_render_poses:]
            #split = [np.split(r, [n_render_poses], 0) for r in self.rays]
            #rint(len(split))
            #split0, split1 = zip(*split)
            #self.render_rays = utils.Rays(*split0)
            #self.rays = utils.Rays(*split1)
            #print(len(self.rays.origins))

    def normalize(self, v):
        """Normalize a vector."""
        return v / np.linalg.norm(v)

    def average_poses(self, poses):
        """
        Calculate the average pose, which is then used to center all poses
        using @center_poses. Its computation is as follows:
        1. Compute the center: the average of pose centers.
        2. Compute the z axis: the normalized average z axis.
        3. Compute axis y': the average y axis.
        4. Compute x' = y' cross product z, then normalize it as the x axis.
        5. Compute the y axis: z cross product x.

        Note that at step 3, we cannot directly use y' as y axis since it's
        not necessarily orthogonal to z axis. We need to pass from x to y.

        Inputs:
            poses: (N_images, 3, 4)

        Outputs:
            pose_avg: (3, 4) the average pose
        """
        # 1. Compute the center
        center = poses[..., 3].mean(0)  # (3)

        # 2. Compute the z axis
        z = self.normalize(poses[..., 2].mean(0))  # (3)

        # 3. Compute axis y' (no need to normalize as it's not the final output)
        y_ = poses[..., 1].mean(0)  # (3)

        # 4. Compute the x axis
        x = self.normalize(np.cross(y_, z))  # (3)

        # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
        y = np.cross(z, x)  # (3)

        pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

        return pose_avg

    def center_poses(self, poses):
        """
        Center the poses so that we can use NDC.
        See https://github.com/bmild/nerf/issues/34

        Inputs:
            poses: (N_images, 3, 4)

        Outputs:
            poses_centered: (N_images, 3, 4) the centered poses
            pose_avg: (3, 4) the average pose
        """

        pose_avg = self.average_poses(poses)  # (3, 4)
        pose_avg_homo = np.eye(4)
        pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
        # by simply adding 0, 0, 0, 1 as the last row
        last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
        poses_homo = \
            np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

        poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
        poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

        return poses_centered, np.linalg.inv(pose_avg_homo)

    def _recenter_poses(self, poses):
        """Recenter poses according to the original NeRF code."""
        poses_ = poses.copy()
        bottom = np.reshape([0, 0, 0, 1.], [1, 4])
        c2w = self._poses_avg(poses)
        c2w = np.concatenate([c2w[:3, :4], bottom], -2)
        bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
        poses = np.concatenate([poses[:, :3, :4], bottom], -2)
        poses = np.linalg.inv(c2w) @ poses
        poses_[:, :3, :4] = poses[:, :3, :4]
        poses = poses_
        return poses, c2w

    def _poses_avg(self, poses):
        """Average poses according to the original NeRF code."""
        hwf = poses[0, :3, -1:]
        center = poses[:, :3, 3].mean(0)
        vec2 = self._normalize(poses[:, :3, 2].sum(0))
        up = poses[:, :3, 1].sum(0)
        c2w = np.concatenate([self._viewmatrix(vec2, up, center), hwf], 1)
        return c2w

    def _viewmatrix(self, z, up, pos):
        """Construct lookat view matrix."""
        vec2 = self._normalize(z)
        vec1_avg = up
        vec0 = self._normalize(np.cross(vec1_avg, vec2))
        vec1 = self._normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, pos], 1)
        return m

    def _normalize(self, x):
        """Normalization helper function."""
        return x / np.linalg.norm(x)

    def _generate_spiral_poses(self, poses, bds):
        """Generate a spiral path for rendering."""
        c2w = self._poses_avg(poses)
        # Get average pose.
        up = self._normalize(poses[:, :3, 1].sum(0))
        # Find a reasonable 'focus depth' for this dataset.
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))
        focal = mean_dz
        # Get radii for spiral path.
        tt = poses[:, :3, 3]
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        n_views = 120
        n_rots = 2
        # Generate poses for spiral path.
        render_poses = []
        rads = np.array(list(rads) + [1.])
        hwf = c2w_path[:, 4:5]
        zrate = .5
        for theta in np.linspace(0., 2. * np.pi * n_rots, n_views + 1)[:-1]:
            c = np.dot(c2w[:3, :4], (np.array(
                [np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads))
            z = self._normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
            render_poses.append(np.concatenate([self._viewmatrix(z, up, c), hwf], 1))
        self.render_poses = np.array(render_poses).astype(np.float32)[:, :3, :4]

    def _generate_spherical_poses(self, poses, bds):
        """Generate a 360 degree spherical path for rendering."""
        # pylint: disable=g-long-lambda
        p34_to_44 = lambda p: np.concatenate([
            p,
            np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])
        ], 1)
        rays_d = poses[:, :3, 2:3]
        rays_o = poses[:, :3, 3:4]

        def min_line_dist(rays_o, rays_d):
            a_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
            b_i = -a_i @ rays_o
            pt_mindist = np.squeeze(-np.linalg.inv(
                (np.transpose(a_i, [0, 2, 1]) @ a_i).mean(0)) @ (b_i).mean(0))
            return pt_mindist

        pt_mindist = min_line_dist(rays_o, rays_d)
        center = pt_mindist
        up = (poses[:, :3, 3] - center).mean(0)
        vec0 = self._normalize(up)
        vec1 = self._normalize(np.cross([.1, .2, .3], vec0))
        vec2 = self._normalize(np.cross(vec0, vec1))
        pos = center
        c2w = np.stack([vec1, vec2, vec0, pos], 1)
        poses_reset = (
                np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4]))
        rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))
        sc = 1. / rad
        poses_reset[:, :3, 3] *= sc
        bds *= sc
        rad *= sc
        centroid = np.mean(poses_reset[:, :3, 3], 0)
        zh = centroid[2]
        radcircle = np.sqrt(rad ** 2 - zh ** 2)
        new_poses = []

        for th in np.linspace(0., 2. * np.pi, 120):
            camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
            up = np.array([0, 0, -1.])
            vec2 = self._normalize(camorigin)
            vec0 = self._normalize(np.cross(vec2, up))
            vec1 = self._normalize(np.cross(vec2, vec0))
            pos = camorigin
            p = np.stack([vec0, vec1, vec2, pos], 1)
            new_poses.append(p)

        new_poses = np.stack(new_poses, 0)
        new_poses = np.concatenate([
            new_poses,
            np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)
        ], -1)
        poses_reset = np.concatenate([
            poses_reset[:, :3, :4],
            np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)
        ], -1)
        if self.split == 'test':
            self.render_poses = new_poses[:, :3, :4]
        return poses_reset


dataset_dict = {
    'carla_dyn': Carla,
    'waymo': Waymo,
}
