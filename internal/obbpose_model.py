# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Different model implementation plus a general port for all the models."""
import functools
from functools import partial
from typing import Any, Callable
from flax import linen as nn
import gin
import jax
from jax import random
from jax import lax
import jax.numpy as jnp
import numpy as np

from internal import mip
from internal import mip360
from internal import math
from internal import utils
from internal import box_helpers


def init_boxes(rng, box_centers):
    if len(box_centers.shape) < 3:
        return box_centers[:, None, :]
    else:
        return box_centers


@gin.configurable
class MipNerfModel(nn.Module):
    """Nerf NN Model with both coarse and fine MLPs."""
    num_samples: int = 128  # The number of samples per level.
    num_levels: int = 2  # The number of sampling levels.
    resample_padding: float = 0.01  # Dirichlet/alpha "padding" on the histogram.
    stop_level_grad: bool = True  # If True, don't backprop across levels')
    use_viewdirs: bool = True  # If True, use view directions as a condition.
    lindisp: bool = False  # If True, sample linearly in disparity, not in depth.
    ray_shape: str = 'cone'  # The shape of cast rays ('cone' or 'cylinder').
    min_deg_point: int = 0  # Min degree of positional encoding for 3D points.
    max_deg_point: int = 10  # Max degree of positional encoding for 3D points.
    deg_view: int = 4  # Degree of positional encoding for viewdirs.
    num_objects: int = 2  # Number of moving objects in the scene
    density_activation: Callable[..., Any] = nn.softplus  # Density activation.
    density_noise: float = 0.1  # Standard deviation of noise added to raw density.
    density_bias: float = -1.  # The shift added to raw densities pre-activation.
    rgb_activation: Callable[..., Any] = nn.sigmoid  # The RGB activation.
    rgb_padding: float = 0.001  # Padding added to the RGB outputs.
    disable_integration: bool = False  # If True, use PE instead of IPE.
    contraction: bool = True  # If True, use the contraction from mipnerf360
    dynamics: bool = True  # If True, use the deformation network
    timesteps: int = 5  # Number of timesteps in the input data
    no_pose_opt: bool = False  # if box pose should get gradients or not
    no_yaw_opt: bool = False  # if box yaw should get gradients or not

    @nn.compact
    def __call__(self, rng, rays, init, ext, ts, randomized, rand_bkgd, white_bkgd, alpha):
        """The mip-NeRF Model.

        Args:
          rng: jnp.ndarray, random number generator.
          rays: util.Rays, a namedtuple of ray origins, directions, and viewdirs.
          randomized: bool, use randomized stratified sampling.
          white_bkgd: bool, if True, use white as the background (black o.w.).

        Returns:
          ret: list, [*(rgb, distance, acc)]
        """
        # Construct the MLPs.
        mlp = MLP()
        if len(init.shape) < 3:
            N_obj = 1
        else:
            N_obj = init.shape[1]

        pose_offsets = self.param('box_centers', init_boxes, init)  # this contains [ts, N_obj, [x,y,z,yaw]]

        obj_mlps = []
        for n in range(N_obj):
            obj_mlps.append(BoxMLP())

        origins = rays.origins
        dirs = rays.directions

        B = origins.shape[0]

        box_pose = jnp.broadcast_to(pose_offsets[ts.squeeze(), :, :3], [B, N_obj, 3])
        if self.no_pose_opt:
            box_pose = lax.stop_gradient(box_pose)
        box_rot = pose_offsets[ts.squeeze(), :, 3:]
        if self.no_yaw_opt:
            box_rot = lax.stop_gradient(box_rot)
        box_mat = box_helpers.aa2matrix(box_rot)
        box_mat = jnp.broadcast_to(box_mat, [B, N_obj, 3, 3])

        box_dims = jnp.broadcast_to(ext, [B, N_obj, 3])

        origins_o, dirs_o = box_helpers.world2object_rpy(origins, dirs, box_pose, box_mat)

        zi, zo, intersection = box_helpers.ray_box_intersection(origins_o, dirs_o, -box_dims, box_dims)
        intersection = lax.stop_gradient(intersection)

        bkgd_mask = (intersection.sum(axis=-1) == 0).astype(jnp.float32)

        obj_pts = origins_o * intersection[..., None]
        obj_dirs = dirs_o * intersection[..., None]

        # TODO: this assumes that objects do not occlude each other, could be fixed with zi, zo
        origins_s = obj_pts.sum(axis=-2) + bkgd_mask[..., None] * origins
        dirs_s = obj_dirs.sum(axis=-2) + bkgd_mask[..., None] * dirs


        # near, far can be used to restrict ray sampling between these values.
        near = (intersection * (zi - 5.0)).sum(axis=-1)[..., None] + bkgd_mask[..., None] * rays.near
        near = lax.stop_gradient(near)
        far = (intersection * (zo + 5.0)).sum(axis=-1)[..., None] + bkgd_mask[..., None] * rays.far
        far = lax.stop_gradient(far)

        zo_ret = (intersection * zo).sum(axis=-1)

        ret = []
        for i_level in range(self.num_levels):
            key, rng = random.split(rng)
            if i_level == 0:
                # Stratified sampling along rays
                t_vals, samples = mip.sample_along_rays(
                    key,
                    origins_s,
                    dirs_s,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.lindisp,
                    self.ray_shape,
                )
            else:
                t_vals, samples = mip.resample_along_rays(
                    key,
                    origins_s,
                    dirs_s,
                    rays.radii,
                    t_vals,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_level_grad,
                    resample_padding=self.resample_padding,
                )

            if self.disable_integration:
                samples = (samples[0], jnp.zeros_like(samples[1]))

            if self.dynamics:
                B, N, _ = samples[0].shape

                masks = []
                raw_rgbs = []
                raw_densities = []
                ret_masks = []
                for i in range(N_obj):
                    mask = intersection[:, i].reshape(-1, 1)
                    ret_masks.append(mask)
                    mask = jnp.broadcast_to(mask[:, None, :], [B, N, 1])
                    masks.append(mask)
                    obj_mean = mask * samples[0]
                    obj_var = mask[..., None] * samples[1]
                    obj_samples = (obj_mean, obj_var)

                    #if self.contraction:
                    #    obj_samples = mip360.new_space(obj_samples)
                    obj_samples_enc = mip.weighted_ipe(
                        obj_samples,
                        self.min_deg_point,
                        self.max_deg_point,
                        alpha=alpha
                    )

                    if self.use_viewdirs:
                        viewdirs_enc = mip.pos_enc(
                            rays.viewdirs,
                            min_deg=0,
                            max_deg=self.deg_view,
                            append_identity=True,
                        )
                    obj_rgb, obj_density = obj_mlps[i](obj_samples_enc, viewdirs_enc)
                    raw_rgbs.append(mask * obj_rgb)
                    raw_densities.append(mask * obj_density)

                raw_rgbs = jnp.array(raw_rgbs).sum(axis=0)
                raw_densities = jnp.array(raw_densities).sum(axis=0)
                bkgd_mask = 1 - jnp.array(masks).sum(axis=0)
                bkgd_mask = lax.stop_gradient(bkgd_mask)
                mean = bkgd_mask * samples[0]
                bkgd_var = bkgd_mask[..., None] * samples[1]

                samples = (mean, bkgd_var)

            if self.contraction:
                samples = mip360.new_space(samples)
            samples_enc = mip.integrated_pos_enc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
            )

            # Point attribute predictions
            if self.use_viewdirs:
                viewdirs_enc = mip.pos_enc(
                    rays.viewdirs,
                    min_deg=0,
                    max_deg=self.deg_view,
                    append_identity=True,
                )
                raw_rgb, raw_density = mlp(samples_enc, viewdirs_enc)
            else:
                raw_rgb, raw_density = mlp(samples_enc)

            if self.dynamics:
                raw_rgb += raw_rgbs
                raw_density += raw_densities

            # Add noise to regularize the density predictions if needed.
            if randomized and (self.density_noise > 0):
                key, rng = random.split(rng)
                raw_density += self.density_noise * random.normal(
                    key, raw_density.shape, dtype=raw_density.dtype)

            # Volumetric rendering.
            rgb = self.rgb_activation(raw_rgb)
            # rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)
            comp_rgb, distance, acc, weights, t_vals, t_mids, t_dists = mip.volumetric_rendering(
                rgb,
                density,
                t_vals,
                dirs_s,
                white_bkgd=white_bkgd,
                rand_bkgd=rand_bkgd,
                key=key
            )
            # if self.lindisp:
            #    distance = 1.0 / distance
            if self.dynamics:
                ret.append((comp_rgb, distance, acc, weights, t_vals, t_mids, t_dists, [box_pose[0], box_rot[0]], jnp.array(ret_masks).sum(axis=0), zo_ret))
            else:
                ret.append((comp_rgb, distance, acc, weights, t_vals, t_mids, t_dists, [box_pose[0], box_rot[0]], intersection.sum(axis=-1)[..., None], zo_ret))
        return ret


def construct_mipnerf(rng, example_batch):
    """Construct a Neural Radiance Field.

    Args:
      rng: jnp.ndarray. Random number generator.
      example_batch: dict, an example of a batch of data.

    Returns:
      model: nn.Model. Nerf model with parameters.
      state: flax.Module.state. Nerf model state for stateful parameters.
    """
    model = MipNerfModel()
    key, rng = random.split(rng)
    ext = example_batch['ext'].squeeze()
    init = example_batch['init'].squeeze()

    init_variables = model.init(
        key,
        rng=rng,
        rays=utils.namedtuple_map(lambda x: x[0], example_batch['rays']),
        init=init,
        ext=ext,
        ts=example_batch['ts'],
        randomized=False,
        rand_bkgd=True,
        white_bkgd=False,
        alpha=0.0)
    return model, init_variables

@gin.configurable
class MLP(nn.Module):
    """A simple MLP."""
    net_depth: int = 8  # The depth of the first part of MLP.
    net_width: int = 256  # The width of the first part of MLP.
    net_depth_condition: int = 1  # The depth of the second part of MLP.
    net_width_condition: int = 128  # The width of the second part of MLP.
    net_activation: Callable[..., Any] = nn.relu  # The activation function.
    skip_layer: int = 4  # Add a skip connection to the output of every N layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    num_density_channels: int = 1  # The number of density channels.

    @nn.compact
    def __call__(self, x, condition=None):
        """Evaluate the MLP.

        Args:
          x: jnp.ndarray(float32), [batch, num_samples, feature], points.
          condition: jnp.ndarray(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
          raw_rgb: jnp.ndarray(float32), with a shape of
               [batch, num_samples, num_rgb_channels].
          raw_density: jnp.ndarray(float32), with a shape of
               [batch, num_samples, num_density_channels].
        """
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        x = x.reshape([-1, feature_dim])
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
        inputs = x
        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)
        raw_density = dense_layer(self.num_density_channels)(x).reshape(
            [-1, num_samples, self.num_density_channels])

        if condition is not None:
            # Output of the first part of MLP.
            bottleneck = dense_layer(self.net_width)(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            condition = jnp.tile(condition[:, None, :], (1, num_samples, 1))
            # Collapse the [batch, num_samples, feature] tensor to
            # [batch * num_samples, feature] so that it can be fed into nn.Dense.
            condition = condition.reshape([-1, condition.shape[-1]])
            x = jnp.concatenate([bottleneck, condition], axis=-1)
            # Here use 1 extra layer to align with the original nerf model.
            for i in range(self.net_depth_condition):
                x = dense_layer(self.net_width_condition)(x)
                x = self.net_activation(x)
        raw_rgb = dense_layer(self.num_rgb_channels)(x).reshape(
            [-1, num_samples, self.num_rgb_channels])
        return raw_rgb, raw_density


@gin.configurable
class BoxMLP(nn.Module):
    """A simple MLP."""
    net_depth: int = 8  # The depth of the first part of MLP.
    net_width: int = 128  # The width of the first part of MLP.
    net_depth_condition: int = 1  # The depth of the second part of MLP.
    net_width_condition: int = 128  # The width of the second part of MLP.
    net_activation: Callable[..., Any] = nn.relu  # The activation function.
    skip_layer: int = 4  # Add a skip connection to the output of every N layers.
    num_rgb_channels: int = 3  # The number of RGB channels.
    num_density_channels: int = 1  # The number of density channels.

    @nn.compact
    def __call__(self, x, condition=None):
        """Evaluate the MLP.

        Args:
          x: jnp.ndarray(float32), [batch, num_samples, feature], points.
          condition: jnp.ndarray(float32), [batch, feature], if not None, this
            variable will be part of the input to the second part of the MLP
            concatenated with the output vector of the first part of the MLP. If
            None, only the first part of the MLP will be used with input x. In the
            original paper, this variable is the view direction.

        Returns:
          raw_rgb: jnp.ndarray(float32), with a shape of
               [batch, num_samples, num_rgb_channels].
          raw_density: jnp.ndarray(float32), with a shape of
               [batch, num_samples, num_density_channels].
        """
        feature_dim = x.shape[-1]
        num_samples = x.shape[1]
        x = x.reshape([-1, feature_dim])
        dense_layer = functools.partial(
            nn.Dense, kernel_init=jax.nn.initializers.glorot_uniform())
        inputs = x
        for i in range(self.net_depth):
            x = dense_layer(self.net_width)(x)
            x = self.net_activation(x)
            if i % self.skip_layer == 0 and i > 0:
                x = jnp.concatenate([x, inputs], axis=-1)
        raw_density = dense_layer(self.num_density_channels)(x).reshape(
            [-1, num_samples, self.num_density_channels])

        if condition is not None:
            # Output of the first part of MLP.
            bottleneck = dense_layer(self.net_width)(x)
            # Broadcast condition from [batch, feature] to
            # [batch, num_samples, feature] since all the samples along the same ray
            # have the same viewdir.
            condition = jnp.tile(condition[:, None, :], (1, num_samples, 1))
            # Collapse the [batch, num_samples, feature] tensor to
            # [batch * num_samples, feature] so that it can be fed into nn.Dense.
            condition = condition.reshape([-1, condition.shape[-1]])
            x = jnp.concatenate([bottleneck, condition], axis=-1)
            # Here use 1 extra layer to align with the original nerf model.
            for i in range(self.net_depth_condition):
                x = dense_layer(self.net_width_condition)(x)
                x = self.net_activation(x)
        raw_rgb = dense_layer(self.num_rgb_channels)(x).reshape(
            [-1, num_samples, self.num_rgb_channels])
        return raw_rgb, raw_density


def render_image(render_fn, rays, init, ext, ts, rng, alpha, chunk=8192):
    """Render all the pixels of an image (in test mode).

    Args:
      render_fn: function, jit-ed render function.
      rays: a `Rays` namedtuple, the rays to be rendered.
      rng: jnp.ndarray, random number generator (used in training mode only).
      chunk: int, the size of chunks to render sequentially.

    Returns:
      rgb: jnp.ndarray, rendered color image.
      disp: jnp.ndarray, rendered disparity image.
      acc: jnp.ndarray, rendered accumulated weights per pixel.
    """
    height, width = rays[0].shape[:2]
    num_rays = height * width


    def reshape(r):
        if r.shape[-1] == 4:
            return r.reshape((num_rays, 4, 4))
        else:
            return r.reshape((num_rays, -1))

    rays = utils.namedtuple_map(lambda r: reshape(r), rays)
    batch = {}

    host_id = jax.host_id()
    results = []
    for i in range(0, num_rays, chunk):
        # pylint: disable=cell-var-from-loop
        chunk_rays = utils.namedtuple_map(lambda r: r[i:i + chunk], rays)
        chunk_size = chunk_rays[0].shape[0]
        rays_remaining = chunk_size % jax.device_count()
        if rays_remaining != 0:
            padding = jax.device_count() - rays_remaining
            chunk_rays = utils.namedtuple_map(
                lambda r: jnp.pad(r, ((0, padding), (0, 0)), mode='edge'), chunk_rays)
        else:
            padding = 0
        # After padding the number of chunk_rays is always divisible by
        # host_count.
        rays_per_host = chunk_rays[0].shape[0] // jax.host_count()
        start, stop = host_id * rays_per_host, (host_id + 1) * rays_per_host
        chunk_rays = utils.namedtuple_map(lambda r: utils.shard(r[start:stop]),
                                          chunk_rays)
        batch['rays'] = chunk_rays
        batch['init'] = jnp.expand_dims(init, 0)
        batch['ext'] = jnp.expand_dims(ext, 0)
        batch['ts'] = jnp.expand_dims(ts, 0)
        batch['alpha'] = jnp.expand_dims(alpha, 0)
        chunk_results = render_fn(rng, batch)[-1]
        results.append([utils.unshard(x[0], padding) for x in chunk_results])
        # pylint: enable=cell-var-from-loop
    rgb, distance, acc, weights, tvals, tmids, t_dists, off, masks, zo = [jnp.concatenate(r, axis=0) for r in zip(*results)]
    rgb = rgb.reshape((height, width, -1))
    distance = distance.reshape((height, width))
    acc = acc.reshape((height, width))
    return (rgb, distance, acc)
