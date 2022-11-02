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
"""Utility functions."""
import collections
import os
from os import path
from absl import flags
import dataclasses
import flax
import gin
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image

gin.add_config_file_search_path('../')


gin.config.external_configurable(flax.nn.relu, module='flax.nn')
gin.config.external_configurable(flax.nn.sigmoid, module='flax.nn')
gin.config.external_configurable(flax.nn.softplus, module='flax.nn')


@flax.struct.dataclass
class TrainState:
  optimizer: flax.optim.Optimizer


@flax.struct.dataclass
class Stats:
  loss: float
  obj_losses: float
  losses: float
  d_losses: float
  n_losses: float
  e_losses: float
  s_losses: float
  #dx_losses: float
  distr_losses: float
  tv_losses: float
  #prop_losses: float
  #pd_loss: float
  #acc_loss: float
  #pose_loss: float
  #pose_reg: float
  offsets: float
  offset_x: float
  offset_y: float
  offset_z: float
  offset_yaw: float
  pose: float
  sampling_stats: float
  weights: float
  samples: float
  weight_l2: float
  psnr: float
  psnrs: float
  obj_psnr: float
  grad_norm: float
  grad_abs_max: float
  grad_norm_clipped: float


Rays = collections.namedtuple(
    'Rays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', 'delta'))

DynRays = collections.namedtuple(
    'DynRays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far', 'time', 'object'))
BoxRays = collections.namedtuple(
    'BoxRays',
    ('origins', 'directions', 'viewdirs', 'radii', 'lossmult', 'near', 'far'))


@gin.configurable()
@dataclasses.dataclass
class Config:
  """Configuration flags for everything."""
  dataset_loader: str = 'multicam'  # The type of dataset loader to use.
  batching: str = 'all_images'  # Batch composition, [single_image, all_images].
  batch_size: int = 4096  # The number of rays/pixels in each batch.
  factor: int = 0  # The downsample factor of images, 0 for no downsampling.
  spherify: bool = False  # Set to True for spherical 360 scenes.
  centering: bool = False  # this determines if poses are centered around zero or not
  random_box: bool = False  # this determines if boxes should be jittered randomly
  random_yaw: bool = False  # if noise should be added to bbox yaw / heading
  box_noise: float = 0.5  # the amount of random noise that should be added to box centers.
  yaw_noise: float = 5.  # rotational noise in degrees added to box yaw / heading angle
  render_path: bool = False  # If True, render a path. Used only by LLFF.
  llffhold: int = 8  # Use every Nth image for the test set. Used only by LLFF.
  timesteps: int = 5  # How many timesteps the current scene has (a bit of a crutch right now, integrate to dataset?)
  lr_init: float = 5e-4  # The initial learning rate.
  lr_final: float = 5e-6  # The final learning rate.
  lr_delay_steps: int = 2500  # The number of "warmup" learning steps.
  eps_delay_steps: int = 0  # The number of "warmup" learning steps.
  eps_init: int = 3  # Initial interval for near loss
  eps_final: int = 0.2  # Final interval for near loss
  eps_max_steps: int = 1000000  # Number of steps after which eps will reach eps_final
  l2_reg: bool = False  # If poses should be regularised between optimisation steps
  alpha_init: float = 0.0  # the inital number of frequencies used for encoding
  alpha_final: float = 10.0  # final number of frequencies used in encoding, should be same as model.max_deg
  alpha_delay_steps: int = 0  # after this many steps start increasing alpha
  alpha_max_steps: int = 1000000  # after this many steps alpha will be alpha_final
  psreg_init: float = 10e5  # start value for pose regularization
  psreg_final: float = 10e-1  # end value for pose regularization
  psreg_delay_steps: int = 5000  # after this many steps, start decreasing psreg
  psreg_delay_mult: float = 1.0
  tv_loss_mult: float = 0.0001
  depth_loss_mult: float = 0.0001
  near_loss_mult: float = 0.01
  empty_loss_mult: float = 1.0
  sky_loss_mult: float = 1.0
  c2f_steps: list = (5000, 10000, 15000)  # The number of steps after which rays of higher resolutions should be loaded
  lr_delay_mult: float = 0.01  # How much sever the "warmup" should be.
  grad_max_norm: float = 0.  # Gradient clipping magnitude, disabled if == 0.
  grad_max_val: float = 0.  # Gradient clipping value, disabled if == 0.
  max_steps: int = 1000000  # The number of optimization steps.
  save_every: int = 100000  # The number of steps to save a checkpoint.
  print_every: int = 100  # The number of steps between reports to tensorboard.
  gc_every: int = 10000  # The number of steps between garbage collections.
  test_render_interval: int = 1  # The interval between images saved to disk.
  disable_multiscale_loss: bool = False  # If True, disable multiscale loss.
  randomized: bool = True  # Use randomized stratified sampling.
  near: float = 2.  # Near plane distance.
  far: float = 6.  # Far plane distance.
  coarse_loss_mult: float = 0.1  # How much to downweight the coarse loss(es).
  box_loss_mult: int = 0  # How much more the rays that terminate inside the box should be weighted in the rgb loss
  weight_decay_mult: float = 0.  # The multiplier on weight decay.
  white_bkgd: bool = False  # If True, use white as the background (black o.w.).
  rand_bkgd: bool = True  # If True, use random color as background


def define_common_flags():
  # Define the flags used by both train.py and eval.py
  flags.DEFINE_multi_string('gin_file', None,
                            'List of paths to the config files.')
  flags.DEFINE_multi_string(
      'gin_param', None, 'Newline separated list of Gin parameter bindings.')
  flags.DEFINE_string('train_dir', None, 'where to store ckpts and logs')
  flags.DEFINE_string('data_dir', None, 'input data directory.')
  flags.DEFINE_string('eval_dir', None, 'where to render traj to.')
  flags.DEFINE_integer(
      'chunk', 8192,
      'the size of chunks for evaluation inferences, set to the value that'
      'fits your GPU/TPU memory.')


def load_config():
  gin.parse_config_files_and_bindings(flags.FLAGS.gin_file,
                                      flags.FLAGS.gin_param)
  return Config()


def open_file(pth, mode='r'):
  return open(pth, mode=mode)


def file_exists(pth):
  return path.exists(pth)


def listdir(pth):
  return os.listdir(pth)


def isdir(pth):
  return path.isdir(pth)


def makedirs(pth):
  os.makedirs(pth)


def namedtuple_map(fn, tup):
  """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
  return type(tup)(*map(fn, tup))


def shard(xs):
  """Split data into shards for multiple devices along the first dimension."""
  return jax.tree_map(
      lambda x: x.reshape((jax.local_device_count(), -1) + x.shape[1:]), xs)


def to_device(xs):
  """Transfer data to devices (GPU/TPU)."""
  return jax.tree_map(jnp.array, xs)


def unshard(x, padding=0):
  """Collect the sharded tensor to the shape before sharding."""
  y = x.reshape([x.shape[0] * x.shape[1]] + list(x.shape[2:]))
  if padding > 0:
    y = y[:-padding]
  return y


def save_img_uint8(img, pth):
  """Save an image (probably RGB) in [0, 1] to disk as a uint8 PNG."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(
        (np.clip(np.nan_to_num(img), 0., 1.) * 255.).astype(jnp.uint8)).save(
            f, 'PNG')


def save_img_float32(depthmap, pth):
  """Save an image (probably a depthmap) to disk as a float32 TIFF."""
  with open_file(pth, 'wb') as f:
    Image.fromarray(np.nan_to_num(depthmap).astype(np.float32)).save(f, 'TIFF')
