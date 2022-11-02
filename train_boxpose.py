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
"""Training script for Nerf."""

import functools
import gc
import time
from absl import app
from absl import flags
import flax
from flax.core import freeze, unfreeze
from flax.metrics import tensorboard
from flax.training import checkpoints
import jax
from jax import random
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import queue

from internal import obbpose_dataset
from internal import c2f_obb_dataset
from internal import math
from internal import obbpose_model
from internal import utils
from internal import vis

FLAGS = flags.FLAGS
utils.define_common_flags()
flags.DEFINE_integer('render_every', 5000,
                     'The number of steps between test set image renderings.')

jax.config.parse_flags_with_absl()


def train_step(model, config, rng, state, batch, lr, eps, alpha, prev):
    """One optimization step.

    Args:
      model: The linen model.
      config: The configuration.
      rng: jnp.ndarray, random number generator.
      state: utils.TrainState, state of the model/optimizer.
      batch: dict, a mini-batch of data for training.
      lr: float, real-time learning rate.

    Returns:
      new_state: utils.TrainState, new training state.
      stats: list. [(loss, psnr), (loss_coarse, psnr_coarse)].
      rng: jnp.ndarray, updated random number generator.
    """
    rng, key = random.split(rng)

    def loss_fn(variables):

        def tree_sum_fn(fn):
            return jax.tree_util.tree_reduce(
                lambda x, y: x + fn(y), variables, initializer=0)

        weight_l2 = config.weight_decay_mult * (
                tree_sum_fn(lambda z: jnp.sum(z ** 2)) /
                tree_sum_fn(lambda z: jnp.prod(jnp.array(z.shape))))

        rays = batch['rays']
        init = batch['init']
        ext = batch['ext']
        ts = batch['ts']

        ret = model.apply(
            variables,
            key,
            rays,
            init,
            ext,
            ts,
            randomized=config.randomized,
            rand_bkgd=config.rand_bkgd,
            white_bkgd=config.white_bkgd,
            alpha=alpha)

        mask = batch['rays'].lossmult
        if config.disable_multiscale_loss:
            mask = jnp.ones_like(mask)

        depth_mask = (batch['depth'].squeeze() > 0.0).astype(jnp.float32)
        sky_mask = (batch['sky'].squeeze() > 0.0).astype(jnp.float32)
        # combined_mask = ((batch['depth'].squeeze() > 0.0) + (batch['sky'].squeeze() > 0.0)).astype(jnp.float32)
        overlap = depth_mask * sky_mask
        sky_mask -= overlap
        # sky_mask = sky_mask.reshape(batch['sky'].shape[0], 1)
        # inv_sky_mask = 1 - sky_mask

        losses = []
        obj_losses = []
        d_losses = []
        distr_losses = []
        tv_losses = []
        s_losses = []
        e_losses = []
        n_losses = []
        sampling_stats = []
        log_weights = []
        log_tvals = []
        offsets = []
        offset_z = []
        offset_x = []
        offset_y = []
        offset_yaw = []

        for (rgb, depth, _, weights, tvals, tmids, t_dists, off, dyn_mask, zo) in ret:
            sampling_stats.append(tvals[0, 0])
            sampling_stats.append(tvals[0, -1])
            log_weights.append(weights)
            log_tvals.append(tvals)

            pose, yaw = off
            offsets.append(((pose - batch['target'][:, :3])**2).sum())
            offset_x.append(((pose[:, 0] - batch['target'][:, 0]) ** 2).sum())
            offset_y.append(((pose[:, 1] - batch['target'][:, 1]) ** 2).sum())
            offset_z.append(((pose[:, 2] - batch['target'][:, 2]) ** 2).sum())
            offset_yaw.append(((yaw - batch['target'][:, 3:])**2).sum())

            tv_losses.append(((pose - prev[:, :, :3]) ** 2).sum())

            box_mask = (batch['depth'].squeeze() < zo).astype(jnp.float32)

            depth_mask = (depth_mask + config.box_loss_mult*dyn_mask.squeeze()*box_mask)

            # print('tvals:',tvals[0,:])
            # print(weights[0,:])
            # print(jnp.argmax(weights[0,:]))
            tvals = tvals[:, :-1]
            Wij = weights[..., :, None] * weights[..., None, :]
            si = jnp.moveaxis(tmids, -1, 0)
            si = jnp.moveaxis(jnp.resize(si, (tmids.shape[1], tmids.shape[1], tmids.shape[0])), -1, 0)
            sj = jnp.moveaxis(si, -1, 1)
            Sij = jnp.abs(si - sj)
            term1 = (Wij * Sij).sum()
            term2 = (1 / 3) * (weights ** 2 * t_dists).sum()
            distr_losses.append(term1 + term2)

            depth_t = jnp.broadcast_to(batch['depth'], tvals.shape)
            sigma = (eps / 3.) ** 2
            t_from_ndc = 1.0 / (1.0 - tvals)
            mask_near = ((tvals > (depth_t - eps)) & (tvals < (depth_t + eps))).astype(jnp.float32)
            mask_near *= depth_mask.reshape(tvals.shape[0], -1)
            mask_empty = (tvals > (depth_t + eps)).astype(jnp.float32)
            mask_empty *= depth_mask.reshape(tvals.shape[0], -1)
            dist = mask_near * (tvals - depth_t)
            distr = 1.0 / (sigma * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-(dist ** 2 / (2 * sigma ** 2)))
            distr /= distr.max()
            distr *= mask_near
            n_losses.append(((mask_near * weights - distr) ** 2).sum() / jnp.maximum(depth_mask.sum(), 1.0))
            e_losses.append(((mask_empty * weights) ** 2).sum() / jnp.maximum(depth_mask.sum(), 1.0))
            # seg_losses.append(((sky_mask.reshape(weights.shape[0], -1) * weights)**2).sum() / jnp.maximum(sky_mask.sum(), 1.0))
            # seg_losses.append((sky_mask * depth).mean())
            # z_from_ndc = depth_mask * (1.0 / (1.0 - depth_mask * depth))
            # print(jnp.max(z_from_ndc))
            # print(jnp.max(depth))
            # inv_depth = depth_mask * (1.0 / jnp.maximum(batch['depth'].squeeze(), 1.0))
            d_losses.append(
                (depth_mask * (depth - batch['depth'].squeeze()) ** 2).sum() / jnp.maximum(depth_mask.sum(), 1.0))

            # calculate total variance in depth for neighbouring consecutive rays
            batch_var = ((depth[:-1] - depth[1:]) ** 2).sum()
            #tv_losses.append(jnp.max(depth))

            # rgb = inv_sky_mask * rgb + sky_mask * sky_rgb
            # print('depth:', depth_mask * depth)
            # print('gtdepth:', batch['depth'].squeeze())
            # print(tvals)
            # print(jnp.min(depth))
            sky_depth = sky_mask * (1.0 - (1.0 / jnp.maximum((sky_mask * depth), 1.0)))
            # print(jnp.max(sky_depth))
            s_losses.append(
                (sky_mask * (sky_depth - batch['sky'].squeeze()) ** 2).sum() / jnp.maximum(sky_mask.sum(), 1.0))

            losses.append(((mask + config.box_loss_mult*dyn_mask*box_mask[..., None]) * (rgb - batch['pixels'][..., :3]) ** 2).sum() / mask.sum())
            obj_losses.append((dyn_mask * (rgb - batch['pixels'][..., :3])**2).sum() / dyn_mask.sum())

        losses = jnp.array(losses)
        obj_losses = jnp.array(obj_losses)
        d_losses = jnp.array(d_losses)
        distr_losses = jnp.array(distr_losses)
        tv_losses = jnp.array(tv_losses)
        n_losses = jnp.array(n_losses)
        e_losses = jnp.array(e_losses)
        s_losses = jnp.array(s_losses)
        sampling_stats = jnp.array(sampling_stats)
        log_weights = jnp.array(log_weights)
        log_tvals = jnp.array(log_tvals)
        offsets = jnp.array(offsets)
        offset_x = jnp.array(offset_x)
        offset_y = jnp.array(offset_y)
        offset_z = jnp.array(offset_z)
        offset_yaw = jnp.array(offset_yaw)

        loss = (config.coarse_loss_mult * jnp.sum(losses[:-1]) + losses[-1] + weight_l2)
        #loss += (5.0 * obj_losses[-1] + 5.0 * jnp.sum(obj_losses[:-1]))
        loss += config.sky_loss_mult * jnp.sum(s_losses[:-1]) + 10.0 * config.sky_loss_mult * s_losses[-1]

        loss += config.depth_loss_mult * d_losses[-1] + 0.1 * config.depth_loss_mult * jnp.sum(d_losses[:-1])
        loss += config.near_loss_mult * n_losses[-1] + 0.1 * config.near_loss_mult * jnp.sum(n_losses[:-1])
        loss += config.empty_loss_mult * e_losses[-1] + 0.1 * config.empty_loss_mult * jnp.sum(e_losses[:-1])

        loss += config.tv_loss_mult * tv_losses[-1] + 0.1 * config.tv_loss_mult * jnp.sum(tv_losses[:-1])
        loss += 0.000001 * distr_losses[-1] + 0.000001 * jnp.sum(distr_losses[:-1])

        stats = utils.Stats(
            loss=loss,
            obj_losses=obj_losses,
            losses=losses,
            d_losses=d_losses,
            n_losses=n_losses,
            e_losses=e_losses,
            s_losses=s_losses,
            distr_losses=distr_losses,
            tv_losses=tv_losses,
            sampling_stats=sampling_stats,
            offsets=offsets,
            offset_x=offset_x,
            offset_y=offset_y,
            offset_z=offset_z,
            offset_yaw=offset_yaw,
            pose=pose,
            weights=log_weights,
            samples=log_tvals,
            weight_l2=weight_l2,
            psnr=0.0,
            psnrs=0.0,
            obj_psnr=0.0,
            grad_norm=0.0,
            grad_abs_max=0.0,
            grad_norm_clipped=0.0,
        )
        return loss, stats

    (_, stats), grad = (
        jax.value_and_grad(loss_fn, has_aux=True)(state.optimizer.target))
    grad = jax.lax.pmean(grad, axis_name='batch')
    pose = stats.pose
    stats = jax.lax.pmean(stats, axis_name='batch')

    def tree_norm(tree):
        return jnp.sqrt(
            jax.tree_util.tree_reduce(
                lambda x, y: x + jnp.sum(y ** 2), tree, initializer=0))

    eps = 1e-6
    nan_fn = lambda g: jnp.nan_to_num(g, eps, posinf=0.0)
    grad = jax.tree_util.tree_map(nan_fn, grad)

    """
    # This will set all gradients for the mlps to zero
    freeze_fn = lambda g: 0.0 * g
    grad = unfreeze(grad)
    grad['params']['BoxMLP_0'] = jax.tree_util.tree_map(freeze_fn, grad['params']['BoxMLP_0'])
    grad['params']['MLP_0'] = jax.tree_util.tree_map(freeze_fn, grad['params']['MLP_0'])
    grad = freeze(grad)
    """

    if config.grad_max_val > 0:
        clip_fn = lambda z: jnp.clip(z, -config.grad_max_val, config.grad_max_val)
        grad = jax.tree_util.tree_map(clip_fn, grad)

    grad_abs_max = jax.tree_util.tree_reduce(
        lambda x, y: jnp.maximum(x, jnp.max(jnp.abs(y))), grad, initializer=0)

    grad_norm = tree_norm(grad)
    if config.grad_max_norm > 0:
        mult = jnp.minimum(1, config.grad_max_norm / (1e-7 + grad_norm))
        grad = jax.tree_util.tree_map(lambda z: mult * z, grad)
    grad_norm_clipped = tree_norm(grad)

    new_optimizer = state.optimizer.apply_gradient(grad, learning_rate=lr)
    new_state = state.replace(optimizer=new_optimizer)

    psnrs = math.mse_to_psnr(stats.losses)
    obj_psnrs = math.mse_to_psnr((stats.obj_losses))
    stats = utils.Stats(
        loss=stats.loss,
        obj_losses=stats.obj_losses,
        losses=stats.losses,
        d_losses=stats.d_losses,
        n_losses=stats.n_losses,
        e_losses=stats.e_losses,
        s_losses=stats.s_losses,
        distr_losses=stats.distr_losses,
        tv_losses=stats.tv_losses,
        sampling_stats=stats.sampling_stats,
        offsets=stats.offsets,
        offset_x=stats.offset_x,
        offset_y=stats.offset_y,
        offset_z=stats.offset_z,
        offset_yaw=stats.offset_yaw,
        pose=stats.pose,
        weights=stats.weights,
        samples=stats.samples,
        weight_l2=stats.weight_l2,
        psnr=psnrs[-1],
        psnrs=psnrs,
        obj_psnr=obj_psnrs[-1],
        grad_norm=grad_norm,
        grad_abs_max=grad_abs_max,
        grad_norm_clipped=grad_norm_clipped,
    )

    return new_state, stats, rng, pose


def main(unused_argv):
    rng = random.PRNGKey(20200823)
    # Shift the numpy random seed by host_id() to shuffle data loaded by different
    # hosts.
    np.random.seed(20201473)

    config = utils.load_config()

    if config.batch_size % jax.device_count() != 0:
        raise ValueError('Batch size must be divisible by the number of devices.')

    dataset = obbpose_dataset.get_dataset('train', FLAGS.data_dir, config)
    test_dataset = obbpose_dataset.get_dataset('test', FLAGS.data_dir, config)

    rng, key = random.split(rng)
    model, variables = obbpose_model.construct_mipnerf(key, dataset.peek())
    num_params = jax.tree_util.tree_reduce(
        lambda x, y: x + jnp.prod(jnp.array(y.shape)), variables, initializer=0)
    print(f'Number of parameters being optimized: {num_params}')
    optimizer = flax.optim.Adam(config.lr_init).create(variables)
    state = utils.TrainState(optimizer=optimizer)
    del optimizer, variables

    learning_rate_fn = functools.partial(
        math.learning_rate_decay,
        lr_init=config.lr_init,
        lr_final=config.lr_final,
        max_steps=config.max_steps,
        lr_delay_steps=config.lr_delay_steps,
        lr_delay_mult=config.lr_delay_mult)

    eps_rate_fn = functools.partial(
        math.learning_rate_decay,
        lr_init=config.eps_init,
        lr_final=config.eps_final,
        max_steps=config.eps_max_steps,
        lr_delay_steps=config.eps_delay_steps,
        lr_delay_mult=config.lr_delay_mult)

    alpha_rate_fn = functools.partial(
        math.freq_alpha_rate,
        alpha_init=config.alpha_init,
        alpha_final=config.alpha_final,
        alpha_delay_steps=config.alpha_delay_steps,
        alpha_max_steps=config.alpha_max_steps)

    train_pstep = jax.pmap(
        functools.partial(train_step, model, config),
        axis_name='batch',
        in_axes=(0, 0, 0, None, None, None, None),
        donate_argnums=(2,))

    # Because this is only used for test set rendering, we disable randomization.
    def render_eval_fn(variables, _, batch):
        return jax.lax.all_gather(
            model.apply(
                variables,
                random.PRNGKey(0),  # Unused.
                batch['rays'],
                batch['init'],
                batch['ext'],
                batch['ts'],
                randomized=False,
                white_bkgd=config.white_bkgd,
                rand_bkgd=False,
                alpha=batch['alpha']),
            axis_name='batch')

    render_eval_pfn = jax.pmap(
        render_eval_fn,
        in_axes=(None, None, 0),  # Only distribute the data input.
        donate_argnums=(2,),
        axis_name='batch',
    )

    ssim_fn = jax.jit(functools.partial(math.compute_ssim, max_val=1.))

    if not utils.isdir(FLAGS.train_dir):
        utils.makedirs(FLAGS.train_dir)
    #traindir = '/home/tristram/nerf_results/Carla_rgb_0_200_8_256_10fipe_obb_noopt_mat_nonoise_baseline/100_no_bkgd/'
    state = checkpoints.restore_checkpoint(FLAGS.train_dir, state)
    # Resume training at the step of the last checkpoint.
    init_step = state.optimizer.state.step + 1
    state = flax.jax_utils.replicate(state)

    if jax.host_id() == 0:
        summary_writer = tensorboard.SummaryWriter(FLAGS.train_dir)

    # Prefetch_buffer_size = 3 x batch_size
    pdataset = flax.jax_utils.prefetch_to_device(dataset, 3)
    #rng = rng + jax.host_id()  # Make random seed separate across hosts.
    keys = random.split(rng, jax.local_device_count())  # For pmapping RNG keys.
    gc.disable()  # Disable automatic garbage collection for efficiency.
    stats_trace = []
    prevs = dataset.peek()['init'][0]
    reset_timer = True
    for step, batch in zip(range(init_step, config.max_steps + 1), pdataset):
        test_dataset.train_it = step
        if reset_timer:
            t_loop_start = time.time()
            reset_timer = False
        lr = learning_rate_fn(step)
        eps = eps_rate_fn(step)
        alpha = alpha_rate_fn(step)
        ts = batch['ts'].squeeze()
        if ts == 0:
            prev = jnp.expand_dims(prevs[ts+1], 0)
        else:
            prev = jnp.expand_dims(prevs[ts-1], 0)

        state, stats, keys, pose = train_pstep(keys, state, batch, lr, eps, alpha, prev)

        # Update estimated pose for next timestep
        prevs[ts, :, :3] = np.array(pose)[0]

        if jax.host_id() == 0:
            stats_trace.append(stats)
        if step % config.gc_every == 0:
            gc.collect()

        # Log training summaries. This is put behind a host_id check because in
        # multi-host evaluation, all hosts need to run inference even though we
        # only use host 0 to record results.
        if jax.host_id() == 0:
            if step % config.print_every == 0:
                summary_writer.scalar('num_params', num_params, step)
                summary_writer.scalar('train_loss', stats.loss[0], step)
                summary_writer.scalar('train_psnr', stats.psnr[0], step)
                for i, l in enumerate(stats.losses[0]):
                    summary_writer.scalar(f'train_losses_{i}', l, step)
                for i, l in enumerate(stats.obj_losses[0]):
                    summary_writer.scalar(f'obj_train_losses_{i}', l, step)
                for i, l in enumerate(stats.d_losses[0]):
                    summary_writer.scalar(f'depth_losses_{i}', l, step)
                for i, l in enumerate(stats.e_losses[0]):
                    summary_writer.scalar(f'empty_losses_{i}', l, step)
                for i, l in enumerate(stats.n_losses[0]):
                    summary_writer.scalar(f'near_losses_{i}', l, step)
                for i, l in enumerate(stats.s_losses[0]):
                    summary_writer.scalar(f'sky_losses_{i}', l, step)
                for i, l in enumerate(stats.distr_losses[0]):
                    summary_writer.scalar(f'distr_reg_{i}', l, step)
                for i, l in enumerate(stats.tv_losses[0]):
                    summary_writer.scalar(f'tv_reg_{i}', l, step)
                for i, l in enumerate(stats.sampling_stats[0]):
                    summary_writer.scalar(f'tvals{i}', l, step)
                for i, l in enumerate(stats.offsets[0]):
                    summary_writer.scalar(f'offsets{i}', l, step)
                for i, l in enumerate(stats.offset_x[0]):
                    summary_writer.scalar(f'offset_x{i}', l, step)
                for i, l in enumerate(stats.offset_y[0]):
                    summary_writer.scalar(f'offset_y{i}', l, step)
                for i, l in enumerate(stats.offset_z[0]):
                    summary_writer.scalar(f'offset_z{i}', l, step)
                for i, l in enumerate(stats.offset_yaw[0]):
                    summary_writer.scalar(f'offset_yaw{i}', l, step)
                for i, p in enumerate(stats.psnrs[0]):
                    summary_writer.scalar(f'train_psnrs_{i}', p, step)
                summary_writer.scalar('weight_l2', stats.weight_l2[0], step)

                plt.bar(stats.samples[0][0, 0, :-1], stats.weights[0][0, 0], color='blue')
                plt.bar(stats.samples[1][0, 0, :-1], stats.weights[1][0, 0], color='green')
                # plt.bar(stats.samples[2][0, 0, :-1], stats.weights[2][0, 0], color='red')
                fig = plt.gcf()
                fig.canvas.draw()
                img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                img = img / 255.0
                summary_writer.image('nerf_weights', img, step)
                plt.clf()

                avg_loss = np.mean(np.concatenate([s.loss for s in stats_trace]))
                avg_psnr = np.mean(np.concatenate([s.psnr for s in stats_trace]))
                avg_obj_psnr = np.mean(np.concatenate([np.nan_to_num(s.obj_psnr, nan=0, posinf=0) for s in stats_trace]))
                max_grad_norm = np.max(
                    np.concatenate([s.grad_norm for s in stats_trace]))
                avg_grad_norm = np.mean(
                    np.concatenate([s.grad_norm for s in stats_trace]))
                max_clipped_grad_norm = np.max(
                    np.concatenate([s.grad_norm_clipped for s in stats_trace]))
                max_grad_max = np.max(
                    np.concatenate([s.grad_abs_max for s in stats_trace]))
                stats_trace = []
                summary_writer.scalar('train_avg_loss', avg_loss, step)
                summary_writer.scalar('train_avg_psnr', avg_psnr, step)
                summary_writer.scalar('train_avg_obj_psnr', avg_obj_psnr, step)
                summary_writer.scalar('train_max_grad_norm', max_grad_norm, step)
                summary_writer.scalar('train_avg_grad_norm', avg_grad_norm, step)
                summary_writer.scalar('train_max_clipped_grad_norm',
                                      max_clipped_grad_norm, step)
                summary_writer.scalar('train_max_grad_max', max_grad_max, step)
                summary_writer.scalar('learning_rate', lr, step)
                summary_writer.scalar('eps_rate', eps, step)
                summary_writer.scalar('alpha_rate', alpha, step)
                steps_per_sec = config.print_every / (time.time() - t_loop_start)
                reset_timer = True
                rays_per_sec = config.batch_size * steps_per_sec
                summary_writer.scalar('train_steps_per_sec', steps_per_sec, step)
                summary_writer.scalar('train_rays_per_sec', rays_per_sec, step)
                precision = int(np.ceil(np.log10(config.max_steps))) + 1
                print(('{:' + '{:d}'.format(precision) + 'd}').format(step) +
                      f'/{config.max_steps:d}: ' + f'i_loss={stats.loss[0]:0.4f}, ' +
                      f'avg_loss={avg_loss:0.4f}, ' +
                      f'weight_l2={stats.weight_l2[0]:0.2e}, ' + f'lr={lr:0.2e}, ' +
                      f'{rays_per_sec:0.0f} rays/sec')
            if step % config.save_every == 0:
                state_to_save = jax.device_get(jax.tree_map(lambda x: x[0], state))
                checkpoints.save_checkpoint(
                    FLAGS.train_dir, state_to_save, int(step), keep=100)

        # Test-set evaluation.
        if FLAGS.render_every > 0 and step % FLAGS.render_every == 0:
            # We reuse the same random number generator from the optimization step
            # here on purpose so that the visualization matches what happened in
            # training.
            t_eval_start = time.time()
            eval_variables = jax.device_get(jax.tree_map(lambda x: x[0],
                                                         state)).optimizer.target

            print(test_dataset.train_it)
            test_case = next(test_dataset)
            print(test_case['rays'].origins.shape)
            sky_mask = (test_case['sky'].squeeze() > 0.0).astype(jnp.float32)
            sky_mask = sky_mask.reshape(test_case['sky'].shape[0], test_case['sky'].shape[1], 1)
            pred_color, pred_distance, pred_acc = obbpose_model.render_image(
                functools.partial(render_eval_pfn, eval_variables),
                test_case['rays'],
                test_case['init'],
                test_case['ext'],
                test_case['ts'],
                keys[0],
                alpha=alpha,
                chunk=FLAGS.chunk)

            vis_suite = vis.visualize_suite(pred_distance, pred_acc)

            # Log eval summaries on host 0.
            if jax.host_id() == 0:
                psnr = math.mse_to_psnr(((pred_color - test_case['pixels']) ** 2).mean())
                ssim = ssim_fn(pred_color, test_case['pixels'])
                eval_time = time.time() - t_eval_start
                num_rays = jnp.prod(jnp.array(test_case['rays'].directions.shape[:-1]))
                rays_per_sec = num_rays / eval_time
                summary_writer.scalar('test_rays_per_sec', rays_per_sec, step)
                print(f'Eval {step}: {eval_time:0.3f}s., {rays_per_sec:0.0f} rays/sec')
                summary_writer.scalar('test_psnr', psnr, step)
                summary_writer.scalar('test_ssim', ssim, step)
                summary_writer.image('test_pred_color', pred_color, step)
                for k, v in vis_suite.items():
                    summary_writer.image('test_pred_' + k, v, step)
                summary_writer.image('test_pred_acc', pred_acc, step)
                summary_writer.image('test_target', test_case['pixels'], step)
                summary_writer.image('depth_target', test_case['depth'], step)

    if config.max_steps % config.save_every != 0:
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        checkpoints.save_checkpoint(
            FLAGS.train_dir, state, int(config.max_steps), keep=100)


if __name__ == '__main__':
    app.run(main)
