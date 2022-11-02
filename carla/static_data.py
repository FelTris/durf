#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

import random
import time

import argparse
import logging

import glob
import os
import sys
import numpy as np
from queue import Queue
from queue import Empty

from natsort import natsorted
from scipy.spatial.transform import Rotation as R


def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K


def get_image_point(loc, K, w2c):
    # Calculate 2D projection of 3D coordinate

    # Format the input coordinate (loc is a carla.Position object)
    point = np.array([loc.x, loc.y, loc.z, 1])
    # transform to camera coordinates
    point_camera = np.dot(w2c, point)

    # New we must change from UE4's coordinate system to an "standard"
    # (x, y ,z) -> (y, -z, x)
    # and we remove the fourth componebonent also
    point_camera = [point_camera[1], -point_camera[2], point_camera[0]]

    # now project 3D->2D using the camera matrix
    point_img = np.dot(K, point_camera)
    # normalize
    point_img[0] /= point_img[2]
    point_img[1] /= point_img[2]

    return point_img[0:2]


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def carla2Nerf(mat):
    #mat = np.array(transform.get_matrix())

    rotz = np.array([[0.0000000, -1.0000000, 0.0000000, 0.0],
                     [1.0000000, 0.0000000, 0.0000000, 0.0],
                     [0.0000000, 0.0000000, 1.0000000, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])  # 90deg around z-axis
    roty = np.array([[0.0000000, 0.0000000, 1.0000000, 0.0],
                     [0.0000000, 1.0000000, 0.0000000, 0.0],
                     [-1.0000000, 0.0000000, 0.0000000, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])  # 90deg around y-axis

    trafo1 = np.array([[0.0000000, 1.0000000, 0.0000000, 0.0],
                       [0.0000000, 0.0000000, 1.0000000, 0.0],
                       [-1.0000000, 0.0000000, 0.0000000, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    trafo2 = np.array([[0.0000000, 0.0000000, -1.0000000, 0.0],
                       [1.0000000, 0.0000000, 0.0000000, 0.0],
                       [0.0000000, 1.0000000, 0.0000000, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    carla2opengl = np.matmul(roty, rotz)
    # pose = np.matmul(mat, carla2opengl)
    # pose[0, 3] = -pose[0, 3]
    pose = np.matmul(trafo1, mat)
    pose = np.matmul(pose, trafo2)

    return pose


def nerf2carla(mat):
    rotz = np.array([[0.0000000, -1.0000000, 0.0000000, 0.0],
                     [1.0000000, 0.0000000, 0.0000000, 0.0],
                     [0.0000000, 0.0000000, 1.0000000, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])  # 90deg around z-axis
    roty = np.array([[0.0000000, 0.0000000, 1.0000000, 0.0],
                     [0.0000000, 1.0000000, 0.0000000, 0.0],
                     [-1.0000000, 0.0000000, 0.0000000, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])  # 90deg around y-axis

    trafo1 = np.array([[0.0000000, 1.0000000, 0.0000000, 0.0],
                       [0.0000000, 0.0000000, 1.0000000, 0.0],
                       [-1.0000000, 0.0000000, 0.0000000, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    trafo2 = np.array([[0.0000000, 0.0000000, -1.0000000, 0.0],
                       [1.0000000, 0.0000000, 0.0000000, 0.0],
                       [0.0000000, 1.0000000, 0.0000000, 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    carla2opengl = np.matmul(roty, rotz)
    # pose = np.matmul(mat, carla2opengl)
    # pose[0, 3] = -pose[0, 3]
    pose = np.matmul(mat, np.linalg.inv(trafo2))
    pose = np.matmul(np.linalg.inv(trafo1), mat)

    return pose



def sensor_callback(sensor_data, sensor_queue, sensor_name, sensor_type, pose_dict):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    if 'RGB' in sensor_name:
        pose_dict[str(sensor_data.frame) + '_' + sensor_name] = carla2Nerf(np.array(sensor_data.transform.get_matrix()))

    sensor_data.convert(sensor_type)
    sensor_data.save_to_disk('/home/tristram/data/carla/testbox/%08d' % sensor_data.frame + '_' + sensor_name)

    sensor_queue.put((sensor_data.frame, sensor_name))


def spawn_cam(parent, sensor, resolution, position, rotation):
    bound_x = 0.5 + parent.bounding_box.extent.x
    bound_y = 0.5 + parent.bounding_box.extent.y
    bound_z = 0.5 + parent.bounding_box.extent.z
    Attachment = carla.AttachmentType

    camera_transforms = (carla.Transform(carla.Location(x=position[0] * bound_x,
                                                        y=position[1] * bound_y,
                                                        z=position[2] * bound_z),
                                         carla.Rotation(pitch=rotation[0],
                                                        yaw=rotation[1],
                                                        roll=rotation[2])),
                         Attachment.Rigid)

    sensor = sensor

    world = parent.get_world()
    bp_library = world.get_blueprint_library()
    bp = bp_library.find(sensor[0])
    bp.set_attribute('image_size_x', resolution[0])
    bp.set_attribute('image_size_y', resolution[1])
    bp.set_attribute('fov', '50')
    bp.set_attribute('sensor_tick', '0.1')
    if bp.has_attribute('motion_blur_intensity'):
        bp.set_attribute('motion_blur_intensity', '0.0')

    cam = world.spawn_actor(
        bp, camera_transforms[0],
        attach_to=parent,
        attachment_type=camera_transforms[1]
    )

    return cam


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-n', '--number-of-vehicles',
        metavar='N',
        default=30,
        type=int,
        help='Number of vehicles (default: 30)')
    argparser.add_argument(
        '-w', '--number-of-walkers',
        metavar='W',
        default=10,
        type=int,
        help='Number of walkers (default: 10)')
    argparser.add_argument(
        '--safe',
        action='store_true',
        help='Avoid spawning vehicles prone to accidents')
    argparser.add_argument(
        '--filterv',
        metavar='PATTERN',
        default='vehicle.*',
        help='Filter vehicle model (default: "vehicle.*")')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '--filterw',
        metavar='PATTERN',
        default='walker.pedestrian.*',
        help='Filter pedestrian type (default: "walker.pedestrian.*")')
    argparser.add_argument(
        '--generationw',
        metavar='G',
        default='2',
        help='restrict to certain pedestrian generation (values: "1","2","All" - default: "2")')
    argparser.add_argument(
        '--tm-port',
        metavar='P',
        default=8000,
        type=int,
        help='Port to communicate with TM (default: 8000)')
    argparser.add_argument(
        '--asynch',
        action='store_true',
        help='Activate asynchronous mode execution')
    argparser.add_argument(
        '--hybrid',
        action='store_true',
        help='Activate hybrid mode for Traffic Manager')
    argparser.add_argument(
        '-s', '--seed',
        metavar='S',
        type=int,
        help='Set random device seed and deterministic mode for Traffic Manager')
    argparser.add_argument(
        '--seedw',
        metavar='S',
        default=0,
        type=int,
        help='Set the seed for pedestrians module')
    argparser.add_argument(
        '--car-lights-on',
        action='store_true',
        default=False,
        help='Enable automatic car light management')
    argparser.add_argument(
        '--hero',
        action='store_true',
        default=False,
        help='Set one of the vehicles as hero')
    argparser.add_argument(
        '--respawn',
        action='store_true',
        default=False,
        help='Automatically respawn dormant vehicles (only in large maps)')
    argparser.add_argument(
        '--no-rendering',
        action='store_true',
        default=False,
        help='Activate no rendering mode')

    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    actor_list = []

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()

        original_settings = world.get_settings()
        settings = world.get_settings()

        if args.no_rendering:
            settings.no_rendering_mode = True

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        synchronous_master = True
        world.apply_settings(settings)

        # ====================================================================
        # ------ Ego Vehicle -------------------------------------------------
        # ====================================================================

        # The world contains the list blueprints that we can use for adding new
        # actors into the simulation.
        blueprint_library = world.get_blueprint_library()

        # Now let's filter all the blueprints of type 'vehicle' and choose one
        # at random.
        bp = random.choice(blueprint_library.filter('vehicle.bmw.*'))

        # A blueprint contains the list of attributes that define a vehicle's
        # instance, we can read them and modify some of them. For instance,
        # let's randomize its color.
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)

        # Now we need to give an initial transform to the vehicle. We choose a
        # random transform from the list of recommended spawn points of the map.
        transform = random.choice(world.get_map().get_spawn_points())
        f = transform.get_forward_vector()
        while (f.x - 1.0)**2 > 1e-5:
            transform = random.choice(world.get_map().get_spawn_points())
            f = transform.get_forward_vector()
        print(f)

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, transform)

        world.tick()
        fwd = vehicle.get_transform().get_forward_vector()
        fwd = np.array([fwd.x, fwd.y, fwd.z])
        faxis = np.argmax(np.abs(fwd))
        print(fwd)

        dyn_trans = vehicle.get_transform()
        mvmt = np.array([1.54400009e+00, 0., 2.11563608e+00-0.66])
        mvmt = np.array([0, 0., 0.])
        if fwd[faxis] < 0:
            mvmt[faxis] -= 20.0
        else:
            mvmt[faxis] += 20.0
        dyn_trans.location += carla.Location(x=mvmt[0], y=mvmt[1], z=mvmt[2])
        dyn_trans.location = carla.Location(x=100.0, y=100.0, z=100.0)
        print(dyn_trans)

        dyn_vehicle = world.spawn_actor(bp, dyn_trans)
        dyn_vehicle.set_enable_gravity(False)
        actor_list.append((dyn_vehicle))
        for i in range(10):
            world.tick()

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        resolutions = [['480', '320'], ['480', '221']]
        res = [[480, 320], [480, 221]]
        position = [[1.54400009e+00, -2.39734336e-02, 2.11563608e+00],
                    [1.49385787, -0.09704899, 2.11561398],
                    [1.42868991, -0.1157639, 2.1156166],
                    [1.49722036, 0.09480753, 2.11569233],
                    [1.43274822e+00, 1.15794517e-01, 2.11571117e+00]]
        rotation = [[0, 0, 0],
                    [0.1940637, -44.4198126, -0.5363985],
                    [0.7835298, -89.8953231, 0.3499943],
                    [-0.1690629, 44.7857132, 0.3824879],
                    [1.1168912, 90.116268, -0.8659727]]

        types = [['sensor.camera.rgb', cc.Raw, 'RGB'],
                 ['sensor.camera.depth', cc.Raw, 'Depth'],
                 ['sensor.camera.semantic_segmentation', cc.Raw, 'Semantic'],
                 ['sensor.camera.instance_segmentation', cc.Raw, 'Instance']]
        ks = []
        for i in range(5):
            if i == 2 or i == 4:
                ks.append(build_projection_matrix(res[1][0], res[1][1], 50.0))
            else:
                ks.append(build_projection_matrix(res[0][0], res[0][1], 50.0))

        sensor_queue = Queue()
        image_queue = Queue()
        sensor_list = []
        pose_dict = {}

        boxcam = spawn_cam(dyn_vehicle, types[0], resolutions[0], position[0], rotation[0])
        boxcam.listen(image_queue.put)
        rgbcam_front = spawn_cam(dyn_vehicle, types[0], resolutions[0], position[0], rotation[0])
        rgbcam_front.listen(lambda data: sensor_callback(data, sensor_queue, "1_RGB", types[0][1], pose_dict))
        sensor_list.append(rgbcam_front)
        depthcam_front = spawn_cam(dyn_vehicle, types[1], resolutions[0], position[0], rotation[0])
        depthcam_front.listen(lambda data: sensor_callback(data, sensor_queue, "1_DEPTH", types[1][1], pose_dict))
        sensor_list.append(depthcam_front)
        semcam_front = spawn_cam(dyn_vehicle, types[2], resolutions[0], position[0], rotation[0])
        semcam_front.listen(lambda data: sensor_callback(data, sensor_queue, "1_SEM", types[2][1], pose_dict))
        sensor_list.append(semcam_front)
        inscam_front = spawn_cam(dyn_vehicle, types[3], resolutions[0], position[0], rotation[0])
        inscam_front.listen(lambda data: sensor_callback(data, sensor_queue, "1_INS", types[3][1], pose_dict))
        sensor_list.append(inscam_front)

        yaw = np.arange(-60., 60.)
        pitch = np.arange(-60., 60.)
        roll = np.arange(-60., 60.)
        position = [[0.5, 0., 0.],
                    [0., 0.5, 0.],
                    [2.5, 1.5, 0.],
                    [-3.5, -1.5, 0.]]

        edges = [[0, 1], [1, 3], [3, 2], [2, 0], [0, 4], [4, 5], [5, 1], [5, 7], [7, 6], [6, 4], [6, 2], [7, 3]]

        old = dyn_vehicle.get_transform()
        old_loc = old.location
        old_rot = old.rotation

        def sample_spherical(npoints, ndim=3):
            np.random.seed(0)
            vec = np.random.randn(ndim, npoints)
            vec /= np.linalg.norm(vec, axis=0)
            vec *= 10.0
            return vec

        def look_at(camera_position, target_position):
            """Returns model-view matrix from camera position to target.

            # Arguments
                camera_position: Numpy-array of length 3. Camera position.
                target_position: Numpy-array of length 3. Target position.
            """
            camera_direction = camera_position - target_position
            camera_direction = camera_direction / np.linalg.norm(camera_direction)
            camera_right = np.cross(np.array([0.0, 1.0, 0.0]), camera_direction)
            camera_right = camera_right / np.linalg.norm(camera_right)
            camera_up = np.cross(camera_direction, camera_right)
            camera_up = camera_up / np.linalg.norm(camera_up)
            rotation_transform = np.zeros((4, 4))
            rotation_transform[0, :3] = camera_right
            rotation_transform[1, :3] = camera_up
            rotation_transform[2, :3] = camera_direction
            rotation_transform[-1, -1] = 1
            translation_transform = np.eye(4)
            translation_transform[:3, -1] = - camera_position
            look_at_transform = np.matmul(rotation_transform, translation_transform)
            return look_at_transform

        def my_lookat(pos, target):
            """
            returns world2cam matrix looking at target in x.forward, y.right, z.up coordinates
            """
            forward = (target - pos) / np.linalg.norm(target - pos)
            right = np.cross([0.0, 0.0, 1.0], forward)
            up = np.cross(forward, right)
            rotation_transform = np.zeros((4, 4))
            rotation_transform[:3, 0] = forward
            rotation_transform[:3, 1] = right
            rotation_transform[:3, 2] = up
            rotation_transform[-1, -1] = 1
            rotation_transform[:3, 3] = pos
            #trans = np.linalg.inv(rotation_transform)
            #trans[:3, 3] = pos
            return rotation_transform

        vec = sample_spherical(200)
        print(vec[:, 0])
        #vec += 100
        vec = vec.transpose(1, 0)
        print(vec[0, :])

        trans = []
        j = 0
        for v in vec:
            if j < 100:
                trans.append(my_lookat(v, [0, 0, 0]))
                j += 1
        trans = np.array(trans)

        time_dict = {}
        # Main loop
        debug = []
        count = 0
        dyn_loc = dyn_vehicle.get_transform().location
        while True:
            t = trans[count]
            loc = t[:3, 3]
            rot = R.from_matrix(t[:3, :3])
            rot = rot.as_euler('xyz') * (180/np.pi)
            carla_t = carla.Transform(carla.Location(x=loc[0], y=loc[1], z=loc[2]),
                                      carla.Rotation(roll=rot[0], pitch=-rot[1], yaw=rot[2]))

            debug.append(np.array(carla_t.get_matrix()))

            raxis = np.random.randint(0, 3)
            radd = np.random.randint(0, 4)
            add = np.array([0.0, 0.0, 0.0])
            add[raxis] += radd
            new_loc = dyn_loc + carla.Location(x=add[0], y=add[1], z=add[2])
            dyn_vehicle.set_location(new_loc)
            print(dyn_vehicle.get_transform().location)

            for s in sensor_list:
                s.set_transform(carla_t)
            boxcam.set_transform(carla_t)
            # Tick the server
            world.tick()
            w_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % w_frame)
            print(len(pose_dict))

            # Now, we wait to the sensors data to be received.
            # As the queue is blocking, we will wait in the queue.get() methods
            # until all the information is processed and we continue with the next frame.
            # We include a timeout of 1.0 s (in the get method) and if some information is
            # not received in this time we continue.
            try:

                for _ in range(len(sensor_list)):
                    s_frame = sensor_queue.get(True, 1.0)
                    print("    Frame: %d   Sensor: %s" % (s_frame[0], s_frame[1]))

                count += 1
                w2cs = []
                for i in range(1):
                    w2cs.append(np.array(sensor_list[4 * i].get_transform().get_inverse_matrix()))

                image = image_queue.get()
                img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

                veh_dict = {}
                for npc in world.get_actors().filter('*vehicle*'):
                    # Filter out the ego vehicle
                    if npc.id != vehicle.id:
                        bb = npc.bounding_box

                        cam_dict = {}
                        visible = False
                        for s in range(1):
                            camera = sensor_list[4 * s]
                            if s == 2 or s == 4:
                                image_w = res[1][0]
                                image_h = res[1][1]
                            else:
                                image_w = res[0][0]
                                image_h = res[0][1]
                            # Calculate the dot product between the forward vector
                            # of the vehicle and the vector between the vehicle
                            # and the other vehicle. We threshold this dot product
                            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                            forward_vec = camera.get_transform().get_forward_vector()
                            ray = npc.get_transform().location - camera.get_transform().location

                            if forward_vec.dot(ray) > 1:
                                box_dict = {}

                                bbox_center = np.array(carla.Transform(bb.location, bb.rotation).get_matrix())
                                npc2w = np.array(npc.get_transform().get_matrix())
                                bbox_center2w = np.dot(npc2w, bbox_center)

                                verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                                x_max = -10000
                                x_min = 10000
                                y_max = -10000
                                y_min = 10000

                                for vert in verts:
                                    p = get_image_point(vert, ks[s], w2cs[s])
                                    # Find the rightmost vertex
                                    if p[0] > x_max:
                                        x_max = p[0]
                                    # Find the leftmost vertex
                                    if p[0] < x_min:
                                        x_min = p[0]
                                    # Find the highest vertex
                                    if p[1] > y_max:
                                        y_max = p[1]
                                    # Find the lowest  vertex
                                    if p[1] < y_min:
                                        y_min = p[1]
                                if x_min < 0 and not x_max < 0 and x_max < image_w:
                                    x_min = 0
                                if x_max > image_w and not x_min > image_w and x_min >= 0:
                                    x_max = image_w
                                if y_min < 0 and not y_max < 0 and y_max < image_h:
                                    y_min = 0
                                if y_max > image_h and not y_min > image_h and y_min >= 0:
                                    y_max = image_h
                                if x_min >= 0 and x_max <= image_w and y_min >= 0 and y_max <= image_h:
                                    visible = True
                                    box_dict['x_min'] = int(x_min)
                                    box_dict['x_max'] = int(x_max)
                                    box_dict['y_min'] = int(y_min)
                                    box_dict['y_max'] = int(y_max)
                                    cam_dict[str(s+1)] = box_dict

                        if visible:
                            cam_dict['center'] = carla2Nerf(bbox_center2w)
                            cam_dict['extent'] = np.array([bb.extent.y, bb.extent.z, bb.extent.x])
                            veh_dict[str(npc.id)] = cam_dict
                time_dict[str(w_frame)] = veh_dict

                ind = np.random.randint(0, len(yaw))
                pind = np.random.randint(0, len(pitch))
                rind = np.random.randint(0, len(roll))
                new_yaw = yaw[ind] + old_rot.yaw
                new_pitch = pitch[pind] + old_rot.pitch
                new_roll = roll[rind] + old_rot.roll
                new_rot = carla.Rotation(pitch=new_pitch, yaw=new_yaw, roll=new_roll)
                #dyn_vehicle.set_transform(carla.Transform(old_loc, new_rot))
                """
                for edge in edges:
                    p1 = get_image_point(verts[edge[0]], ks[s], w2cs[s])
                    p2 = get_image_point(verts[edge[1]], ks[s], w2cs[s])
                    cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                             (255, 0, 0, 255), 1)
                
                cv2.imshow('ImageWindowName', img)
                cv2.waitKey(10)
                if cv2.waitKey(1) == ord('q'):
                    break"""

            except Empty:
                print("    Some of the sensor information is missed")
            #world.tick()

    finally:
        #cv2.destroyAllWindows()
        np.save('/home/tristram/data/carla/testbox/' + 'bboxes.npy', time_dict)
        debug = np.array(debug)
        np.save('/home/tristram/data/carla/testbox/debug.npy', debug)
        np.save('/home/tristram/data/carla/testbox/trans.npy', trans)
        poses = []
        keys = natsorted(list(pose_dict.keys()))
        for key in keys:
            print(key)
            poses.append(pose_dict[key])
        poses = np.array(poses)
        print('saving poses.')
        print(poses.shape)
        np.save('/home/tristram/data/carla/testbox/' + 'poses.npy', poses)
        """
        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])"""
        boxcam.destroy()
        print('returning original settings')
        world.apply_settings(original_settings)
        print('destroying actors')
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('destroying sensors')
        for sensor in sensor_list:
            sensor.destroy()
        print('done.')


if __name__ == '__main__':
    main()
