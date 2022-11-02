#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

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


def sensor_callback(sensor_data, sensor_queue, sensor_name, sensor_type, pose_dict):
    # Do stuff with the sensor_data data like save it to disk
    # Then you just need to add to the queue
    if 'RGB' in sensor_name:
        pose_dict[str(sensor_data.frame) + '_' + sensor_name] = carla2Nerf(np.array(sensor_data.transform.get_matrix()))

    sensor_data.convert(sensor_type)
    sensor_data.save_to_disk('/home/tristram/data/carla/_out/%08d' % sensor_data.frame + '_' + sensor_name)

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
    bp.set_attribute('sensor_tick', '0.2')
    if bp.has_attribute('motion_blur_intensity'):
        bp.set_attribute('motion_blur_intensity', '0.0')

    cam = parent.get_world().spawn_actor(
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

    vehicles_list = []
    walkers_list = []
    all_id = []
    actor_list = []

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    random.seed(args.seed if args.seed is not None else int(time.time()))

    try:
        world = client.get_world()

        traffic_manager = client.get_trafficmanager(args.tm_port)
        traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        if args.respawn:
            traffic_manager.set_respawn_dormant_vehicles(True)
        if args.hybrid:
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(70.0)
        if args.seed is not None:
            traffic_manager.set_random_device_seed(args.seed)

        original_settings = world.get_settings()
        settings = world.get_settings()

        if args.no_rendering:
            settings.no_rendering_mode = True

        # We set CARLA syncronous mode
        settings.fixed_delta_seconds = 0.1
        settings.synchronous_mode = True
        synchronous_master = True
        world.apply_settings(settings)

        # ==============================================================
        # ------- Traffic Generation -----------------------------------
        # ==============================================================

        blueprints = get_actor_blueprints(world, args.filterv, args.generationv)
        blueprintsWalkers = get_actor_blueprints(world, args.filterw, args.generationw)

        if args.safe:
            blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
            blueprints = [x for x in blueprints if not x.id.endswith('microlino')]
            blueprints = [x for x in blueprints if not x.id.endswith('carlacola')]
            blueprints = [x for x in blueprints if not x.id.endswith('cybertruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('t2')]
            blueprints = [x for x in blueprints if not x.id.endswith('sprinter')]
            blueprints = [x for x in blueprints if not x.id.endswith('firetruck')]
            blueprints = [x for x in blueprints if not x.id.endswith('ambulance')]

        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.number_of_vehicles < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.number_of_vehicles > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.number_of_vehicles = number_of_spawn_points

        # @todo cannot import these directly.
        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        # --------------
        # Spawn vehicles
        # --------------
        batch = []
        hero = args.hero
        for n, transform in enumerate(spawn_points):
            if n >= args.number_of_vehicles:
                break
            blueprint = random.choice(blueprints)
            if blueprint.has_attribute('color'):
                color = random.choice(blueprint.get_attribute('color').recommended_values)
                blueprint.set_attribute('color', color)
            if blueprint.has_attribute('driver_id'):
                driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
                blueprint.set_attribute('driver_id', driver_id)
            if hero:
                blueprint.set_attribute('role_name', 'hero')
                hero = False
            else:
                blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

        for response in client.apply_batch_sync(batch, synchronous_master):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # Set automatic vehicle lights update if specified
        if args.car_lights_on:
            all_vehicle_actors = world.get_actors(vehicles_list)
            for actor in all_vehicle_actors:
                traffic_manager.update_vehicle_lights(actor, True)

        # -------------
        # Spawn Walkers
        # -------------
        # some settings
        """
        percentagePedestriansRunning = 0.0  # how many pedestrians will run
        percentagePedestriansCrossing = 0.0  # how many pedestrians will walk through the road
        if args.seedw:
            world.set_pedestrians_seed(args.seedw)
            random.seed(args.seedw)
        # 1. take all the random locations to spawn
        spawn_points = []
        for i in range(args.number_of_walkers):
            spawn_point = carla.Transform()
            loc = world.get_random_location_from_navigation()
            if (loc != None):
                spawn_point.location = loc
                spawn_points.append(spawn_point)
        # 2. we spawn the walker object
        batch = []
        walker_speed = []
        for spawn_point in spawn_points:
            walker_bp = random.choice(blueprintsWalkers)
            # set as not invincible
            if walker_bp.has_attribute('is_invincible'):
                walker_bp.set_attribute('is_invincible', 'false')
            # set the max speed
            if walker_bp.has_attribute('speed'):
                if (random.random() > percentagePedestriansRunning):
                    # walking
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
                else:
                    # running
                    walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
            else:
                print("Walker has no speed")
                walker_speed.append(0.0)
            batch.append(SpawnActor(walker_bp, spawn_point))
        results = client.apply_batch_sync(batch, True)
        walker_speed2 = []
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list.append({"id": results[i].actor_id})
                walker_speed2.append(walker_speed[i])
        walker_speed = walker_speed2
        # 3. we spawn the walker controller
        batch = []
        walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
        for i in range(len(walkers_list)):
            batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
        results = client.apply_batch_sync(batch, True)
        for i in range(len(results)):
            if results[i].error:
                logging.error(results[i].error)
            else:
                walkers_list[i]["con"] = results[i].actor_id
        # 4. we put together the walkers and controllers id to get the objects from their id
        for i in range(len(walkers_list)):
            all_id.append(walkers_list[i]["con"])
            all_id.append(walkers_list[i]["id"])
        all_actors = world.get_actors(all_id)

        world.tick()

        # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
        # set how many pedestrians can cross the road
        world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
        for i in range(0, len(all_id), 2):
            # start walker
            all_actors[i].start()
            # set walk to random point
            all_actors[i].go_to_location(world.get_random_location_from_navigation())
            # max speed
            all_actors[i].set_max_speed(float(walker_speed[int(i / 2)]))
        """
        print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))

        # Example of how to use Traffic Manager parameters
        traffic_manager.global_percentage_speed_difference(30.0)

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

        # So let's tell the world to spawn the vehicle.
        vehicle = world.spawn_actor(bp, transform)

        # It is important to note that the actors we create won't be destroyed
        # unless we call their "destroy" function. If we fail to call "destroy"
        # they will stay in the simulation even after we quit the Python script.
        # For that reason, we are storing all the actors we create so we can
        # destroy them afterwards.
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True)
        for i in range(10):
            world.tick()

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
        sensor_list = []
        pose_dict = {}

        rgbcam_front = spawn_cam(vehicle, types[0], resolutions[0], position[0], rotation[0])
        rgbcam_front.listen(lambda data: sensor_callback(data, sensor_queue, "1_RGB", types[0][1], pose_dict))
        sensor_list.append(rgbcam_front)
        depthcam_front = spawn_cam(vehicle, types[1], resolutions[0], position[0], rotation[0])
        depthcam_front.listen(lambda data: sensor_callback(data, sensor_queue, "1_DEPTH", types[1][1], pose_dict))
        sensor_list.append(depthcam_front)
        semcam_front = spawn_cam(vehicle, types[2], resolutions[0], position[0], rotation[0])
        semcam_front.listen(lambda data: sensor_callback(data, sensor_queue, "1_SEM", types[2][1], pose_dict))
        sensor_list.append(semcam_front)
        inscam_front = spawn_cam(vehicle, types[3], resolutions[0], position[0], rotation[0])
        inscam_front.listen(lambda data: sensor_callback(data, sensor_queue, "1_INS", types[3][1], pose_dict))
        sensor_list.append(inscam_front)

        rgbcam_frontleft = spawn_cam(vehicle, types[0], resolutions[0], position[1], rotation[1])
        rgbcam_frontleft.listen(lambda data: sensor_callback(data, sensor_queue, "2_RGB", types[0][1], pose_dict))
        sensor_list.append(rgbcam_frontleft)
        depthcam_frontleft = spawn_cam(vehicle, types[1], resolutions[0], position[1], rotation[1])
        depthcam_frontleft.listen(lambda data: sensor_callback(data, sensor_queue, "2_DEPTH", types[1][1], pose_dict))
        sensor_list.append(depthcam_frontleft)
        semcam_frontleft = spawn_cam(vehicle, types[2], resolutions[0], position[1], rotation[1])
        semcam_frontleft.listen(lambda data: sensor_callback(data, sensor_queue, "2_SEM", types[2][1], pose_dict))
        sensor_list.append(semcam_frontleft)
        inscam_frontleft = spawn_cam(vehicle, types[3], resolutions[0], position[1], rotation[1])
        inscam_frontleft.listen(lambda data: sensor_callback(data, sensor_queue, "2_INS", types[3][1], pose_dict))
        sensor_list.append(inscam_frontleft)

        rgbcam_sideleft = spawn_cam(vehicle, types[0], resolutions[1], position[2], rotation[2])
        rgbcam_sideleft.listen(lambda data: sensor_callback(data, sensor_queue, "3_RGB", types[0][1], pose_dict))
        sensor_list.append(rgbcam_sideleft)
        depthcam_sideleft = spawn_cam(vehicle, types[1], resolutions[1], position[2], rotation[2])
        depthcam_sideleft.listen(lambda data: sensor_callback(data, sensor_queue, "3_DEPTH", types[1][1], pose_dict))
        sensor_list.append(depthcam_sideleft)
        semcam_sideleft = spawn_cam(vehicle, types[2], resolutions[1], position[2], rotation[2])
        semcam_sideleft.listen(lambda data: sensor_callback(data, sensor_queue, "3_SEM", types[2][1], pose_dict))
        sensor_list.append(semcam_sideleft)
        inscam_sideleft = spawn_cam(vehicle, types[3], resolutions[1], position[2], rotation[2])
        inscam_sideleft.listen(lambda data: sensor_callback(data, sensor_queue, "3_INS", types[3][1], pose_dict))
        sensor_list.append(inscam_sideleft)

        rgbcam_frontright = spawn_cam(vehicle, types[0], resolutions[0], position[3], rotation[3])
        rgbcam_frontright.listen(lambda data: sensor_callback(data, sensor_queue, "4_RGB", types[0][1], pose_dict))
        sensor_list.append(rgbcam_frontright)
        depthcam_frontright = spawn_cam(vehicle, types[1], resolutions[0], position[3], rotation[3])
        depthcam_frontright.listen(lambda data: sensor_callback(data, sensor_queue, "4_DEPTH", types[1][1], pose_dict))
        sensor_list.append(depthcam_frontright)
        semcam_frontright = spawn_cam(vehicle, types[2], resolutions[0], position[3], rotation[3])
        semcam_frontright.listen(lambda data: sensor_callback(data, sensor_queue, "4_SEM", types[2][1], pose_dict))
        sensor_list.append(semcam_frontright)
        inscam_frontright = spawn_cam(vehicle, types[3], resolutions[0], position[3], rotation[3])
        inscam_frontright.listen(lambda data: sensor_callback(data, sensor_queue, "4_INS", types[3][1], pose_dict))
        sensor_list.append(inscam_frontright)

        rgbcam_sideright = spawn_cam(vehicle, types[0], resolutions[1], position[4], rotation[4])
        rgbcam_sideright.listen(lambda data: sensor_callback(data, sensor_queue, "5_RGB", types[0][1], pose_dict))
        sensor_list.append(rgbcam_sideright)
        depthcam_sideright = spawn_cam(vehicle, types[1], resolutions[1], position[4], rotation[4])
        depthcam_sideright.listen(lambda data: sensor_callback(data, sensor_queue, "5_DEPTH", types[1][1], pose_dict))
        sensor_list.append(depthcam_sideright)
        semcam_sideright = spawn_cam(vehicle, types[2], resolutions[1], position[4], rotation[4])
        semcam_sideright.listen(lambda data: sensor_callback(data, sensor_queue, "5_SEM", types[2][1], pose_dict))
        sensor_list.append(semcam_sideright)
        inscam_sideright = spawn_cam(vehicle, types[3], resolutions[1], position[4], rotation[4])
        inscam_sideright.listen(lambda data: sensor_callback(data, sensor_queue, "5_INS", types[3][1], pose_dict))
        sensor_list.append(inscam_sideright)


        time_dict = {}
        # Main loop
        while True:
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

                w2cs = []
                for i in range(5):
                    w2cs.append(np.array(sensor_list[4 * i].get_transform().get_inverse_matrix()))

                veh_dict = {}
                for npc in world.get_actors().filter('*vehicle*'):
                    # Filter out the ego vehicle
                    if npc.id != vehicle.id:
                        bb = npc.bounding_box
                        speed = np.sum(
                            [np.abs(npc.get_velocity().x), np.abs(npc.get_velocity().y), np.abs(npc.get_velocity().z)])
                        dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                        if speed > 1.0:
                            # Filter for the vehicles within 75m
                            if dist < 75:
                                cam_dict = {}
                                visible = False
                                for s in range(int(len(sensor_list) / 5)):
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
                #print(time_dict[str(w_frame)])

            except Empty:
                print("    Some of the sensor information is missed")

    finally:
        np.save('/home/tristram/data/carla/_out/' + 'bboxes.npy', time_dict)
        poses = []
        keys = natsorted(list(pose_dict.keys()))
        for key in keys:
            print(key)
            poses.append(pose_dict[key])
        poses = np.array(poses)
        print('saving poses.')
        print(poses.shape)
        np.save('/home/tristram/data/carla/_out/' + 'poses.npy', poses)
        """
        print('\ndestroying %d vehicles' % len(vehicles_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])

        # stop walker controllers (list is [controller, actor, controller, actor ...])
        for i in range(0, len(all_id), 2):
            all_actors[i].stop()

        print('\ndestroying %d walkers' % len(walkers_list))
        client.apply_batch([carla.command.DestroyActor(x) for x in all_id])"""

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
