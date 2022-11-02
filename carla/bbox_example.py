import carla
import math
import random
import time
import queue
import numpy as np
import cv2

def build_projection_matrix(w, h, fov):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    return K

def carla2Nerf(transform):

    #mat = np.array(transform.get_matrix())
    mat = transform

    rotz = np.array([[0.0000000,  -1.0000000,  0.0000000, 0.0],
                     [1.0000000,  0.0000000,  0.0000000, 0.0],
                     [0.0000000, 0.0000000,  1.0000000, 0.0],
                     [0.0, 0.0, 0.0, 1.0]])  # 90deg around z-axis
    roty = np.array([[0.0000000,  0.0000000,  1.0000000, 0.0],
                     [0.0000000,  1.0000000,  0.0000000, 0.0],
                     [-1.0000000,  0.0000000,  0.0000000, 0.0],
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
    #pose = np.matmul(mat, carla2opengl)
    #pose[0, 3] = -pose[0, 3]
    pose = np.matmul(trafo1, mat)
    pose = np.matmul(pose, trafo2)

    return pose


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

def main():
    client = carla.Client('localhost', 2000)
    world  = client.get_world()
    bp_lib = world.get_blueprint_library()

    spawn_points = world.get_map().get_spawn_points()

    # spawn vehicle
    vehicle_bp =bp_lib.find('vehicle.lincoln.mkz_2020')
    vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))

    # spawn camera
    camera_bp = bp_lib.find('sensor.camera.rgb')
    camera_init_trans = carla.Transform(carla.Location(z=2), carla.Rotation(yaw=45))
    camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=vehicle)
    vehicle.set_autopilot(True)

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True # Enables synchronous mode
    settings.fixed_delta_seconds = 0.025
    world.apply_settings(settings)

    # Create a queue to store and retrieve the sensor data
    image_queue = queue.Queue()
    camera.listen(image_queue.put)

    for i in range(50):
        vehicle_bp = random.choice(bp_lib.filter('vehicle'))
        npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if npc:
            npc.set_autopilot(True)

    # Get the world to camera matrix
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())

    # Get the attributes from the camera
    image_w = camera_bp.get_attribute("image_size_x").as_int()
    image_h = camera_bp.get_attribute("image_size_y").as_int()
    fov = camera_bp.get_attribute("fov").as_float()

    # Calculate the camera projection matrix to project from 3D -> 2D
    K = build_projection_matrix(image_w, image_h, fov)
    """
    # Set up the set of bounding boxes from the level
    # We filter for traffic lights and traffic signs
    bounding_box_set = world.get_level_bbs(carla.CityObjectLabel.TrafficLight)
    bounding_box_set.extend(world.get_level_bbs(carla.CityObjectLabel.TrafficSigns))
    """
    # Remember the edge pairs
    edges = [[0,1], [1,3], [3,2], [2,0], [0,4], [4,5], [5,1], [5,7], [7,6], [6,4], [6,2], [7,3]]

    # Retrieve the first image
    world.tick()
    image = image_queue.get()

    # Reshape the raw data into an RGB array
    img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    # Display the image in an OpenCV display window
    cv2.namedWindow('ImageWindowName', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('ImageWindowName', img)
    cv2.waitKey(1)
    time_dict = {}
    while True:
        # Retrieve and reshape the image
        world.tick()
        image = image_queue.get()
        w_frame = world.get_snapshot().frame

        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

        # Get the camera matrix
        world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
        veh_dict = {}

        for npc in world.get_actors().filter('*vehicle*'):

            # Filter out the ego vehicle
            if npc.id != vehicle.id:

                bb = npc.bounding_box
                speed = np.sum([np.abs(npc.get_velocity().x),np.abs(npc.get_velocity().y), np.abs(npc.get_velocity().z)])
                dist = npc.get_transform().location.distance(vehicle.get_transform().location)

                if speed > 1.0:

                    # Filter for the vehicles within 75m
                    if dist < 75:

                        # Calculate the dot product between the forward vector
                        # of the vehicle and the vector between the vehicle
                        # and the other vehicle. We threshold this dot product
                        # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
                        forward_vec = camera.get_transform().get_forward_vector()
                        right_vec = vehicle.get_transform().get_right_vector()
                        left_vec = -1.0 * right_vec
                        #print(npc.get_transform().get_matrix())
                        ray = npc.get_transform().location - vehicle.get_transform().location

                        #if left_vec.dot(ray) > 1 or forward_vec.dot(ray) > 1 or right_vec.dot(ray) > 1:
                        if forward_vec.dot(ray) > 1:
                            box_dict = {}
                            print(speed)
                            bbox_center = np.array(carla.Transform(bb.location, bb.rotation).get_matrix())
                            npc2w = np.array(npc.get_transform().get_matrix())
                            bbox_center2w = np.dot(npc2w, bbox_center)

                            p1 = get_image_point(bb.location, K, world_2_camera)
                            verts = [v for v in bb.get_world_vertices(npc.get_transform())]
                            x_max = -10000
                            x_min = 10000
                            y_max = -10000
                            y_min = 10000

                            for vert in verts:
                                p = get_image_point(vert, K, world_2_camera)
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
                                # TODO: change to nerf coordinate system here for center and extent

                                box_dict['center'] = carla2Nerf(bbox_center2w)
                                box_dict['extent'] = np.array([bb.extent.y, bb.extent.z, bb.extent.x])

                                box_dict['x_max'] = int(x_max)
                                box_dict['x_min'] = int(x_min)
                                box_dict['y_max'] = int(y_max)
                                box_dict['y_min'] = int(y_min)
                                cv2.line(img, (int(x_min), int(y_min)), (int(x_max), int(y_min)), (0, 0, 255, 255), 1)
                                cv2.line(img, (int(x_min), int(y_max)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)
                                cv2.line(img, (int(x_min), int(y_min)), (int(x_min), int(y_max)), (0, 0, 255, 255), 1)
                                cv2.line(img, (int(x_max), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255, 255), 1)

                                for edge in edges:
                                    p1 = get_image_point(verts[edge[0]], K, world_2_camera)
                                    p2 = get_image_point(verts[edge[1]], K, world_2_camera)
                                    #cv2.line(img, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), (255, 0, 0, 255), 1)
                                veh_dict[str(npc.id)] = box_dict
        time_dict[str(w_frame)] = veh_dict
        print(time_dict[str(w_frame)])

        cv2.imshow('ImageWindowName', img)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':

    main()