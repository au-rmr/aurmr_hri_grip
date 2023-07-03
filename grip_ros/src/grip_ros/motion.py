
from smach import State
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, PointCloud2, JointState
from visualization_msgs.msg import Marker
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from grip_ros.msg import StepTransition, CropInfo
import ros_numpy
import cv2
import tf2_ros
import image_geometry
from scipy.spatial import cKDTree
from tf2_geometry_msgs import PointStamped
import tf2_sensor_msgs
import numpy as np
# from geometry_msgs.msg import Vector3Stamped, PointStamped


class MotionState(State):
    def __init__(self, robot, \
                 step_transition_publisher=None, step_transition_message="", \
                 available_primitive_type=StepTransition.PRIMITIVE_TYPE_NONE, \
                 input_keys=['target_bin_id', 'target_item_id'],
                 output_keys=[], outcomes=['succeeded', 'preempted', 'aborted']):
        super().__init__(input_keys=input_keys, output_keys=output_keys, outcomes=outcomes)
        self.robot = robot
        self.step_transition_publisher = step_transition_publisher
        self.step_transition_message = step_transition_message
        self.available_primitive_type = available_primitive_type
    
    def publish_step_transition(self, userdata):
        if self.step_transition_publisher is None:
            rospy.loginfo("Skipping publish_step_transition because step_transition_publisher is not set")
            return
        msg = StepTransition()
        msg.message = self.step_transition_message
        msg.available_primitive_type = self.available_primitive_type
        msg.bin_id = userdata['target_bin_id']
        msg.object_id = userdata['target_item_id']
        self.step_transition_publisher.publish(msg)

class MotionPrimitiveServiceState(MotionState):
    def __init__(self, robot, service, \
                 step_transition_publisher=None, step_transition_message="", \
                 available_primitive_type=StepTransition.PRIMITIVE_TYPE_NONE, \
                 outcomes=['succeeded', 'preempted', 'aborted'],
                 output_keys=[]):
        super().__init__(robot, outcomes=outcomes, \
                            step_transition_publisher=step_transition_publisher, \
                            step_transition_message=step_transition_message, \
                            available_primitive_type=available_primitive_type, \
                            output_keys=output_keys)
        self.service = service
    
    def get_param(self, params, key, type, default=None):
        for x in params:
            if x.key == key:
                return type(x.value)
        return default


class MoveToBin(MotionState):
    def __init__(self, robot, step_transition_publisher=None):
        super().__init__(robot, outcomes=['succeeded', 'preempted', 'aborted'], \
                            step_transition_publisher=step_transition_publisher, \
                            step_transition_message="Moving to bin...")

    def execute(self, userdata):
        self.publish_step_transition(userdata)

        base_goal = MoveBaseGoal()
        base_goal.target_pose.header.frame_id = "map"
        base_goal.target_pose.header.stamp = rospy.Time.now()
        # Move 0.5 meters forward along the x axis of the "map" coordinate frame 
        # base_goal.target_pose.pose.position.x = -0.992
        base_goal.target_pose.pose.position.x = -1.27
        base_goal.target_pose.pose.position.y = -0.6407
        # No rotation of the mobile base frame w.r.t. map frame
        base_goal.target_pose.pose.orientation.x = 0.0
        base_goal.target_pose.pose.orientation.y = 0.0
        base_goal.target_pose.pose.orientation.z = 0.0
        base_goal.target_pose.pose.orientation.w = 1.4
        # self.robot.move_base(base_goal)
        # -0.992, -0.307
        # pose = {
        #     'joint_head_tilt': -0.5,
        #     'joint_head_pan': -3.14/2,
        #     # 'joint_lift': 0.5,
        #     'joint_lift': 0.3,
        #     'joint_wrist_yaw': 0.0,
        #     'joint_arm_l0': 0.0,
        #     'joint_arm_l1': 0.0,
        #     'joint_arm_l2': 0.0,
        #     'joint_arm_l3': 0.0,
        #     'gripper_aperture': 0.0
        # }
        self.robot.reset_pose(userdata['target_bin_id'])
        rospy.sleep(3)

        return "succeeded"


class Grasp(MotionPrimitiveServiceState):
    def __init__(self, robot, service, step_transition_publisher=None):
        super().__init__(robot, service, step_transition_publisher=step_transition_publisher, \
                            available_primitive_type=StepTransition.PRIMITIVE_TYPE_GRASP, \
                            output_keys=['target_grasp_point', 'target_grasp_z'])
        print("subscribing...")
        self.image_subscriber = rospy.Subscriber("/camera/color/image_raw/compressed",
            CompressedImage, self.image_callback,  queue_size = 1)
        self.latest_image = None
        self.cv_bridge = CvBridge()

        if robot.in_sim:
            self.depth_image_subscriber = rospy.Subscriber("/camera/depth/image_raw", Image, self.depth_image_callback)
        else:
            self.depth_image_subscriber = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback)

        self.latest_depth_image = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.gripper_camera_model = None
        self.gripper_camera_info_subscriber = rospy.Subscriber("/gripper_camera/camera_info",
            CameraInfo, self.gripper_camera_info_callback,  queue_size = 1)
        self.camera_model = None
        self.camera_info_subscriber = rospy.Subscriber("/camera/color/camera_info",
            CameraInfo, self.camera_info_callback,  queue_size = 1)

        self.latest_bin_image = None
        self.bin_image_subscriber = rospy.Subscriber("/binCamera/compressed", CompressedImage, self.bin_image_callback)
        self.latest_bin_image_info = None
        self.bin_image_info_subscriber = rospy.Subscriber("/binCamera/camera_info", CameraInfo, self.bin_image_info_callback)
        
        
        self.latest_points = None
        self.points_subscriber = rospy.Subscriber("/camera/depth/color/points",
            PointCloud2, self.points_callback, queue_size = 1)
        
        self.marker_publisher = rospy.Publisher("visualization_marker", Marker)

        self.debug_image_publisher = rospy.Publisher("debug_image", Image, queue_size=10)

        self.bin_affine_configs = None
        self.crop_info = None
        self.crop_info_sub = rospy.Subscriber("/binCamera/crop_info", CropInfo, self.crop_info_callback)

        self.userdata = {}

        self.map_point = None

    def build_affine_transform(self, roi_xyxy, des_width, des_height):
        src_points = [[roi_xyxy[0], roi_xyxy[1]], [roi_xyxy[2], roi_xyxy[1]], [roi_xyxy[2], roi_xyxy[3]]]
        src_points = np.array(src_points, dtype=np.float32)

        des_points = [[0, 0], [des_width, 0], [des_width, des_height]]
        des_points = np.array(des_points, dtype=np.float32)

        M = cv2.getAffineTransform(des_points, src_points)
        return M

    def crop_info_callback(self, msg):
        self.crop_info = msg

    def image_callback(self, msg):
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_image = img
    
    def depth_image_callback(self, ros_depth_image):
        self.latest_depth_image = ros_numpy.numpify(ros_depth_image).astype(np.float32)

    def bin_image_callback(self, msg):
        img = self.cv_bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.latest_bin_image = img
    
    def bin_image_info_callback(self, msg):
        self.latest_bin_image_info = msg
    
    def camera_info_callback(self, msg):
        self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(msg)
        self.camera_info_subscriber.unregister() #Only subscribe once
    
    def gripper_camera_info_callback(self, msg):
        self.gripper_camera_model = image_geometry.PinholeCameraModel()
        self.gripper_camera_model.fromCameraInfo(msg)
        self.gripper_camera_info_subscriber.unregister() #Only subscribe once
    
    def points_callback(self, msg):
        self.latest_points = msg

    def primitive_request_callback(self, request):
        print(f"Primitive request: {request.primitive_name}")
        if request.primitive_name == 'align_target':
            x = self.get_param(request.params, 'x', float, default=0.0)
            y = self.get_param(request.params, 'y', float, default=0.0)
            self.execute_align_target_from_realsense(x, y)
        elif request.primitive_name == 'pre_align_target':
            self.execute_pre_align_target()
        elif request.primitive_name == 'set_gripper_width':
            width = self.get_param(request.params, 'width', float, default=0.0)
            self.execute_set_gripper_width(width)
        elif request.primitive_name == 'extend_arm':
            padding = self.get_param(request.params, 'padding', float, default=0.1)
            self.execute_extend_arm(padding)
        elif request.primitive_name == 'reset_pose':
            bin_id = self.get_param(request.params, 'bin_id', str, default="1A")
            self.robot.reset_pose(bin_id)
        elif request.primitive_name == 'close_gripper':
            self.robot.move_to_pose({
                'gripper_aperture': -0.07
            })
        elif request.primitive_name == 'move_to_pose':
            pose = {}
            for param in request.params:
                pose[param.key] = float(param.value)
            self.robot.move_to_pose(pose)
        elif request.primitive_name == 'move_base':
            base_goal = MoveBaseGoal()
            base_goal.target_pose.header.frame_id = self.get_param(request.params, 'frame_id', str, default="map")
            base_goal.target_pose.header.stamp = rospy.Time.now()
            # Move 0.5 meters forward along the x axis of the "map" coordinate frame 
            # base_goal.target_pose.pose.position.x = -0.992
            base_goal.target_pose.pose.position.x = self.get_param(request.params, 'x', float, default=0.0)
            base_goal.target_pose.pose.position.y = self.get_param(request.params, 'y', float, default=0.0)
            # No rotation of the mobile base frame w.r.t. map frame
            base_goal.target_pose.pose.orientation.x = self.get_param(request.params, 'ox', float, default=0.0)
            base_goal.target_pose.pose.orientation.y = self.get_param(request.params, 'oy', float, default=0.0)
            base_goal.target_pose.pose.orientation.z = self.get_param(request.params, 'oz', float, default=0.0)
            base_goal.target_pose.pose.orientation.w = self.get_param(request.params, 'ow', float, default=1.0)
            self.robot.move_base(base_goal)
        else:
            return {"success": False}
        
        return {"success": True}
    
    def visualize_point(self, point, frame_id):
        marker2 = Marker()
        # marker2.header.frame_id = self.camera_model.tfFrame()
        marker2.header.frame_id = frame_id
        marker2.header.stamp = rospy.rostime.Time.now()
        marker2.ns = "basic_shapes"
        marker2.id = 1
        marker2.type = Marker.SPHERE
        marker2.action = Marker.ADD

        print(f"publishing {point}")
        marker2.pose.position.x = point.point.x
        marker2.pose.position.y = point.point.y
        marker2.pose.position.z = point.point.z
        marker2.pose.orientation.x = 0.0;
        marker2.pose.orientation.y = 0.0;
        marker2.pose.orientation.z = 0.0;
        marker2.pose.orientation.w = 1.0;

        # marker2.points = [p1, p2]
        

        # Set the scale of the marker -- 1x1x1 here means 1m on a side
        marker2.scale.x = 0.05
        marker2.scale.y = 0.05
        marker2.scale.z = 0.05

        # Set the color -- be sure to set alpha to something non-zero!
        marker2.color.r = 0.0
        marker2.color.g = 1.0
        marker2.color.b = 0.0
        marker2.color.a = 1.0

        marker2.lifetime = rospy.rostime.Duration()

        # self.marker_publisher.publish(marker)
        self.marker_publisher.publish(marker2)
    
    def visualize_ray(self, ray, frame_id, length=2):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = rospy.rostime.Time.now()
        marker.ns = "basic_shapes"
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        p1 = Point()
        p1.x = 0
        p1.y = 0
        p1.z = 0

        p2 = Point()
        p2.x = ray[0] * length
        p2.y = ray[1] * length
        p2.z = ray[2] * length

        marker.points = [p1, p2]
        

        # Set the scale of the marker -- 1x1x1 here means 1m on a side
        marker.scale.x = 0.01
        marker.scale.y = 0.01
        marker.scale.z = 0.05

        # Set the color -- be sure to set alpha to something non-zero!
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        marker.lifetime = rospy.rostime.Duration()

        self.marker_publisher.publish(marker)

    def execute_pre_align_with_target(self):
        pose = {
            'joint_lift': 0.7,
        }
        self.robot.move_to_pose(pose)
    
    def execute_extend_arm(self, padding):
        if self.map_point is None:
            return
        self.robot.extend_arm_towards_goal(self.map_point, padding)
    
    def execute_set_gripper_width(self, width):
        print(f"execute_set_gripper_width({width})")
        self.robot.move_to_pose({'gripper_aperture': width})

    def execute_align_with_target_from_gripper_cam(self, u, v):
        print(f"execute_align_with_target_from_gripper_cam({u}, {v})")
        gripper_image_width = self.gripper_camera_model.width
        gripper_image_height = self.gripper_camera_model.height

        if self.gripper_camera_model is not None and self.latest_points is not None:
            print('computing point...')
            gripper_u = u * gripper_image_width
            gripper_v = v * gripper_image_height

            ray = self.gripper_camera_model.projectPixelTo3dRay((gripper_u, gripper_v))
            ray_z = [el / ray[2] for el in ray]
            self.visualize_ray(ray, self.gripper_camera_model.tfFrame())

            points_gripper = self.tf_buffer.transform(self.latest_points, self.gripper_camera_model.tfFrame())
            xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(points_gripper, remove_nans=True)

            pc = xyz_array
            p1 = np.array([0,0,0])
            p0 = np.array(ray_z) * 2
            closest_point = pc[np.argmin(np.linalg.norm(np.cross(p1-p0, p0-pc, axisb=1), axis=1)/np.linalg.norm(p1-p0))]
            self.visualize_point(closest_point, self.gripper_camera_model.tfFrame())

    def execute_align_target_from_realsense(self, u, v):
        print(f"execute_align_target_from_realsense({u}, {v})")

        if self.latest_depth_image is not None and self.crop_info is not None:
            # u and v are ratios of the image size
            bin_u = u * self.latest_bin_image_info.width
            bin_v = v * self.latest_bin_image_info.height

            # The user is clicking on a "bin image", which is a cropped/scaled portion of the realsense image.
            # We need to transform the bin image coordinates to the realsense by reversing the affine transform.
            bin_point = np.array([[bin_u, bin_v]], dtype=np.float32)
            bin_id = self.userdata['target_bin_id']
            M = self.build_affine_transform(self.crop_info.xxyy, self.crop_info.width, self.crop_info.height)
            transformed_point = cv2.transform(bin_point.reshape(1, 1, 2), M)
            rotated_image_u = transformed_point[0, 0, 0]
            rotated_image_v = transformed_point[0, 0, 1]

            # The realsense image is also rotated 90 degrees, so we need to rotate the coordinates back.
            image_point = (rotated_image_v, self.camera_model.height - rotated_image_u)

            self.debug_image_publisher.publish(self.cv_bridge.cv2_to_imgmsg( cv2.circle(self.latest_image, (int(image_point[0]), int(image_point[1])), 10, (0,255,0), -1)))

            # Now we can use the camera model + depth image to project the point into 3d space.
            ray = self.camera_model.projectPixelTo3dRay((image_point[0], image_point[1]))
            depth_z = self.latest_depth_image[int(image_point[1]), int(image_point[0])]
            ray_z = [el/ray[2] for el in ray]
            camera_point = np.array(ray_z) * (depth_z/1000)

            # Transform the point from the camera frame to the map frame.
            camera_point_stamped = PointStamped()
            camera_point_stamped.point.x = camera_point[0]
            camera_point_stamped.point.y = camera_point[1]
            camera_point_stamped.point.z = camera_point[2]
            camera_point_stamped.header.frame_id = self.camera_model.tfFrame()
            map_point = self.tf_buffer.transform(camera_point_stamped, "map")

            # grasp_point = self.tf_buffer.transform(camera_point_stamped, "link_grasp_center")
            self.visualize_point(map_point, "map")

            # Move the robot to align with the target point
            self.robot.command_velocity_towards_goal(map_point)
            self.robot.move_lift_towards_goal(map_point)
            self.robot.extend_arm_towards_goal(map_point)

            self.userdata['target_grasp_point'] = map_point
            self.map_point = map_point

    def execute(self, userdata):
        self.userdata = userdata
        self.publish_step_transition(userdata)
        self.service.set_registered_callback(self.primitive_request_callback)
        self.service.set_available_primitives([
            'align_target',
            'set_gripper_width',
            'move_to_pose',
            'move_base',
            'extend_arm',
            'close_gripper',
            'reset_pose'])
        return self.service.spin()


class Probe(MotionPrimitiveServiceState):
    def __init__(self, robot, service, step_transition_publisher=None):
        super().__init__(robot, service, step_transition_publisher=step_transition_publisher, \
                            available_primitive_type=StepTransition.PRIMITIVE_TYPE_PROBE,
                            outcomes=['succeeded', 'retry', 'preempted', 'aborted'])
    
    def primitive_request_callback(self, request):
        print(f"Primitive request: {request.primitive_name}")
        if request.primitive_name == 'probe_gripper':
            magnitude = self.get_param(request.params, 'magnitude', float, default=0.1)
            self.execute_probe_gripper(magnitude)
        elif request.primitive_name == 'probe_lift':
            magnitude = self.get_param(request.params, 'magnitude', float, default=0.1)
            self.execute_probe_lift(magnitude)
        elif request.primitive_name == 'probe_pull':
            magnitude = self.get_param(request.params, 'magnitude', float, default=0.1)
            self.execute_probe_pull(magnitude)
        elif request.primitive_name == 'probe_push':
            magnitude = self.get_param(request.params, 'magnitude', float, default=0.1)
            self.execute_probe_push(magnitude)
        elif request.primitive_name == 'probe_sides':
            magnitude = self.get_param(request.params, 'magnitude', float, default=0.1)
            self.execute_probe_sides(magnitude)
        elif request.primitive_name == 'probe_super':
            magnitude = self.get_param(request.params, 'magnitude', float, default=0.1)
            self.execute_probe_super(magnitude)
        else:
            return {"success": False}
        
        return {"success": True}

    def execute_probe_gripper(self, magnitude):
        self.robot.move_to_pose({
            'gripper_aperture': 0.02
        })
        rospy.sleep(1.0)
        joint_state = self.robot.joint_state
        # TODO: return to current aperture
        self.robot.move_to_pose({
            'gripper_aperture': -0.06
        })

    def execute_probe_lift(self, magnitude):
        joint_state = self.robot.joint_state
        cur_position = joint_state.position[joint_state.name.index('joint_lift')]
        self.robot.move_to_pose({
            'joint_lift': cur_position + 0.025
        })
        rospy.sleep(1.0)
        self.robot.move_to_pose({
            'joint_lift': cur_position
        })

    def execute_probe_pull(self, magnitude):
        joint_state = self.robot.joint_state
        cur_position = joint_state.position[joint_state.name.index('wrist_extension')]
        self.robot.move_to_pose({
            'wrist_extension': cur_position - 0.03
        })
        rospy.sleep(1.0)
        self.robot.move_to_pose({
            'wrist_extension': cur_position
        })

    def execute_probe_push(self, magnitude):
        joint_state = self.robot.joint_state
        cur_position = joint_state.position[joint_state.name.index('wrist_extension')]
        self.robot.move_to_pose({
            'wrist_extension': cur_position + 0.02
        })
        rospy.sleep(1.0)
        self.robot.move_to_pose({
            'wrist_extension': cur_position
        })

    def execute_probe_sides(self, magnitude):
        self.robot.command_velocity(0.05)
        rospy.sleep(1.0)
        self.robot.command_velocity(-0.05)
        rospy.sleep(1.0)
        self.robot.command_velocity(-0.05)
        rospy.sleep(1.0)
        self.robot.command_velocity(0.05)

    def execute_probe_super(self, magnitude):
        joint_state = self.robot.joint_state
        cur_wrist_position = joint_state.position[joint_state.name.index('wrist_extension')]
        cur_position = joint_state.position[joint_state.name.index('joint_lift')]
        self.robot.move_to_pose({
            # 'joint_lift': cur_position + 0.025
            'joint_lift': cur_position + 0.04
        })
        rospy.sleep(0.2)
        self.robot.move_to_pose({
            # 'wrist_extension': cur_wrist_position - 0.03
            'wrist_extension': cur_wrist_position - 0.03
        })
        rospy.sleep(0.5)
        self.robot.move_to_pose({
            'wrist_extension': cur_wrist_position + 0.02
        })
        self.robot.move_to_pose({
            'joint_lift': cur_position
        })
    

    def execute(self, userdata):
        self.publish_step_transition(userdata)
        self.service.set_registered_callback(self.primitive_request_callback)
        self.service.set_available_primitives([
            'probe_gripper',
            'probe_lift',
            'probe_pull',
            'probe_push',
            'probe_sides',
            'probe_super',
            'regrasp'])
        return self.service.spin()


class Extract(MotionPrimitiveServiceState):
    def __init__(self, robot, service, step_transition_publisher=None):
        super().__init__(robot, service, step_transition_publisher=step_transition_publisher, \
                            available_primitive_type=StepTransition.PRIMITIVE_TYPE_EXTRACT)

    def primitive_request_callback(self, request):
        print(f"Primitive request: {request.primitive_name}")
        if request.primitive_name == 'lift':
            magnitude = self.get_param(request.params, 'magnitude', float, default=0.1)
            self.execute_lift(magnitude)
        elif request.primitive_name == 'pull_back':
            self.execute_pull_back()
        else:
            return {"success": False}
        
        return {"success": True}

    def execute_pull_back(self):
        self.robot.move_to_pose({
            'wrist_extension': 0.0
        })

    def execute_lift(self, magnitude):
        joint_state = self.robot.joint_state
        cur_position = joint_state.position[joint_state.name.index('joint_lift')]
        self.robot.move_to_pose({
            'joint_lift': cur_position + 0.01
        })

    def execute(self, userdata):
        self.publish_step_transition(userdata)
        self.service.set_registered_callback(self.primitive_request_callback)
        self.service.set_available_primitives([
            'lift',
            'pull_back',])
        return self.service.spin()


class DropTargetObject(MotionState):
    def __init__(self, robot, step_transition_publisher=None):
        super().__init__(robot, outcomes=['succeeded', 'preempted', 'aborted'], \
                            step_transition_publisher=step_transition_publisher, \
                            step_transition_message="Dropping object...")
    
    def execute(self, userdata):
        self.publish_step_transition(userdata)

        self.robot.move_to_pose({
            'joint_lift': 0.2,
        })

        rospy.sleep(1.)
        self.robot.move_to_pose({
            'gripper_aperture': 0.05,
        })
        return "succeeded"


class ResetForGrasp(MotionState):
    def __init__(self, robot, step_transition_publisher=None):
        super().__init__(robot, outcomes=['succeeded', 'preempted', 'aborted'], \
                            step_transition_publisher=step_transition_publisher, \
                            step_transition_message="Resetting...")
    
    def execute(self, userdata):
        self.publish_step_transition(userdata)

        self.robot.move_to_pose({
            'gripper_aperture': 0.03,
        })
        rospy.sleep(1.)
        self.robot.move_to_pose({
            'wrist_extension': 0.1
        })
        self.robot.move_to_pose({
            'gripper_aperture': 0.0,
        })
        return "succeeded"