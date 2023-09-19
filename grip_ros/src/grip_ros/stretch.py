import sys
import rospy
import actionlib
from control_msgs.msg import FollowJointTrajectoryAction
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
import tf2_ros
from sensor_msgs.msg import PointCloud2, JointState
from std_srvs.srv import Trigger, TriggerRequest
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Twist
from tf2_geometry_msgs import PointStamped


class Stretch:
    def __init__(self, in_sim=False):
        self.in_sim = in_sim
        self.joint_state = None
        self.point_cloud = None

        self.trajectory_client = actionlib.SimpleActionClient('/stretch_controller/follow_joint_trajectory', FollowJointTrajectoryAction)
        server_reached = self.trajectory_client.wait_for_server(timeout=rospy.Duration(60.0))
        if not server_reached:
            print("cannot connect to follow_joint_trajectory")
            rospy.signal_shutdown('Unable to connect to arm action server. Timeout exceeded.')
            sys.exit()

        self.move_base_client = actionlib.SimpleActionClient('/move_base',MoveBaseAction)
        server_reached = self.move_base_client.wait_for_server(timeout=rospy.Duration(60.0))
        if not server_reached:
            print("unable to connect to move_base")
            rospy.signal_shutdown('Unable to connect to base action server. Timeout exceeded.')
            sys.exit()
        
        self.cmd_vel_publisher = rospy.Publisher('/stretch/cmd_vel', Twist, queue_size=10)


        self.tf2_buffer = tf2_ros.Buffer()
        self.tf2_listener = tf2_ros.TransformListener(self.tf2_buffer)
        
        self.point_cloud_subscriber = rospy.Subscriber('/camera/depth/color/points', PointCloud2, self.point_cloud_callback)
        # self.point_cloud_pub = rospy.Publisher('/' + node_topic_namespace + '/point_cloud2', PointCloud2, queue_size=1)

        self.joint_state_subscriber = rospy.Subscriber("/stretch/joint_states/", JointState, \
            self.joint_state_callback, queue_size=10)

        rospy.wait_for_service('/stop_the_robot')
        rospy.loginfo('Connected to /stop_the_robot service.')
        self.stop_the_robot_service = rospy.ServiceProxy('/stop_the_robot', Trigger)

        self.delta_x = 0.0
        self.starting_x = None

    def set_starting_x(self):
        t = self.tf2_buffer.get_latest_common_time('map', 'link_grasp_center')
        transform = self.tf2_buffer.lookup_transform('map', 'link_grasp_center', t)
        self.starting_x = transform.transform.translation.x

    def joint_state_callback(self, joint_state):
        self.joint_state = joint_state

    def point_cloud_callback(self, point_cloud):
        self.point_cloud = point_cloud
    
    def reset_pose(self, bin_id="1A"):
        print(f"RESET POSE, bin_id={bin_id}")
        joint_lift = {
            "1A": 0.3,
            "1B": 0.5,
            "1C": 0.9
        }[bin_id]
        print("RESET POSE W/ 1.5 DURATION??")
        self.move_to_pose({
            'joint_head_tilt': -0.5,
            'joint_head_pan': -3.14/2,
            'wrist_extension': 0.0,
            'joint_lift': joint_lift,
            'gripper_aperture': 0.0
        }, time_from_start=rospy.Duration(1.5))

        starting_x = self.starting_x
        self.set_starting_x()

        if starting_x == None:
            return
        
        print(f"Resetting to: x={starting_x}")

        goal = PointStamped()
        goal.point.x = starting_x
        self.command_velocity_towards_goal(goal, padding=0.0)
        self.set_starting_x()

    def extend_arm_towards_goal(self, goal, padding=0.1):
        t = self.tf2_buffer.get_latest_common_time('map', 'link_grasp_center')
        transform = self.tf2_buffer.lookup_transform('map', 'link_grasp_center', t)

        print(f'goal: {goal}')
        print(f'transform: {transform}')
        print(f'diff_x = {goal.point.x - transform.transform.translation.x}')
        print(f'diff_y = {goal.point.y - transform.transform.translation.y}')
        
        t = self.tf2_buffer.get_latest_common_time('map', 'link_grasp_center')
        transform = self.tf2_buffer.lookup_transform('map', 'link_grasp_center', t)
        print(f'diff_x = {goal.point.x - transform.transform.translation.x}')
        print(f'diff_y = {goal.point.y - transform.transform.translation.y}')
        cur_position = self.joint_state.position[self.joint_state.name.index('wrist_extension')]
        print(f'cur_position={cur_position}')

        diff_y = goal.point.y - transform.transform.translation.y

        diff_extension = -(diff_y) - padding
        print(f"+= {diff_extension}")
        self.move_to_pose({
            'wrist_extension': cur_position + diff_extension
        }, time_from_start=rospy.Duration(2.0))

    def move_lift_towards_goal(self, goal):
        t = self.tf2_buffer.get_latest_common_time('map', 'link_grasp_center')
        transform = self.tf2_buffer.lookup_transform('map', 'link_grasp_center', t)

        print(f'goal: {goal}')
        print(f'transform: {transform}')
        print(f'diff_y = {goal.point.y - transform.transform.translation.y}')
        print(f'diff_z = {goal.point.z - transform.transform.translation.z}')
        # for _ in range(3):
        t = self.tf2_buffer.get_latest_common_time('map', 'link_grasp_center')
        transform = self.tf2_buffer.lookup_transform('map', 'link_grasp_center', t)

        print('---')
        # print(f'diff_y = {goal.point.y - transform.transform.translation.y}')
        print(f'diff_z = {goal.point.z - transform.transform.translation.z}')
        diff_z = goal.point.z - transform.transform.translation.z
        
        cur_position = self.joint_state.position[self.joint_state.name.index('joint_lift')]
        print(f'cur_position={cur_position}')
        self.move_to_pose({
            'joint_lift': cur_position + diff_z
        })
    
    def command_velocity_towards_goal(self, goal, padding=-0.04):
        print('--goal--')
        print(goal)
        print('----')

        t = self.tf2_buffer.get_latest_common_time('map', 'link_grasp_center')
        transform = self.tf2_buffer.lookup_transform('map', 'link_grasp_center', t)
    
        moving = None
        sign = 1
        goal_x = goal.point.x + padding
        if (goal_x - transform.transform.translation.x) > 0:
            moving = 'forward'
            sign = 1
        if (goal_x - transform.transform.translation.x) < 0:
            moving = 'backward'
            sign = -1

        print(f"moving: {moving}")

        for _ in range(20):
            t = self.tf2_buffer.get_latest_common_time('map', 'link_grasp_center')
            transform = self.tf2_buffer.lookup_transform('map', 'link_grasp_center', t)

            diff = goal_x - transform.transform.translation.x

            print(f'diff: {diff}')

            if abs(diff) < 0.001:
                break

            if moving == 'backward' and diff > 0:
                break
        
            if moving == 'forward' and diff < 0:
                break

            magnitude = max(abs(diff), 0.005)
            magnitude = min(magnitude, 0.05)

            print(f"diff: {diff} ,  moving: {magnitude}")
            self.command_velocity(sign * magnitude)
            rospy.sleep(0.5)

    def command_velocity(self, x):
        twist = Twist()

        twist.linear.x = x
        twist.linear.y = 0.0
        twist.linear.z = 0.0
        twist.angular.x = 0.0
        twist.angular.y = 0.0
        twist.angular.z = 0.0

        self.delta_x += x
        self.cmd_vel_publisher.publish(twist)
    
    def move_base(self, goal):
        self.move_base_client.send_goal(goal)
        self.move_base_client.wait_for_result()
    
    def move_to_pose(self, pose, return_before_done=False, custom_contact_thresholds=False, time_from_start=rospy.Duration(1.0)):
        joint_names = [key for key in pose]
        point = JointTrajectoryPoint()
        point.time_from_start = time_from_start

        trajectory_goal = FollowJointTrajectoryGoal()
        trajectory_goal.goal_time_tolerance = rospy.Time(2.0)
        trajectory_goal.trajectory.joint_names = joint_names
        if not custom_contact_thresholds: 
            joint_positions = [pose[key] for key in joint_names]
            point.positions = joint_positions
            trajectory_goal.trajectory.points = [point]
        else:
            pose_correct = all([len(pose[key])==2 for key in joint_names])
            if not pose_correct:
                rospy.logerr("move_to_pose: Not sending trajectory due to improper pose. custom_contact_thresholds requires 2 values (pose_target, contact_threshold_effort) for each joint name, but pose = {0}".format(pose))
                return
            joint_positions = [pose[key][0] for key in joint_names]
            joint_efforts = [pose[key][1] for key in joint_names]
            point.positions = joint_positions
            point.effort = joint_efforts
            trajectory_goal.trajectory.points = [point]
        trajectory_goal.trajectory.header.stamp = rospy.Time.now()
        self.trajectory_client.send_goal(trajectory_goal)
        if not return_before_done: 
            self.trajectory_client.wait_for_result()
