from smach import State
import rospy
from grip_ros.msg import StepTransition
from grip_ros.srv import EvaluatePick

class Evaluate(State):
    def __init__(self, robot, step_transition_publisher):
        super().__init__(input_keys=['target_bin_id', 'target_item_id'], outcomes=['done'])
        self.step_transition_publisher = step_transition_publisher
        self.robot = robot
        self.evaluation = None
        self.service = rospy.Service('~eval', EvaluatePick, self.callback)

    # TODO: move this somewhere common (also used in motion.py)
    def publish_step_transition(self, userdata):
        if self.step_transition_publisher is None:
            rospy.loginfo("Skipping publish_step_transition because step_transition_publisher is not set")
            return
        msg = StepTransition()
        msg.message = 'Evaluation'
        msg.available_primitive_type = StepTransition.PRIMITIVE_TYPE_NONE
        msg.bin_id = userdata['target_bin_id']
        msg.object_id = userdata['target_item_id']
        self.step_transition_publisher.publish(msg)

    def callback(self, msg):
        # self.robot.reset_pose()
        self.evaluation = True
        return {"success": True}

    def execute(self, userdata):
        self.publish_step_transition(userdata)

        while self.evaluation is None and not rospy.is_shutdown():
            rospy.sleep(.1)

        return "done"
    