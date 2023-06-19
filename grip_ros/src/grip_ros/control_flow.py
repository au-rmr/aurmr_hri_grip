import copy

import rospy
import smach
import uuid
from smach import State, StateMachine

from grip_ros.srv import PickRequest, ExecutePrimitive, RecordEvent, RecordImage, RecordImageRequest, SetBin
from grip_ros.msg import StepTransition, Event, KeyValue

# from aurmr_tasks.common import states


def break_dict_out(name, to, input_key, output_keys):
    class BreakDataOut(State):
        def __init__(self, ):
            State.__init__(self, outcomes=['succeeded'], input_keys=[input_key], output_keys=output_keys)
            self.output_keys = output_keys
            self.input_key = input_key

        def execute(self, userdata):
            for key in self.output_keys:
                userdata[key] = userdata[self.input_key][key]
            return 'succeeded'

    StateMachine.add(name, BreakDataOut(), transitions={'succeeded': to})


def break_dict_out_auto(name, input_key, output_keys):
    class BreakDataOut(State):
        def __init__(self, ):
            State.__init__(self, outcomes=['succeeded'], input_keys=[input_key], output_keys=output_keys)
            self.output_keys = output_keys
            self.input_key = input_key

        def execute(self, userdata):
            for key in self.output_keys:
                userdata[key] = userdata[self.input_key][key]
            return 'succeeded'

    StateMachine.add_auto(name, BreakDataOut(), ["succeeded"])


def inject_userdata(name, to, output_key, value):
    class InjectUserdata(State):
        def __init__(self, ):
            State.__init__(self, outcomes=['succeeded'], output_keys=[output_key])
            self.output_key = output_key

        def execute(self, userdata):
            userdata[output_key] = value
            return 'succeeded'

    StateMachine.add(name, InjectUserdata(), transitions={'succeeded': to})


def inject_userdata_auto(name, output_key, value):
    class InjectUserdata(State):
        def __init__(self, ):
            State.__init__(self, outcomes=['succeeded'], output_keys=[output_key])
            self.output_key = output_key

        def execute(self, userdata):
            userdata[output_key] = value
            return 'succeeded'

    StateMachine.add_auto(name, InjectUserdata(), ["succeeded"])


def input_to_output(input_key, output_key):
    class InputToOutput(smach.State):
        def __init__(self):
            smach.State.__init__(self, outcomes=['succeeded'], input_keys=[input_key], output_keys=[output_key])

        def execute(self, userdata):
            userdata[output_key] = userdata[input_key]
            return 'succeeded'

    return InputToOutput()


class Splat(smach.State):
    def __init__(self, input_key, output_keys):
        assert isinstance(output_keys, list)
        smach.State.__init__(self, outcomes=['succeeded'], input_keys=[input_key], output_keys=output_keys)
        # output keys need to be ordered, smach turns them into sets (unordered) save them here as a list
        self.output_key_list = output_keys

    def execute(self, userdata):
        input_key = list(self._input_keys)[0]
        in_data = userdata[input_key]
        for output_key, data in zip(self.output_key_list, in_data):
            userdata[output_key] = data
        return 'succeeded'


def splat_auto(name, input_key, output_keys, transitions=None):
    if transitions is None:
        transitions = {}
    StateMachine.add_auto(name, Splat(input_key, output_keys),["succeeded"], transitions=transitions)


def chain_states(*args):
    class ChainedStates(smach.State):
        def __init__(self):
            input_keys = []
            output_keys = []
            for state in args:
                input_keys += state._input_keys
                output_keys += state._output_keys

            # We want the outcome to be that of the last state
            smach.State.__init__(self, outcomes=args[-1]._outcomes, input_keys=input_keys, output_keys=output_keys)
            self.states = args

        def execute(self, userdata):
            outcome = None
            for state in self.states:
                outcome = state.execute(userdata)
            return outcome

    return ChainedStates()


class Sleep(smach.State):
    def __init__(self, seconds):
        self.seconds = seconds
        smach.State.__init__(self, outcomes=['succeeded'])

    def execute(self, userdata):
        rospy.sleep(self.seconds)
        return 'succeeded'


def retry_n(state, n, failure_status='aborted'):
    class RetryN(State):
        def __init__(self):
            State.__init__(self, outcomes=list(state._outcomes), input_keys=list(state._input_keys),
                           output_keys=list(state._output_keys))
            self.failure_status = failure_status
            self.repeat_state = state
            self.tries = 0

        def execute(self, userdata):
            status = self.failure_status
            while (status == self.failure_status
                   and self.tries < n):
                status = self.repeat_state.execute(userdata)
                self.tries += 1

            return status

    return RetryN()


class RepeatN(State):
    def __init__(self, n):
        State.__init__(self, outcomes=['repeat', 'done'], output_keys=[f"index"])
        self.counter = 0
        self.num_repetitions = n

    def execute(self, userdata):
        self.counter = self.counter + 1
        userdata["index"] = self.counter
        if self.counter >= self.num_repetitions:
            self.counter = 0
            userdata["index"] = self.counter
            return 'done'
        else:
            return 'repeat'


class IterateList(State):
    def __init__(self, key, outkey=None):
        outputs = ["index"]
        self.outkey = outkey
        if outkey:
            outputs.append(outkey)
        State.__init__(self, outcomes=['repeat', 'done'], input_keys=[key], output_keys=outputs)
        self.counter = -1
        self.key = key

    def execute(self, userdata):
        self.counter = self.counter + 1
        userdata["index"] = self.counter

        if self.counter >= len(userdata[self.key]):
            self.counter = 0
            userdata["index"] = self.counter
            return 'done'
        else:
            if self.outkey:
                userdata[self.outkey] = userdata[self.key][self.counter]
            return 'repeat'


class ResetRepeat(State):
    def __init__(self, repeat_state):
        State.__init__(self, outcomes=["succeeded"])
        self.repeat_state = repeat_state
        assert isinstance(repeat_state, RepeatN)

    def execute(self, ud):
        self.repeat_state.counter = 0
        return "succeeded"


def call_func(name, func, params):
    class CallFunction(State):
        def __init__(self):
            State.__init__(self, outcomes=["succeeded"])

        def execute(self, userdata):
            if len(params) == 0:
                func()
            else:
                func(params)
            return "succeeded"

    smach.StateMachine.add_auto(name, CallFunction(), ["succeeded"])


class TransitionBasedOnUserdata(State):
    def __init__(self, decide_based_on, forward_to):
        targets_copy = copy.deepcopy(forward_to)
        forwarding_targets = set(targets_copy)
        forwarding_targets.add("aborted")
        State.__init__(self, outcomes=list(forwarding_targets), input_keys=[decide_based_on])
        self.forward_to = targets_copy

    def execute(self, userdata):
        target = self.extract_from_userdata(userdata)
        # If we have the target registered, head towards it.
        if target in self.forward_to:
            return target
        return 'aborted'

    def extract_from_userdata(self, userdata):
        raise NotImplementedError


def remap_auto(name, from_key, to_key):
    class Remap(State):
        def __init__(self):
            State.__init__(self, outcomes=["succeeded"], input_keys=[from_key], output_keys=[to_key])

        def execute(self, ud):
            ud[to_key] = ud[from_key]
            return "succeeded"

    smach.StateMachine.add_auto(name, Remap(), ["succeeded"])


def remap(name, from_key, to_key, transitions={}):
    class Remap(State):
        def __init__(self):
            State.__init__(self, outcomes=["succeeded"], input_keys=[from_key], output_keys=[to_key])

        def execute(self, ud):
            ud[to_key] = ud[from_key]
            return "succeeded"

    smach.StateMachine.add(name, Remap(), transitions=transitions)


def select_nth_auto(name, from_key, i, to_key):
    class SelectNth(State):
        def __init__(self):
            State.__init__(self, outcomes=["succeeded"], input_keys=[from_key], output_keys=[to_key])

        def execute(self, ud):
            ud[to_key] = ud[from_key][i]
            return "succeeded"

    StateMachine.add_auto(name, SelectNth(), ["succeeded"])


def select_ith_auto(name, from_key, to_key):
    class SelectIth(State):
        def __init__(self):
            State.__init__(self, outcomes=["succeeded"], input_keys=[from_key, "index"], output_keys=[to_key])

        def execute(self, ud):
            ud[to_key] = ud[from_key][ud["index"]]
            return "succeeded"

    StateMachine.add_auto(name, SelectIth(), ["succeeded"])

class WaitForPick(State):
    def __init__(self):
        State.__init__(self, outcomes=['pick', 'done'], output_keys=["request"])
        self.bin_image_proxy = rospy.ServiceProxy("publish_bin_images/set_bin", SetBin)
        self.pick_service = rospy.Service('~pick', PickRequest, self.pick_cb)
        self.request = None

    def pick_cb(self, request: PickRequest):
        rospy.loginfo("PICK REQUEST: " + str(request))
        self.request = request
        self.bin_image_proxy(request.bin_id)
        return {"success": True}

    def execute(self, userdata):
        print("Waiting for pick...")

        while self.request is None and not rospy.is_shutdown():
            rospy.sleep(.1)

        # userdata["request"] = ["3f", "can"]
        if self.request is None:
            return "done"

        userdata["request"] = [self.request.bin_id, self.request.item_id]
        self.request = None
        return "pick"

class PublishStepTransition(State):
    def __init__(self, publisher, message=None, available_primitive_type=StepTransition.PRIMITIVE_TYPE_NONE):
        State.__init__(self, outcomes=['succeeded'], output_keys=[])
        self.message = message
        self.available_primitive_type = available_primitive_type
        self.publisher = publisher

    def execute(self, userdata):
        msg = StepTransition()
        msg.message = self.message
        msg.available_primitive_type = self.available_primitive_type
        msg.bin_id = userdata['target_bin_id']
        msg.object_id = userdata['target_object_id']

        self.publisher.publish(msg)

class PrimitiveExecutionService():
    def __init__(self):
        self.service = rospy.Service('~execute_primitive', ExecutePrimitive, self.callback)
        self.event_proxy = rospy.ServiceProxy('/data_collection/record_event', RecordEvent)
        self.img_proxy = rospy.ServiceProxy('/data_collection/record_image', RecordImage)
        self.registered_callback = None
        self.available_primitives = []
        self.done = False
        self.retry = False
        self.record_from_topic = RecordImageRequest.CAMERA_WRIST
        self.record_to_topics = {}

    def set_record_to_topic(self, primitive_name, topic):
        self.record_to_topics[primitive_name] = topic

    def set_registered_callback(self, callback):
        self.registered_callback = callback
        self.done = False
        self.retry = False

    def set_available_primitives(self, primitives):
        self.available_primitives = primitives + ['done', 'retry']
    
    def record_event(self, metadata, event_type=Event.EVENT_PRIMITIVE_EXEC):
        event = Event()
        event.event_type = event_type
        for key, value in metadata.items():
            kv = KeyValue()
            kv.key = key
            kv.value = value
            event.metadata.append(kv)
        event.stamp = rospy.Time.now()
        self.event_proxy(event)
    
    def record_image(self, input_camera, output_topic, continue_recording_for_steps=0.0):
        # img = RecordImage()
        # img.input_camera = input_camera
        # img.output_topic = output_topic
        # img.continue_recording = continue_recording
        self.img_proxy(input_camera, output_topic, continue_recording_for_steps)

    def callback(self, request):
        if request.primitive_name == 'done':
            self.done = True
            return {"success": True}
        if request.primitive_name == 'retry':
            self.retry = True
            return {"success": True}
        if self.registered_callback is None:
            return {"success": False}
        if request.primitive_name not in self.available_primitives:
            return {"success": False}

        prim_uuid = str(uuid.uuid4())

        # Record an event
        metadata = dict([(x.key, x.value) for x in request.params])
        metadata['primitive_name'] = request.primitive_name
        metadata['primitive_uuid'] = prim_uuid
        self.record_event(metadata)

        # Record an image
        # if request.primitive_name in self.record_to_topics:
        if request.primitive_name == 'probe_super':
            self.record_image(RecordImageRequest.CAMERA_WRIST, f'/probe_super/{prim_uuid}', 24)
        if request.primitive_name == 'align_target':
            self.record_image(RecordImageRequest.CAMERA_BIN, f'/align_target/{prim_uuid}', 0)
            self.record_image(RecordImageRequest.CAMERA_BIN_DEPTH, f'/align_target/{prim_uuid}/depth', 0)

        # Call the registered callback
        self.registered_callback(request)

        return {"success": True}

    def spin(self):
        while not self.done and not rospy.is_shutdown() and not self.retry:
            rospy.sleep(.1)
        
        if self.retry:
            return "retry"

        if rospy.is_shutdown():
            return "done"
        
        return "succeeded"

