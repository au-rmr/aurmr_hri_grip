import ROSLIB, { Message } from "roslib";

export interface SensorMessage {
    type: "sensor",
    subtype: string,
    name: "effort" | "transform" | "value" | "voltage" | "transcript",
    value: number | ROSLIB.Transform | string
}
export interface JointStateMessage {
    type: "jointState",
    jointState: RobotPose
}

export interface CameraInfo {
    [key: string]: string
}

// TODO (kavidey): this is currently a modified version of `ALL_JOINTS` in `robot.ts`, find a way to define them both in the same place
export type ValidJoints = 'joint_head_tilt' | 'joint_head_pan' | 'joint_gripper_finger_left' | 'wrist_extension' | 'joint_lift' | 'joint_wrist_yaw' | "translate_mobile_base" | "rotate_mobile_base" | 'gripper_aperture' | 'joint_arm_l0' | 'joint_arm_l1' | 'joint_arm_l2' | 'joint_arm_l3';

export type RobotPose = { [key in ValidJoints]?: number }

export interface NamedPose {
    name: string,
    description: string,
    jointState: RobotPose
}

export interface ROSCompressedImage extends Message {
    // header: string,
    format: string,
    data: string
}

export interface ROSJointState extends Message {
    name: [ValidJoints?],
    position: [number],
    effort: [number],
    velocity: [number],
}

export interface ROSBatteryState extends Message {
    voltage: number,
    percentage: number,
}

export interface ROSSpeechRecognition extends Message {
    transcript: string,
}

//http://docs.ros.org/en/lunar/api/nav_msgs/html/msg/MapMetaData.html
export interface ROSMapMetaData extends ROSLIB.Message {
    time: number,
    width: number,
    height: number,
    resolution: number,
    origin: ROSLIB.Pose
}

// http://docs.ros.org/en/lunar/api/nav_msgs/html/msg/OccupancyGrid.html
export interface ROSOccupancyGrid extends ROSLIB.Message {
    info: ROSMapMetaData,
    data: number[]
}

export interface Pose2D {
    x: number,
    y: number,
    theta?: number
}

export interface PickRequest {
    bin_id: string,
    item_id: number,
    item_description: string
}

export interface AutomatePick {
    bin_id: string,
    item_id: number,
    item_description: string,
    image: any,
    mask: any
}

export enum PrimitiveType {
    None = 0,
    Grasp,
    Probe,
    Extract
}

export interface StepTransition {
    available_primitive_type: PrimitiveType,
    message: string,
    bin_id: string,
    object_id: number
}

export interface KeyValue {
    key: string,
    value: string
}

export interface ExecutePrimitive {
    primitive_name: string,
    params: KeyValue[]
}

export interface Timestamp {
    secs: number,
    nsecs: number
}

export interface Event {
    event_type: string,
    metadata: KeyValue[],
    stamp: Timestamp
}

export interface RecordEvent {
    event: Event
}

export interface RecordImage {
    input_camera: number,
    output_topic: string,
    continue_recording: boolean
}

export type VelocityGoalArray = [{[key in ValidJoints]?: number}, {[key in ValidJoints]?: number}]