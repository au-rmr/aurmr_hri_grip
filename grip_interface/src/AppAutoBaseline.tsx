import React from 'react';
import ROSLIB from "roslib";

import {
  PickRequest,
  StepTransition,
  PrimitiveType,
  ExecutePrimitive,
  KeyValue,
  ValidJoints,
  ROSJointState,
  VelocityGoalArray,
  Event,
  ROSCompressedImage,
  AutomatePick
} from './messages';

import { timestamp } from './utils';

import ImageTopicView from './ImageTopicView'
import ImagePrompt from './ImagePrompt';
import Loading from './Loading';
import {
  StreamlinedPickModal,
  InventoryManager,
  RandomizeInventoryModal,
  ReviewPickOrderModal,
  randomizeInventory
} from './InventoryManager';

// import {SegmentAnything} from "react-segment-anything"
import ImageTopicMaskTool from './ImageTopicMaskTool';

import { Button, Stack, Card, CardContent, Modal, Box, Typography, TextField, Slider } from "@mui/material";
import ModeStandbyIcon from '@mui/icons-material/ModeStandby';
import NavigateNextIcon from '@mui/icons-material/NavigateNext';
import MenuItem from '@mui/material/MenuItem';
import HelpIcon from '@mui/icons-material/Help';
import InfoIcon from '@mui/icons-material/Info';
import CachedIcon from '@mui/icons-material/Cached';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';

import ArrowDropUpIcon from '@mui/icons-material/ArrowDropUp';
import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown';
import ArrowLeftIcon from '@mui/icons-material/ArrowLeft';
import ArrowRightIcon from '@mui/icons-material/ArrowRight';

import './App.css';
import connected from './green-circle.png';
import loading from './loading.svg';

import '@fontsource/roboto/300.css';
import '@fontsource/roboto/400.css';
import '@fontsource/roboto/500.css';
import '@fontsource/roboto/700.css';

const AGENT_NODE = "pick_agent_baseline"

interface IProps {}

interface IState {
  ros: ROSLIB.Ros | null;
  rosPickService: ROSLIB.Service | null;
  rosExecPrimitiveService: ROSLIB.Service | null;
  rosEvalService: ROSLIB.Service | null;
  rosRecordEventService: ROSLIB.Service | null;
  rosRecordImageService: ROSLIB.Service | null;
  rosAdjustCropService: ROSLIB.Service | null;
  rosAutomatePickService: ROSLIB.Service | null;
  rosConnected: boolean;
  rosError: string;
  jointState: ROSJointState | null;
  targetObject: number | null;
  targetBin: string | null;
  availablePrimitiveType: PrimitiveType;
  currentStep: number;
  stepMessage: string;
  pickModalOpen: boolean;
  pickModalBinId: string;
  pickModalObj: string;
  pickModalLoading: boolean;
  evalModalLoading: boolean;
  streamModalOpen: boolean;
  streamModalLoading: boolean;
  streamModalSelected: number;
  streamModalSessionName: string;
  randomizeModalOpen: boolean;
  gripperWidth: number;
  distanceToTarget: number;
  selectingTarget: boolean;
  selectedTarget: {x: number, y: number} | null;
  inventory: { [key: string]: {id: number, description: string, images: string[]}[] };
  pickQueue: PickRequest[];
  evalCode: string;
  evalNotes: string;
  finished: boolean;
  loading: boolean;
}

const modalStyle = {
  position: 'absolute' as 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 240,
  bgcolor: 'background.paper',
  border: '1px solid #444',
  boxShadow: 24,
  p: 4,
};

const targetModalStyle = {
  position: 'absolute' as 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  width: 520,
  height: 575,
  bgcolor: 'background.paper',
  border: '1px solid #444',
  boxShadow: 24,
  p: 4,
};

const stepsPerPrimitiveType = (primitiveType: PrimitiveType) => {
  if (primitiveType == PrimitiveType.Grasp) {
    return 3;
  }
  return 1;
}

function getJointValue(jointStateMessage: ROSJointState, jointName: ValidJoints): number {
  // Paper over Hello's fake joint implementation
  if (jointName === "wrist_extension") {
      return getJointValue(jointStateMessage, "joint_arm_l0") +
          getJointValue(jointStateMessage, "joint_arm_l1") +
          getJointValue(jointStateMessage, "joint_arm_l2") +
          getJointValue(jointStateMessage, "joint_arm_l3")
  } else if (jointName === "translate_mobile_base" || jointName === "rotate_mobile_base") {
      return 0
  }
  let jointIndex = jointStateMessage.name.indexOf(jointName)
  return jointStateMessage.position[jointIndex]
}

function makeVelocityGoal(positions: VelocityGoalArray, velocities: VelocityGoalArray, trajectoryClient: ROSLIB.ActionClient) {
  let points = [];
  let jointNames;
  for (let i = 0; i < positions.length; i++) {
      let positionsT = positions[i];
      let velocitiesT = velocities[i];
      let positionsOut = [];
      let velocitiesOut = [];
      let names: [ValidJoints?] = [];
      for (let key in positionsT) {
          // Make sure that typescript knows that key will be a valid key
          const typedKey = key as ValidJoints;

          names.push(typedKey);
          positionsOut.push(positionsT[typedKey]);
          velocitiesOut.push(velocitiesT[typedKey]);
      }
      points.push({
          positions: positionsOut, velocities: velocitiesOut, time_from_start: {
              secs: i * 60,
              nsecs: 1
          }
      });
      jointNames = names;
  }
  let newGoal = new ROSLIB.Goal({
      actionClient: trajectoryClient,
      goalMessage: {
          trajectory: {
              header: {
                  stamp: {
                      secs: 0,
                      nsecs: 0
                  }
              },
              joint_names: jointNames,
              points: points
          }
      }
  });
  newGoal.on('feedback', function (feedback) {
      //console.log('Feedback: ', feedback);
  });

  newGoal.on('result', function (result) {
      console.log('Final Result: ', result);

      // if (goalCallback) {
      //     goalCallback({
      //         type: "velocity",
      //         goal: {
      //             positions: positions,
      //             velocities: velocities
      //         }
      //     })
      // }
  });

  //this.velocityGoal = newGoal;
  return newGoal;

}

class AppAutoBaseline extends React.Component<IProps, IState> {

  constructor(props: IProps) {
    super(props);
    this.state = {
      ros: null,
      rosPickService: null,
      rosExecPrimitiveService: null,
      rosConnected: false,
      rosEvalService: null,
      rosRecordEventService: null,
      rosRecordImageService: null,
      rosAdjustCropService: null,
      rosAutomatePickService: null,
      rosError: "",
      jointState: null,
      targetObject: null,
      targetBin: null,
      // targetObject: "Urban Infant Pipsqueak Small Pillow - Mini 11x7 - Tiny Pillow for Travel, Dogs, Kids and Chairs - Gray",
      // targetBin: "1C",
      pickModalOpen: false,
      pickModalBinId: "",
      pickModalObj: "",
      pickModalLoading: false,
      evalModalLoading: false,
      streamModalOpen: false,
      streamModalLoading: false,
      streamModalSelected: -1,
      streamModalSessionName: "",
      randomizeModalOpen: false,
      availablePrimitiveType: PrimitiveType.None,
      currentStep: 0,
      // availablePrimitiveType: PrimitiveType.Grasp,
      stepMessage: "",
      gripperWidth: 0.0,
      distanceToTarget: 0.1,
      selectingTarget: false,
      selectedTarget: null,
      pickQueue: [],
      inventory: {
        "1C": [],
        "1B": [],
        "1A": []
      },
      evalCode: 'success',
      evalNotes: "",
      finished: false,
      loading: false
    };
  }

  componentDidMount() {
    const ros = new ROSLIB.Ros({
      // url: 'ws://forky.hcrlab.cs.washington.edu:9090'
      // url: 'ws://kp:9090'
      // url: 'ws://localhost:9090'
      url: `ws://${process.env.REACT_APP_ROSBRIDGE_HOST}:9090`
    });

    ros.on('connection', () => {
      console.log('Connected to websocket server.');
      this.setState({
        rosConnected: true
      })
    });

    ros.on('error', (error: any) => {
      console.log('Error connecting to ROS websocket server: ', error);
      this.setState({
        rosError: "Error connecting to ROS websocket proxy",
        rosConnected: false
      });
    });

    ros.on('close', () => {
      console.log('Connection to ROS websocket server closed.');
      this.setState({
        rosError: "Connection to ROS websocket server closed.",
        rosConnected: false
      });
    });

    const rosEvalService = new ROSLIB.Service({
      ros: ros,
      name: "pick/eval",
      serviceType: "/grip_ros/EvaluatePick"
    });

    const rosPickService = new ROSLIB.Service({
      ros: ros,
      name: "pick/pick",
      serviceType: "/grip_ros/PickRequest"
    });

    const rosExecPrimitiveService = new ROSLIB.Service({
      ros: ros,
      name: "pick/execute_primitive",
      serviceType: "/grip_ros/ExecutePrimitive"
    });

    const rosRecordEventService = new ROSLIB.Service({
      ros: ros,
      name: "data_collection/record_event",
      serviceType: "/grip_ros/RecordEvent"
    });

    const rosRecordImageService = new ROSLIB.Service({
      ros: ros,
      name: "data_collection/record_image",
      serviceType: "/grip_ros/RecordImage"
    });

    const rosAdjustCropService = new ROSLIB.Service({
      ros: ros,
      name: "publish_bin_images/adjust_bin_crop",
      serviceType: "/grip_ros/AdjustBinCrop"
    });

    const rosAutomatePickService = new ROSLIB.Service({
      ros: ros,
      name: `${AGENT_NODE}/auto_pick`,
      serviceType: "/grip_ros/AutomatePick"
    });

    const stepTransitionListener = new ROSLIB.Topic<StepTransition>({
      ros : ros,
      name : 'pick/step_transition',
      messageType : '/grip_ros/StepTransition'
    });
  
    stepTransitionListener.subscribe((msg: StepTransition) => this.handleStepTransition(msg));

    this.setState({
      ros,
      rosPickService,
      rosExecPrimitiveService,
      rosEvalService,
      rosRecordEventService,
      rosRecordImageService,
      rosAdjustCropService,
      rosAutomatePickService
    });
  }

  handleStepTransition(message: StepTransition) {
    this.setState({
      availablePrimitiveType: message.available_primitive_type,
      stepMessage: message.message,
      targetBin: message.bin_id,
      targetObject: message.object_id
    })
  }

  handleSubmitEvalClick() {

    const { targetBin, targetObject, evalCode, evalNotes, pickQueue } = this.state;

    const request = new ROSLIB.ServiceRequest({
      code: evalCode,
      notes: evalNotes
    });

    this.state.rosEvalService!.callService(request, (success: boolean) => {
      if (!success) {
        alert("Failed to sendeval.");
      }

      this.sendRecordEvent({
        event_type: 'pick_eval',
        metadata: [
          {key: 'bin_id', value: targetBin!},
          {key: 'item_id', value: targetObject!.toString()},
          {key: 'eval_code', value: evalCode},
          {key: 'eval_notes', value: evalNotes}
        ],
        stamp: timestamp()
      });


      this.setState({
        evalModalLoading: false,
        stepMessage: "",
        loading: true
      });

      const nextPick = pickQueue.shift();
      if (!nextPick) {
        this.sendRecordEvent({
          event_type: 'session_end',
          metadata: [],
          stamp: timestamp()
        });
        this.setState({finished: true});
        return;
      }

      setTimeout(
        () => {
          this.sendRecordEvent({
            event_type: 'pick_start',
            metadata: [{key: 'bin_id', value: nextPick.bin_id}, {key: 'item_id', value: nextPick.item_id.toString()}],
            stamp: timestamp()
          });
          this.setState({pickQueue: pickQueue});
          this.sendPickRequest(nextPick!);
        }, 8000);
      
    });

    this.setState({
      rosError: "",
      targetObject: null,
      targetBin: null,
      // targetObject: "mug",
      // targetBin: "E4",
      pickModalOpen: false,
      pickModalBinId: "",
      pickModalObj: "",
      pickModalLoading: false,
      evalModalLoading: true,
      availablePrimitiveType: PrimitiveType.None,
      currentStep: 0,
      // availablePrimitiveType: PrimitiveType.Grasp,
      gripperWidth: 0.0,
      distanceToTarget: 0.1,
      selectingTarget: false,
      selectedTarget: null,
    });
  }

  sendPickRequest(pickRequest: PickRequest) {
    const request = new ROSLIB.ServiceRequest(pickRequest);

    this.setState({
      pickModalLoading: true,
    });

    this.state.rosPickService!.callService(request, (success: boolean) => {
      if (!success) {
        alert("Failed to send pick request.");
      }
      this.setState({
        pickModalLoading: false,
        pickModalOpen: false
      })
      
    });
  }

  sendRecordEvent(event: Event, cb: (success: boolean) => void = () => {}) {
    console.log(`sending event:`);
    console.log(event);
    const request = new ROSLIB.ServiceRequest({event: event});

    this.state.rosRecordEventService!.callService(request, (success: boolean) => {
      cb(success);
      if (!success) {
        alert("Failed to record event.");
      }
    });
  }

  sendAutomatePick(request: AutomatePick) {
    this.state.rosAutomatePickService!.callService(new ROSLIB.ServiceRequest(request), (success: boolean) => {
      if (!success) {
        alert("Failed to send automate pick request.");
      }
    });
  }

  sendDone() {
    const { rosExecPrimitiveService } = this.state;
    const request = new ROSLIB.ServiceRequest({
      primitive_name: "done",
      params: []
    });

    rosExecPrimitiveService!.callService(request, (success: boolean) => {
      if (!success) {
        alert("Failed to move to next step");
      }
    });
  }

  handleNextStepClick() {
    const { rosExecPrimitiveService, availablePrimitiveType, currentStep } = this.state;

    const steps = stepsPerPrimitiveType(availablePrimitiveType);
    if (currentStep >= steps-1) {
      const request1 = new ROSLIB.ServiceRequest({
        primitive_name: "close_gripper",
        params: []
      });
  
      rosExecPrimitiveService!.callService(request1, (success: boolean) => {
        if (!success) {
          alert("Failed to close gripper");
        }

        const request2 = new ROSLIB.ServiceRequest({
          primitive_name: "done",
          params: []
        });
    
        rosExecPrimitiveService!.callService(request2, (success: boolean) => {
          if (!success) {
            alert("Failed to move to next step");
          }
        });
      });

      this.setState({currentStep: 0, gripperWidth: 0.0, distanceToTarget: 0.1});
    } else {
      this.setState({currentStep: currentStep + 1});
    }
  }

  handleApplyDistanceToTargetClick() {
    const { rosExecPrimitiveService, distanceToTarget } = this.state;

    const request = new ROSLIB.ServiceRequest({
      primitive_name: "extend_arm",
      params: [{key: "padding", value: distanceToTarget.toString()}]
    });

    rosExecPrimitiveService!.callService(request, (success: boolean) => {
      if (!success) {
        alert("Failed to set distance to target");
      }
    });
  }

  handleApplyGripperWidthClick() {
    const { rosExecPrimitiveService, gripperWidth } = this.state;

    const request = new ROSLIB.ServiceRequest({
      primitive_name: "set_gripper_width",
      params: [{key: "width", value: gripperWidth.toString()}]
    });

    rosExecPrimitiveService!.callService(request, (success: boolean) => {
      if (!success) {
        alert("Failed to set gripper width");
      }
    });
  }

  handleApplyGraspPointClick() {
    const { rosExecPrimitiveService, selectedTarget } = this.state;

    const request = new ROSLIB.ServiceRequest({
      primitive_name: "align_target",
      params: [{key: "x", value: selectedTarget!.x.toString()},
               {key: "y", value: selectedTarget!.y.toString()}]
    });

    rosExecPrimitiveService!.callService(request, (success: boolean) => {
      if (!success) {
        alert("Failed to apply grasp point");
      }
    });

    this.setState({selectingTarget: false, selectedTarget: null});
  }

  handleSelectGraspTargetClick() {
    const { rosExecPrimitiveService, selectedTarget, targetBin } = this.state;

    const request = new ROSLIB.ServiceRequest({
      primitive_name: "reset_pose",
      params: [{key: "bin_id", value: targetBin!}]
    });

    rosExecPrimitiveService!.callService(request, (success: boolean) => {
      if (!success) {
        alert("Failed to reset pose");
      }
    });

    this.setState({selectingTarget: true});
  }
  
  handleRegraspClick(pickEval = "") {
    const { rosExecPrimitiveService, targetBin, targetObject } = this.state;

    const request = new ROSLIB.ServiceRequest({
      primitive_name: "retry",
      params: []
    });

    rosExecPrimitiveService!.callService(request, (success: boolean) => {
      if (!success) {
        alert("Failed to reset pose");
      }

      if (pickEval !== "") {
        this.sendRecordEvent({
          event_type: 'pick_eval',
          metadata: [
            {key: 'bin_id', value: targetBin!},
            {key: 'item_id', value: targetObject!.toString()},
            {key: 'eval_code', value: pickEval},
            {key: 'eval_notes', value: 'streamlined'},
            {key: 'streamlined', value: 'true'},
            {key: 'ignore', value: 'false'},
          ],
          stamp: timestamp()
        });
      }
    });
  }

  // gripper probe handlers
  handleProbeClick(primitiveName: string) {
    const { rosExecPrimitiveService, selectedTarget } = this.state;

    const request = new ROSLIB.ServiceRequest({
      primitive_name: primitiveName,
      params: []
    });

    rosExecPrimitiveService!.callService(request, (success: boolean) => {
      if (!success) {
        alert(`Failed to execute ${primitiveName}`);
      }
    });

    this.setState({selectingTarget: false, selectedTarget: null});
  }

  /////

  handleExtractLiftClick() {
    const { rosExecPrimitiveService, selectedTarget } = this.state;

    const request = new ROSLIB.ServiceRequest({
      primitive_name: "lift",
      params: []
    });

    rosExecPrimitiveService!.callService(request, (success: boolean) => {
      if (!success) {
        alert("Failed to execute lift");
      }
    });

    this.setState({selectingTarget: false, selectedTarget: null});
  }

  handleExtractPullBackClick() {
    const { rosExecPrimitiveService, selectedTarget } = this.state;

    const request = new ROSLIB.ServiceRequest({
      primitive_name: "pull_back",
      params: []
    });

    rosExecPrimitiveService!.callService(request, (success: boolean) => {
      if (!success) {
        alert("Failed to execute pull back");
      }
    });

    this.setState({selectingTarget: false, selectedTarget: null});
  }

  executeVelocityMove(jointName: ValidJoints, velocity: number) {
    // this.stopExecution();

    // let velocities: VelocityGoalArray = [{}, {}];
    // velocities[0][jointName] = velocity;
    // velocities[1][jointName] = velocity;
    // let positions: VelocityGoalArray = [{}, {}];
    // positions[0][jointName] = getJointValue(this.jointState!, jointName)

    // const jointLimit = JOINT_LIMITS[jointName];
    // if (!jointLimit) throw `Joint ${jointName} does not have limits`
    // positions[1][jointName] = jointLimit[Math.sign(velocity) === -1 ? 0 : 1]

    // this.velocityGoal = makeVelocityGoal(positions, velocities, this.trajectoryClient!)
    // this.velocityGoal.send()
    // this.affirmExecution()
  }

  executeCropAdjustment(direction: string, magnitude: number = 5) {
    const { rosAdjustCropService } = this.state;

    const request = new ROSLIB.ServiceRequest({direction, magnitude});

    rosAdjustCropService!.callService(request, (success: boolean) => {
      if (!success) {
        alert("Failed to execute crop adjustment");
      }
    });

    return false;
  }

  renderControls() {
    const {stepMessage, availablePrimitiveType, gripperWidth, distanceToTarget, selectingTarget, selectedTarget, currentStep, rosExecPrimitiveService, targetBin, targetObject} = this.state;

    if (availablePrimitiveType == PrimitiveType.None) {
      return (
        <Stack spacing={2}>
            <Card>
                <CardContent>
                  <Typography sx={{ mb: 2 }}>
                    {stepMessage}
                  </Typography>
                  <img src={loading} />
                </CardContent>
              </Card>
            </Stack>
      )
    }

    if (availablePrimitiveType == PrimitiveType.Grasp) {
      if (currentStep == 0) {
        return (
          <Stack spacing={2}>
            <Card>
              <CardContent>
                <Typography sx={{ mb: 2 }}>
                  Grasp Target
                </Typography>
                <Button onClick={() => this.handleSelectGraspTargetClick()} variant="contained">Select</Button>
                {/* {!selectingTarget ? (
                  <Button onClick={() => this.setState({selectingTarget: true})} variant="contained">Select</Button>
                ) : (
                  <div>
                    <Button onClick={() => this.handleApplyGraspPointClick()} variant="contained" size="small" disabled={selectedTarget == null}>Apply</Button>
                    &nbsp;
                    <Button onClick={() => this.setState({selectingTarget: false, selectedTarget: null})} variant="contained" size="small" color="error">Cancel</Button>
                  </div>
                )} */}
              </CardContent>
            </Card>
            <Button onClick={() => this.handleNextStepClick()} endIcon={<NavigateNextIcon />} style={{marginTop:'40px', fontWeight: 'bold'}} size="large">Next Step</Button>
            {window.exp('streamlined') ? (
              <Button onClick={() => this.sendDone()}  style={{marginTop:'10px', fontWeight: 'bold'}} size="large">Skip</Button>
            ) : null}
          </Stack>
        )
      } else if (currentStep == 1) {
        return(
        <Stack spacing={2}>
          <Card>
            <CardContent>
              <Typography sx={{ mb: 2 }}>
                Gripper Width
              </Typography>
              <Slider
                aria-label="Gripper Width"
                value={gripperWidth}
                onChange={(e, newval) => {
                  if (typeof newval === 'number') {
                    this.setState({gripperWidth: newval})
                  }
                }}
                onChangeCommitted={(e, newval) => {
                  if (typeof newval === 'number') {
                    this.handleApplyGripperWidthClick()
                  }
                }}
                valueLabelDisplay="auto"
                step={0.005}
                marks
                min={0.0}
                max={0.05}
              />
              {/* <Button onClick={() => this.handleApplyGripperWidthClick()} size="small" variant="contained">Apply</Button> */}
            </CardContent>
          </Card>
          <Button onClick={() => this.handleNextStepClick()} endIcon={<NavigateNextIcon />} style={{marginTop:'40px', fontWeight: 'bold'}} size="large">Next Step</Button>
        </Stack>
        )
      } else if (currentStep == 2) {
        return(
        <Stack spacing={2}>
           <Card>
            <CardContent>
              <Typography sx={{ mb: 2 }}>
                Distance to Target
              </Typography>
              <Slider
                aria-label="Distance to target"
                value={distanceToTarget}
                onChange={(e, newval) => {
                  if (typeof newval === 'number') {
                    this.setState({distanceToTarget: newval})
                  }
                }}
                onChangeCommitted={(e, newval) => {
                  if (typeof newval === 'number') {
                    this.handleApplyDistanceToTargetClick()
                  }
                }}
                valueLabelDisplay="auto"
                step={0.05}
                marks
                min={-0.1}
                max={0.2}
              />
              {/* <Button onClick={() => this.handleApplyDistanceToTargetClick()} size="small" variant="contained">Apply</Button> */}
            </CardContent>
          </Card>
          <Button onClick={() => this.handleNextStepClick()} endIcon={<NavigateNextIcon />} style={{marginTop:'40px', fontWeight: 'bold'}} size="large">Next Step</Button>
        </Stack>
        )
      }
    }

    if (availablePrimitiveType == PrimitiveType.Probe) {
      return (
        <Stack spacing={2}>
          <Card>
            <CardContent>
              <Typography sx={{ mb: 2 }}>
                Execute probes
              </Typography>
              {window.exp('allprobes') ? (
              <Stack spacing={2}>
                <Button onClick={() => this.handleProbeClick('probe_gripper')} variant="contained">Gripper</Button>
                <Button onClick={() => this.handleProbeClick('probe_lift')} variant="contained">Lift</Button>
                <Button onClick={() => this.handleProbeClick('probe_pull')} variant="contained">Pull</Button>
                <Button onClick={() => this.handleProbeClick('probe_push')} variant="contained">Push</Button>
                <Button onClick={() => this.handleProbeClick('probe_sides')} variant="contained">Side-to-side</Button>
              </Stack>
              ) : (
                <Stack spacing={2}>
                  <Button onClick={() => this.handleProbeClick('probe_super')} variant="contained">Lift and Pull</Button>
                </Stack>
              )}

            </CardContent>
          </Card>
          {window.exp('streamlined') ? (
            <Card>
              <CardContent>
                <Typography sx={{ mb: 2 }}>
                  Outcome
                </Typography>
                <Stack spacing={2}>
                <Button onClick={() => this.handleRegraspClick('success')} style={{fontWeight: 'bold'}} size="large" color="success" variant="contained">Good grasp</Button>
                <Button onClick={() => this.handleRegraspClick('fail_not_picked')} style={{fontWeight: 'bold'}} size="large" color="error" variant="contained">Ungrasped</Button>
                <Button onClick={() => this.handleRegraspClick('fail_multipick')} style={{fontWeight: 'bold'}} size="large" color="error" variant="contained">Multipick</Button>
                <Button onClick={() => {
                  this.sendDone()
                  this.sendRecordEvent({
                    event_type: 'pick_eval',
                    metadata: [
                      {key: 'bin_id', value: targetBin!},
                      {key: 'item_id', value: targetObject!.toString()},
                      {key: 'eval_code', value: 'success'},
                      {key: 'eval_notes', value: 'streamlined'},
                      {key: 'streamlined', value: 'false'},
                      {key: 'ignore', value: 'true'},
                    ],
                    stamp: timestamp()
                  });
                  this.setState({targetObject: null});
                }} style={{fontWeight: 'bold'}} size="large" color="warning">Done</Button>
                </Stack>
            </CardContent>
            </Card>
          ) : (
          <Card>
            <CardContent>
              <Typography sx={{ mb: 2 }}>
                Re-grasp
              </Typography>
              <Button onClick={() => this.handleRegraspClick()} style={{fontWeight: 'bold'}} size="large" color="error" variant="contained">Re-grasp</Button>
          </CardContent>
          </Card>
          )}
          <Button onClick={() => this.handleNextStepClick()} endIcon={<NavigateNextIcon />} style={{marginTop:'40px', fontWeight: 'bold'}} size="large">Next Step</Button>
          
        </Stack>
      );
    }

    if (availablePrimitiveType == PrimitiveType.Extract) {
      return (
        <Stack spacing={2}>
          <Card>
            <CardContent>
              <Typography sx={{ mb: 2 }}>
                Extraction
              </Typography>
              <Stack spacing={2}>
                <Button onClick={() => this.handleExtractLiftClick()} variant="contained">Lift</Button>
                <Button onClick={() => this.handleExtractPullBackClick()} variant="contained">Pull back</Button>
              </Stack>
    
            </CardContent>
          </Card>
          <Button onClick={() => this.handleNextStepClick()} endIcon={<NavigateNextIcon />} style={{marginTop:'40px', fontWeight: 'bold'}} size="large" color="success" variant="outlined">Finish</Button>
        </Stack>
      );
    }
  }

  renderPickInterface() {

    const {
      ros,
      rosError,
      rosConnected,
      targetObject,
      targetBin,
      pickModalOpen,
      pickModalBinId,
      pickModalObj,
      pickModalLoading,
      evalModalLoading,
      randomizeModalOpen,
      selectingTarget,
      stepMessage,
      inventory,
      pickQueue,
      evalCode,
      evalNotes
    } = this.state;

    let targetItem: {id: number, description: string, images: string[]} | undefined = undefined;
    if (targetObject && targetBin) {
      targetItem = inventory[targetBin!].find((item) => item.id == targetObject);
    }

    if (!targetItem) {
      return <div>Target item not found in inventory</div>
    }

    return (
      <div className="App-wrapper">
        <div className="App-header">
          <div className="App-header-content">
            <div className='App-header-connected'>
              <div className="connected-icon"></div>
                <div className="connected-label">Connected to robot</div>
            </div>
            <div style={{float:'left'}}>
            </div>
            <div className='App-header-abortPick'>
            <Button startIcon={<InfoIcon/>} color="primary" variant="contained" size="small" style={{marginRight:'10px'}}>
                Instructions
              </Button>

              <Button color="warning" variant="contained" size="small">
                Abort Pick
              </Button>
            </div>
          </div>
        </div>

        <div className="App-prompt">
          <div className="App-prompt-description-container App-prompt-container">
            <div className="App-prompt-header">
            <Typography variant="overline" display="block" gutterBottom>
              Item Description:
            </Typography>
            </div>
            <div className="App-prompt-description">
              <Typography variant="body1">
                {targetItem.description}
              </Typography>
            </div>
          </div>
          <div className="App-prompt-images App-prompt-container">
          <div className="App-prompt-header">
            <Typography variant="overline" display="block" gutterBottom>
              Images:
            </Typography>
            </div>
            <div className="App-prompt-images-gallery">
              <ImagePrompt images={targetItem.images} id={targetItem.id} />
            </div>
          </div>
          <div className="App-prompt-bin App-prompt-container">
          <div className="App-prompt-header">
            <Typography variant="overline" display="block" gutterBottom>
              Bin:
            </Typography>
            </div>
            <Typography variant="h5" display="block" gutterBottom style={{marginLeft:'20px'}}>
              {targetBin}
            </Typography>
          </div>
        </div>

        <div className="App-body">
          {!window.exp("autobaseline") ? (
            <>
          <div className="App-sidebar">
            {this.renderControls()}
          </div>
          <div className="App-imageTopicView">
            
            <ImageTopicView
              ros={ros!}
              topicName={(
                "/gripper_camera/image_raw/compressed"
              )}
              // targetSelectEnabled={selectingTarget}
              targetSelectEnabled={false}
              targetPosition={null}
              onTargetSelected={(pos) => {
                this.setState({
                  selectedTarget: pos
                })
              }}
            />
          </div>
          </>
          ) : (
            <>
            <ImageTopicMaskTool
              topicName="/binCamera/compressed"
              ros={ros!}
              handleMaskSaved={(mask, image) => {
                const maskData = mask.src.replace('data:image/png;base64,', '');
                const imageData = image.src.replace('data:image/jpg;base64,', '');
                // const maskBuffer = Buffer.from(maskData, 'base64');
                // const imageBuffer = Buffer.from(imageData, 'base64');

                const automatePickReq = {
                  'bin_id': targetBin!,
                  'item_id': targetObject!,
                  'item_description': targetItem!.description,
                  'mask': {
                    'format': "png",
                    'data': maskData
                  },
                  'image': {
                    'format': "jpeg",
                    'data': imageData
                  }
                }
                this.sendAutomatePick(automatePickReq);
              }} />
            </>
          )}
          <Modal
            open={selectingTarget}
            onClose={() => this.setState({selectingTarget: false})}
            aria-labelledby="modal-modal-title"
            aria-describedby="modal-modal-description"
            className="targetSelectModal"
          >
            <Card sx={targetModalStyle}>
              <Typography id="modal-modal-title" variant="h6" component="h2">
                Select Grasp Target
              </Typography>
            
              <ImageTopicView
                ros={ros!}
                topicName={(
                  "/binCamera/compressed"
                )}
                // targetSelectEnabled={selectingTarget}
                targetSelectEnabled={true}
                targetPosition={this.state.selectedTarget}
                onTargetSelected={(pos) => {
                  this.setState({
                    selectedTarget: pos
                  })
                }}
              />

                <Typography sx={{ mt: 2 }}>
                  <Button onClick={() => this.handleApplyGraspPointClick()} variant="contained" size="small" disabled={this.state.selectedTarget == null}>Apply</Button>
                  &nbsp;
                  <Button onClick={() => this.setState({selectingTarget: false, selectedTarget: null})} variant="contained" size="small" color="error">Cancel</Button>
                </Typography>
          <div>
        </div>
              </Card>
          </Modal>

        <Modal
          open={stepMessage == 'Evaluation'}
          onClose={() => this.setState({stepMessage: 'Done'})}
          aria-labelledby="modal-modal-title"
          aria-describedby="modal-modal-description"
        >
          <Card sx={modalStyle}>
            <Typography id="modal-modal-title" variant="h6" component="h2">
              Evaluation
            </Typography>
            {evalModalLoading ? (
              <Typography sx={{ mt: 2 }}>
                <img src={loading} />
              </Typography>
            ) : (
              <div>
                <Typography sx={{ mt: 2 }}>
                <TextField
                    id="outlined-select-currency"
                    select
                    label="Select"
                    value={evalCode}
                    onChange={(event: React.ChangeEvent<HTMLInputElement>) => {
                      this.setState({evalCode: event.target.value});
                    }}
                    helperText="Outcome"
                  >
                    {[
                      {value:'success', label: 'Successful pick'},
                      {value: 'fail_not_picked', label: 'Unable to pick target item'},
                      {value: 'fail_multipick', label: 'Picked additional items'},
                      {value: 'fail_wrong_item', label: 'Picked wrong item'}
                      ].map((option) => (
                      <MenuItem key={option.value} value={option.value}>
                        {option.label}
                      </MenuItem>
                    ))}
                  </TextField>
                </Typography>
                <Typography sx={{ mt: 2 }}>
                  <TextField  id="outlined-basic" label="Notes" variant="outlined" value={evalNotes} />
                </Typography>
                <Typography sx={{ mt: 2 }}>
                  <Button color="success" variant="contained" onClick={() => this.handleSubmitEvalClick()}>
                    Submit
                  </Button>
                </Typography>
              </div>
            )}
          </Card>
        </Modal>
        </div>

      </div>
    );
  }

  renderApp() {
    const {
      rosError,
      rosConnected,
      targetObject,
      pickModalOpen,
      pickModalLoading,
      randomizeModalOpen,
      streamModalLoading,
      streamModalOpen,
      streamModalSelected,
      streamModalSessionName,
      inventory,
      pickQueue,
      finished,
      loading
    } = this.state;

    if (finished) {
      return <p className='error'>Session complete.</p>;
    }

    if (rosError) {
      return <p className='error'>{rosError}</p>;
    }

    if (!rosConnected) {
      return <Loading label="Connecting to robot..." />;
    }

    if (targetObject != null) {
      return this.renderPickInterface();
    }

    if (pickQueue.length > 0) {
      return (
        <div className='connected'>
          <div className="connected-icon"></div>
          <div className="connected-label">Connected to robot. Waiting for pick request...</div>
        </div>
      );
    }

    const hasInventory = inventory['1C'].length > 0 || inventory['1B'].length > 0 || inventory['1A'].length > 0;

    return (
      <div className='connected'>
          <div className="connected-icon"></div>
          <div className="connected-label">Connected to robot. Waiting for mask request...</div>
          <div className="connected-pick-button">
            {window.exp('streamlined') ? (
              <Button color="success" startIcon={<PlayArrowIcon />} variant="contained" size="small" onClick={() => this.setState({streamModalOpen: true})}>
                Start Picks
              </Button>
            ) : (
              <>
            <Button startIcon={<CachedIcon/>} variant="contained" size="small" onClick={() => this.setState({randomizeModalOpen: true})} style={{marginRight: '10px'}}>
              Randomize Inventory
            </Button>
            <Button color="success" disabled={!hasInventory} startIcon={<PlayArrowIcon />} variant="contained" size="small" onClick={() => this.setState({pickModalOpen: true})}>
              Start Picks
            </Button>
            </>
            )}
          </div>
          {!window.exp('streamlined') ? (
            <>
          <InventoryManager inventory={inventory} onUpdate={(inventory: any) => this.setState({inventory})} />
          <ReviewPickOrderModal
            loading={pickModalLoading}
            open={pickModalOpen}
            onClose={() => this.setState({pickModalOpen: false})}
            onSubmit={(items: any, name: string) => {
              let pickReqs = items.map((item: any) => (
                {bin_id: item.bin, item_id: item.id, item_description: item.description}
              ));
              let pickOrder = pickReqs.map((pickReq: any) => pickReq.item_id);
              const firstPick = pickReqs[0];
              this.sendRecordEvent({
                event_type: 'session_start',
                metadata: [
                  {key: 'picks', value: JSON.stringify(pickReqs)},
                  {key: 'pick_order', value: JSON.stringify(pickOrder)},
                  {key: 'name', value: name}],
                stamp: timestamp()
              }, (success: boolean) => {
                if (!success) return;
                this.sendRecordEvent({
                  event_type: 'pick_start',
                  metadata: [{key: 'bin_id', value: firstPick.bin_id}, {key: 'item_id', value: firstPick.item_id.toString()}],
                  stamp: timestamp()
                });
              });
              pickReqs.shift();
              this.setState({pickQueue: pickReqs});
              this.sendPickRequest(firstPick);
            }}
            inventory={inventory}
          />
          <RandomizeInventoryModal
            open={randomizeModalOpen}
            onClose={() => this.setState({randomizeModalOpen: false})}
            onRandomize={(numItemsPerBin: number) => this.setState({inventory: randomizeInventory(numItemsPerBin), randomizeModalOpen: false})}
          />
          </>
          ) : (
          <StreamlinedPickModal
            open={streamModalOpen}
            onClose={() => this.setState({streamModalOpen: false})}
            onSubmit={(item: {bin: string, id: number, description: string}, sessionName) => {
              let pickReqs = [{bin_id: item.bin, item_id: item.id, item_description: item.description}];
              const firstPick = pickReqs[0];

              const recordPickStart = () => {
                this.sendRecordEvent({
                  event_type: 'pick_start',
                  metadata: [{key: 'bin_id', value: firstPick.bin_id}, {key: 'item_id', value: firstPick.item_id.toString()}],
                  stamp: timestamp()
                });
              };

              if (!streamModalSessionName || streamModalSessionName === "") {
                this.sendRecordEvent({
                  event_type: 'session_start',
                  metadata: [
                    {key: 'picks', value: JSON.stringify([])},
                    {key: 'pick_order', value: JSON.stringify([])},
                    {key: 'name', value: "_SL_" + sessionName}],
                  stamp: timestamp()
                }, (success: boolean) => {
                  if (!success) return;
                  recordPickStart();
                });
                this.setState({streamModalSessionName: sessionName});
              } else {
                recordPickStart();
              }

              let inventory: any = {'1A': [], '1B': [], '1C': []};
              inventory[item.bin] = [item];
              pickReqs.shift();
              this.setState({
                pickQueue: pickReqs,
                inventory
              });

              this.sendPickRequest(firstPick);
              
            }}
            onFinished={(sessionName) => {
              this.sendRecordEvent({
                event_type: 'session_end',
                metadata: [],
                stamp: timestamp()
              });
              this.setState({finished: true});
            }}
            loading={streamModalLoading}
            sessionName={streamModalSessionName}
            />
          )}
        </div>
    );
  }

  render() {
    return (
      <div className="App">
        { this.renderApp() }
      </div>
    );
  }
}

export default AppAutoBaseline;
