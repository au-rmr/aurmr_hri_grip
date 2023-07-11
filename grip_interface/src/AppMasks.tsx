
import React, { useState, useEffect } from 'react';
import ROSLIB from "roslib";

import Loading from './Loading';
import ImagePrompt from './ImagePrompt';
import { AnnotateMaskGoal, AnnotateMaskResult, Pose2D } from './messages';
import { SegmentAnything } from 'react-segment-anything';

import { Button, Stack, Card, CardContent, Modal, Box, Typography, TextField, Slider } from "@mui/material";

import './App.css';
import { request } from 'http';

const ort = require("onnxruntime-web");
const idb = require('./inventory.json');

let inventory = [...idb.items];
inventory = inventory.filter((item: any) => !item.hasOwnProperty('disabled') || item.disabled === false);

function getItem(id: number) {
    return inventory.find((item: any) => item['id'] === id);
}

export default function AppMasks() {
    // ROS connection from state
    const [ros, setRos] = useState<ROSLIB.Ros | undefined>(undefined);
    const [rosConnected, setRosConnected] = useState(false);
    const [rosError, setRosError] = useState("");
    const [rosAnnotationServer, setRosAnnotationServer] = useState<ROSLIB.SimpleActionServer | undefined>(undefined);

    // Action server goal and result from state
    const [maskGoal, setMaskGoal] = useState<AnnotateMaskGoal | undefined>(undefined);
    const [embedding, setEmbedding] = useState<any>(undefined);
    const [imageData, setImageData] = useState<string | null>(null);
    // const [regraspResult, setRegraspResult] = useState<AnnotateMask | undefined>(undefined);

    // Connect to ROS
    useEffect(() => {
        setRos(new ROSLIB.Ros({
            url: `ws://${process.env.REACT_APP_ROSBRIDGE_HOST}:9090`
        }));
    }, []);

    // Initialize ROS callbacks and services
    useEffect(() => {
        if (!ros) return;

        ros.on("connection", () => {
            console.log('Connected to websocket server.');
            setRosConnected(true);
        });

        ros.on('error', (error: any) => {
            console.log('Error connecting to ROS websocket server: ', error);
            setRosError('Error connecting to ROS websocket server');
        });

        setRosAnnotationServer(new ROSLIB.SimpleActionServer({
            ros: ros,
            serverName: 'annotate_mask',
            actionName: 'grip_ros/AnnotateMaskAction'
        }));
    }, [ros]);

    // Initialize retry action server callbacks
    useEffect(() => {
        if (!rosAnnotationServer) return;

        /* @ts-ignore */
        rosAnnotationServer.on('goal', (goalMessage: RetryGraspGoal) => {
            console.log(goalMessage);
            const tensor = new ort.Tensor('float32', goalMessage.tensor_data, goalMessage.tensor_shape);
            const imageData = "data:image/jpg;base64," + goalMessage.image.data
            setMaskGoal(goalMessage);
            setEmbedding(tensor);
            setImageData(imageData);
        });

        /* @ts-ignore */
        rosAnnotationServer.on('cancel', (goalMessage: any) => {
            console.log("Cancelled...");
            rosAnnotationServer.setPreempted();
        });
    }, [rosAnnotationServer]);

    // Render error or loading screen if necessary
    if (rosError) {
        return <p className='error'>{rosError}</p>;
    }
    if (!rosConnected) {
        return <Loading label="Connecting to robot..." />;
    }

    // Render waiting screen if no requests are in the queue
    if (!maskGoal || !imageData || !embedding) {
        return (
            <div className='App'>
            <div className='connected'>
                <div className="connected-icon"></div>
                <div className="connected-label">Connected to ROS. Waiting for mask request...</div>
            </div>
            </div>
        );
    }

    const image = new window.Image();
    image.src = imageData;

    const targetItem = getItem(maskGoal!.item_id);

    return (
        <div className='App'>
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
                    {maskGoal!.bin_id}
                    </Typography>
                </div>
            </div>
            <div style={{width:'80%', maxWidth: '900px', margin: 'auto'}}>
            <SegmentAnything
                image={image}
                embedding={embedding}
                modelUrl={"/sam_onnx_quantized_example.onnx"}
                initialClicks={[{x: maskGoal.grasp_x, y: maskGoal.grasp_y, pointType: 1}]}
                handleMaskSaved={(mask, image) => {
                    const maskData = mask.src.replace('data:image/png;base64,', '');
                    rosAnnotationServer!.setSucceeded({
                        mask: { format: "png", data: maskData },
                        success: true
                    });
                    setMaskGoal(undefined);
                    setEmbedding(undefined);
                    setImageData(null);
                }} />
            </div>
        </div>
    );
    
}