import React from 'react';
import ROSLIB from 'roslib';
import './ImageTopicView.css';
import loading from './loading.svg';
import crosshair from './crosshair.svg';
import {ROSCompressedImage} from './messages';

import Konva from 'konva';
// import { Stage, Layer, Circle, Rect, Text, Image } from 'react-konva';
import {SegmentAnything} from "react-segment-anything"
import useImage from 'use-image';

import { Button, Stack, Card, CardContent, Modal, Box, Typography, TextField, Slider } from "@mui/material";

const ort = require("onnxruntime-web");

interface IProps {
  topicName: string,
  ros: ROSLIB.Ros,
  handleMaskSaved: (mask: HTMLImageElement, image: HTMLImageElement) => void
}

interface IState {
  imageData: string | null
  width: number,
  height: number,
  embedding: any,
  rosEmbeddingService: any,
}

class ImageTopicMaskTool extends React.Component<IProps, IState> {
  private containerRef = React.createRef<HTMLDivElement>()
  private imageTopic: ROSLIB.Topic<ROSCompressedImage> | null = null
  // private cursorCircleRef = React.createRef<any>()

  constructor(props: IProps) {
    super(props);
    this.state = {
      imageData: null,
      width: 0,
      height: 0,
      embedding: null,
      rosEmbeddingService: null,
    };
  }

  componentDidMount() {
    // let topic = new ROSLIB.Topic<ROSCompressedImage>({
    //   ros: this.props.ros,
    //   name: this.props.topicName,
    //   messageType: 'sensor_msgs/CompressedImage'
    // });
    // topic.subscribe(this.callback.bind(this));
    // this.imageTopic = topic;

    let rosEmbeddingService = new ROSLIB.Service({
      ros: this.props.ros,
      name: "segment_embeddings/get_embedding",
      serviceType: "/grip_ros/GetBinEmbedding"
    });
    let request = new ROSLIB.ServiceRequest({})
    rosEmbeddingService.callService(request, (response) => {
      console.log("Embedding response:")
      const tensor = new ort.Tensor('float32', response.tensor_data, response.tensor_shape);
      console.log(tensor);
      const imageData = "data:image/jpg;base64," + response.image.data
      this.setState({embedding: tensor, imageData});
    });
    this.setState({rosEmbeddingService})
    this.syncWithContainerSize();
  }

  componentDidUpdate() {
    this.syncWithContainerSize();
  }

  // callback(message: ROSCompressedImage) {
  //   this.setState({
  //     imageData: "data:image/jpg;base64," + message.data
  //   });
  //   this.imageTopic?.unsubscribe();
  // }

  syncWithContainerSize() {
    if (!this.containerRef.current) return;
    if (this.containerRef.current?.offsetHeight && this.containerRef.current?.offsetWidth) {
      if (this.state.width !== this.containerRef.current?.offsetWidth ||
          this.state.height !== this.containerRef.current?.offsetHeight) {
        this.setState({
          width: this.containerRef.current.offsetWidth,
          height: this.containerRef.current.offsetHeight
        });
      }
    }
  }

  renderImageTopic() {
    let image = new window.Image();
    image.src = this.state.imageData!;
    const { width, height, embedding, rosEmbeddingService } = this.state;

    let crosshairImage = new window.Image();
    crosshairImage.src = crosshair;
    return (
      <div className="ImageTopicView-image">
        <SegmentAnything
            image={image}
            embedding={embedding}
            modelUrl={"/sam_onnx_quantized_example.onnx"}
            handleMaskSaved={this.props.handleMaskSaved} />
        <Button variant="contained" onClick={() => {
          let request = new ROSLIB.ServiceRequest({})
          rosEmbeddingService.callService(request, (response: any) => {
            const tensor = new ort.Tensor('float32', response.tensor_data, response.tensor_shape);
            const imageData = "data:image/jpg;base64," + response.image.data
            this.setState({embedding: tensor, imageData});
          });
        }}>Resync Embedding</Button>
      </div>
    );
  }
  

  render() {
    return (
      <div className="ImageTopicView" ref={this.containerRef}>
        {!this.state.imageData || !this.state.embedding ? (
          <div className="ImageTopicView-loading">
            <img src={loading} className="Loading-icon" alt="Loading" />
            Loading {this.props.topicName} image and embedding...
          </div>
        ) : this.renderImageTopic()}
      </div>
    );
  }
}

export default ImageTopicMaskTool;
