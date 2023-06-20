import React from 'react';
import ROSLIB from 'roslib';
import './ImageTopicView.css';
import loading from './loading.svg';
import crosshair from './crosshair.svg';
import {ROSCompressedImage} from './messages';

import Konva from 'konva';
import { Stage, Layer, Circle, Rect, Text, Image } from 'react-konva';
import useImage from 'use-image';

interface IProps {
  topicName: string,
  ros: ROSLIB.Ros,
  targetSelectEnabled: boolean,
  targetPosition: {x: number, y: number} | null,
  onTargetSelected: (e: {x: number, y: number}) => void
}

interface IState {
  imageData: string | null
  width: number,
  height: number,
}

class ImageTopicView extends React.Component<IProps, IState> {
  private containerRef = React.createRef<HTMLDivElement>()
  // private cursorCircleRef = React.createRef<any>()

  constructor(props: IProps) {
    super(props);
    this.state = {
      imageData: null,
      width: 0,
      height: 0,
    };
  }

  componentDidMount() {
    let topic = new ROSLIB.Topic<ROSCompressedImage>({
      ros: this.props.ros,
      name: this.props.topicName,
      messageType: 'sensor_msgs/CompressedImage'
    });
    topic.subscribe(this.callback.bind(this))
    this.syncWithContainerSize();
  }

  componentDidUpdate() {
    this.syncWithContainerSize();
  }

  callback(message: ROSCompressedImage) {
    this.setState({
      imageData: "data:image/jpg;base64," + message.data
    });
  }

  syncWithContainerSize() {
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

  onMouseDown(e: any) {
    if (!this.props.targetSelectEnabled) return;
    const pos = e.target.getStage().getPointerPosition();
    
    this.props.onTargetSelected({
      x: pos.x / this.state.width,
      y: pos.y / this.state.height
    });
  }

  renderImageTopic() {
    let image = new window.Image();
    image.src = this.state.imageData!;

    const { targetSelectEnabled, targetPosition } = this.props;
    const { width, height } = this.state;

    let crosshairImage = new window.Image();
    crosshairImage.src = crosshair;

    return (
      <div className="ImageTopicView-image">
        <Stage width={this.state.width} height={this.state.height} style={{cursor: targetSelectEnabled ? 'crossHair' : 'default'}}>
            <Layer
              onMouseDown={(e) => this.onMouseDown(e)}>
                <Image image={image} x={0} y={0} width={this.state.width} height={this.state.height} />
                {/* <Circle radius={10} fill='red' stroke='black' strokeWidth={3} opacity={0.4} x={305} y={250} /> */}
                {targetPosition ? (
                    // <Circle radius={10}
                    //         fill='white' 
                    //         stroke='black' 
                    //         strokeWidth={3} 
                    //         opacity={0.75} 
                    //         x={targetPos.x} 
                    //         y={targetPos.y} />
                    <Image image={crosshairImage}
                            width={40}
                            height={40}
                            x={(targetPosition!.x * width) - 20} 
                            y={(targetPosition!.y * height)- 20} />
                ) : null}
                {/* {targetSelectEnabled && showCursorCircle ? (
                    <Circle 
                            radius={10}
                            fill='white' 
                            stroke='black' 
                            strokeWidth={1} 
                            opacity={0.75} 
                            x={cursorCirclePos.x} 
                            y={cursorCirclePos.y} />
                ) : null}  */}
            </Layer>
        </Stage>
        {/* <img src={image} /> */}
      </div>
    );
  }
  

  render() {
    return (
      <div className="ImageTopicView" ref={this.containerRef}>
        {!this.state.imageData ? (
          <div className="ImageTopicView-loading">
            <img src={loading} className="Loading-icon" alt="Loading" />
            Loading {this.props.topicName}
          </div>
        ) : this.renderImageTopic()}
      </div>
    );
  }
}

export default ImageTopicView;
