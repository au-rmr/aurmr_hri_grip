import React from 'react';
import loading from './loading.svg';
import './Loading.css';

function Loading(props: {label: string}) {
  return (
    <div className="Loading">
        <img src={loading} className="Loading-icon" alt="Loading" />
        <div className="Loading-label">{props.label}</div>
    </div>
  );
}

export default Loading;
