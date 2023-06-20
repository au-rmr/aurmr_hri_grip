import React from 'react';
import ReactDOM from 'react-dom/client';
import ROSLIB from 'roslib';
import './index.css';
import App from './App';
import AppAutoBaseline from './AppAutoBaseline';

// Add some helpers to the window for easily retrieving params from the URL
declare global {
  interface Window { 
    params: (key: string, value: string) => string;
    exp: (key: string) => boolean;
  }
}

const searchParams = (new URL(document.location.toString())).searchParams;
function params(key: string, defaultValue: string = "") {
  const val = searchParams.get(key);
  if (val == null) {
    return defaultValue;
  }
  return val;
}

const exps = params("exp", "").split(",");
function exp(key: string): boolean {
  return exps.includes(key);
}

window.params = params
window.exp = exp

// Start the react app
const root = ReactDOM.createRoot(
  document.getElementById('root') as HTMLElement
);
if (exp("autobaseline")) {
    root.render(
        <AppAutoBaseline />
    );
} else {
  root.render(
      <App />
  );
}