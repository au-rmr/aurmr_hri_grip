import ROSLIB, { Message } from "roslib";
import { Timestamp } from "./messages";

export function timestamp(): Timestamp {
    return dateToTimestamp(new Date());
}

export function dateToTimestamp(date: Date): Timestamp {
    const secs = Math.floor(date.getTime()/1000);
    const nsecs = Math.round(1000000000*(date.getTime()/1000-secs));
    return {secs, nsecs};
}