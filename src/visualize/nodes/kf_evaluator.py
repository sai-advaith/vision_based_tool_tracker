#!/usr/bin/env python

"""
kf_evaluator.py

Evaluates kalman filter by comparing its pose estimate to the measurement estimate

It listens to the following messages:

Pose Message:
`/kalman_filter/state` (of type `nav_msgs/Odometry`)

Measurement:
`/tag_detections` (of type `apriltag_ros/AprilTagDetectionArray)

It publishes a message:
`/kf/evaluator` (of type `visualize/FilterErr`)
"""

import numpy as np
import rospy
import math
from nav_msgs.msg import Odometry
from visualize.msg import FilterErr
from apriltag_ros.msg import AprilTagDetectionArray
import tf

class KFEvaluator(object):
    
    def __init__(self,namespace='kf_evaluator'):
        """

        Initializing parameters of the evaluator
        """
        rospy.init_node("kf_evaluator", anonymous=True)

        # no publishing without receiving the messages
        self.received_tool_pose = False
        self.received_measurement_pose = False

        # input message
        self.tool_pose = Odometry()
        self.measurement_pose = AprilTagDetectionArray()

        # output message
        self.err_msg = FilterErr()

        # interval between every message published
        self.OUTPUT_INTERVAL = rospy.get_param("~output_interval", default=1)

        # output publisher
        self.output = rospy.Publisher("/kalman_filter/evaluator", FilterErr, queue_size=1)


        # timer for the output message
        rospy.Timer(
            rospy.Duration(self.OUTPUT_INTERVAL),
            lambda x: self.evaluate()
        )

        # Subscribe to topics
        self.tool_pose_sub = rospy.Subscriber("/kalman_filter/state", Odometry,self.tool_pose_receiver)
        self.measurement_sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray,self.measurement_receiver)

    def tool_pose_receiver(self,msg):
        """
        Receives the tool pose estimate
        """
        
        if str(msg._type) == "nav_msgs/Odometry":
            self.received_tool_pose = True
            self.tool_pose = msg

    def evaluate(self):
        """
        Does the actual evaluation of the tool pose, called at a consistent rate
        """

        if not self.received_tool_pose:
            # exiting if no KF prediction is received
            return

        if not self.received_measurement_pose:
            # exiting if no measurement is received
            return

        # receiving pose and velocities from the topics
        if (len(self.measurement_pose.detections) > 0):
            # Position from the tags
            measurement_linear_pose = np.array([self.measurement_pose.detections[0].pose.pose.pose.position.x,
                                                self.measurement_pose.detections[0].pose.pose.pose.position.y,
                                                self.measurement_pose.detections[0].pose.pose.pose.position.z])
            # Position from the Kalman Filter
            tool_linear_pose = np.array([self.tool_pose.pose.pose.position.x,
                                        self.tool_pose.pose.pose.position.y,
                                        self.tool_pose.pose.pose.position.z])
            # Orientation from tags
            measurement_quaternion = self.measurement_pose.detections[0].pose.pose.pose.orientation
            measurement_euler = tf.transformations.euler_from_quaternion((measurement_quaternion.x,
                                                                        measurement_quaternion.y,
                                                                        measurement_quaternion.z,
                                                                        measurement_quaternion.w))

            measurement_angular_pose = np.array([measurement_euler[0],
                                                measurement_euler[1],
                                                measurement_euler[2]])
            # Orientation from the Kalman Filter
            tool_quaternion = self.tool_pose.pose.pose.orientation
            tool_euler = tf.transformations.euler_from_quaternion((tool_quaternion.x,
                                                                tool_quaternion.y,
                                                                tool_quaternion.z,
                                                                tool_quaternion.w))

            tool_angular_pose = np.array([tool_euler[0],
                                                tool_euler[1],
                                                tool_euler[2]])
            # comparing and finding the norm of the difference

            self.err_msg.position_err = self.compare(measurement_linear_pose, tool_linear_pose)
            self.err_msg.orientation_err = self.compare(measurement_angular_pose, tool_angular_pose)

            self.err_msg.header.stamp = rospy.Time.now()
            self.output.publish(self.err_msg)
        else:
            # no detections received
            return


    def compare(self,v1,v2):
        """
        Comparing the vectors by calculating the norm of the difference
        """
        return np.linalg.norm(v1-v2)

    def measurement_receiver(self, msg):
        """Receives measurement from the tags"""
        if str(msg._type) == "apriltag_ros/AprilTagDetectionArray":
            self.received_measurement_pose = True
            self.measurement_pose = msg

if __name__ == "__main__":
    kf_evaluator = KFEvaluator()
    rospy.spin()