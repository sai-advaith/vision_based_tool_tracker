#!/usr/bin/env python

"""
kalman_filter.py

Publishes kalman filter prediction based on measurements from apriltag detections

It listens to the following messages:

Measurement Message:
`/tag_detections` (of type `apriltag_ros/AprilTagDetectionArray`)


It publishes a message:
`/kalman_filter/pose` (of type `geometry_msgs/PoseWithCovarianceStamped`)

`/kalman_filter/state` (of type `nav_msgs/Odometry`)
"""

# ROS
import rospy

# Python libraries
import numpy as np
import math
from scipy.linalg import block_diag

# ROS messages and libraries
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Header
from std_msgs.msg import Int8
import tf

# Python files
import utils

class KalmanFilter(object):

    def __init__(self, namespace='kalman_filter'):
        """
        Initializing the parameters of the kalman filter
        """

        # Node Initialization
        rospy.init_node("kalman_filter", anonymous=True)

        # time stamp
        self.current_time = rospy.Time.now()
        self.previous_time = rospy.Time.now()
        self.delta_t = 0.0

        # broadcasting transform
        self.br = tf.TransformBroadcaster()

        # not publishing without receiving tag measurement
        self.received_tag_detection = False

        # Input Meassage
        self.tag_pose = AprilTagDetectionArray()

        # Output Message
        self.tool_state = Odometry()
        self.tool_pose = PoseWithCovarianceStamped()
        self.measurement = PoseWithCovarianceStamped()

        # interval between every message published
        self.OUTPUT_INTERVAL = rospy.get_param("~output_interval")

        # ROS publisher
        self.kf_state = rospy.Publisher("/kalman_filter/state", Odometry, queue_size=10)
        self.kf_pose = rospy.Publisher("/kalman_filter/pose", PoseWithCovarianceStamped, queue_size=10)

        self.filtered_measurement = rospy.Publisher("/iir/filter", PoseWithCovarianceStamped, queue_size=10)

        # Position initialization
        self.x = 0.
        self.y = 0.
        self.z = 0.

        # Orientation Initialization
        self.roll = 0.
        self.pitch = 0.
        self.yaw = 0.

        # Linear Velocity Initialization
        self.vel_x = 0.
        self.vel_y = 0.
        self.vel_z = 0.

        # Angular Velocity Initialization
        self.vel_roll = 0.
        self.vel_pitch = 0.
        self.vel_yaw = 0.

        # Tag Detected
        self.first_tag_detection = False

        # Vectors and Matices     
        self.X = np.array([self.x,self.y, self.z, self.vel_x, self.vel_y, self.vel_z, self.roll, self.pitch, self.yaw, self.vel_roll, self.vel_pitch, self.vel_yaw]).reshape(12,1)

        # linear elements of the state vector   
        self.p = np.array([self.x, self.y, self.z, self.vel_x, self.vel_y, self.vel_z]).reshape(6,1)
        self.pBar = np.zeros(self.p.shape)

        # Covariane matrices for Linear Elements
        self.P_1 = np.diag(rospy.get_param("~P1_diag"))
        self.P_1Bar = np.zeros(self.P_1.shape)

        # Covariance matrices for Angular Elements
        self.P_2 = np.diag(rospy.get_param("~P2_diag"))
        self.P_2Bar = np.zeros(self.P_2.shape)

        # Covariance Matrix
        self.P = np.identity(12)

        # Angular elements expressed a Rotation matrix
        self.R_t = np.identity(3)

        # State Transition Matrix
        self.F1 = np.zeros((6,6))
        self.F2 = np.zeros((6,6))

        # Process Noise for linear and angular
        self.Q1 = np.diag(rospy.get_param("~Q1_diag"))
        self.Q2 = np.diag(rospy.get_param("~Q2_diag"))

        # Measurement state vector and noise & transition matrix
        self.R1 = np.diag(rospy.get_param("~R1_diag"))
        self.R2 = np.diag(rospy.get_param("~R2_diag"))
        self.H = np.array( [[1., 0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0., 0.]])

        # Measurement vectors for orientation and position
        self.Z1 = np.array([self.x, self.y, self.z]).reshape(3,1)
        self.Z2 = np.array([self.roll, self.pitch, self.yaw]).reshape(3,1)

        # Orientation Vectors
        self.Q = np.array([self.roll, self.pitch, self.yaw, self.vel_roll, self.vel_pitch, self.vel_yaw]).reshape(6,1)
        self.QBar = np.zeros(self.Q.shape)

        # Data Vectors
        self.data_x = []
        self.data_y = []
        self.data_z = []

        self.data_roll = []
        self.data_pitch = []
        self.data_yaw = []

        # timer for the output message
        rospy.Timer(
            rospy.Duration(self.OUTPUT_INTERVAL),
            lambda x: self.kalman()
        )
        # Subscribe to topics
        self.tag_subscriber = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self.detections_receiver)

    def detections_receiver(self, msg):
        """
        Receives the Tag detections
        """

        if str(msg._type) == "apriltag_ros/AprilTagDetectionArray":
            self.received_tag_detection = True
            self.tag_pose = msg

    def predictPose(self, X, P, F, Q):
        """
        Predict Step of Kalman Filter
        """

        X_bar = F.dot(X)
        P_bar = F.dot(P.dot(np.transpose(F))) + Q

        return X_bar, P_bar

    def updateState(self, X_bar, P_bar, z, h, r):
        """
        Update Step of the Kalman Filter
        """
        try:
            y = z - h.dot(X_bar)

            S = h.dot(P_bar.dot(np.transpose(h))) + r

            K = P_bar.dot(np.transpose(h).dot(np.linalg.inv(S)))

            x = X_bar + K.dot(y)

            P = (np.identity(6) - K.dot(h)).dot(P_bar)

        except:
            # Handling situation where S is singular so that code does not crash
            rospy.logerr('Singular Matrix')
            x = X_bar
            P = P_bar

        return x,P

    def publishFilter(self, X, P):
        """
        Publishing the messages
        """

        self.tool_pose.header.frame_id = "camera"
        self.tool_state.header.frame_id = "camera"
        self.tool_state.child_frame_id = "odom"

        self.tool_pose.header.stamp.secs = rospy.Time.now().to_sec()
        self.tool_state.header.stamp.secs = rospy.Time.now().to_sec()

        self.br.sendTransform((X[0][0], X[1][0], X[2][0]),
            tf.transformations.quaternion_from_euler(X[6][0], X[7][0], X[8][0]),
            rospy.Time.now(),
            "camera",
            "odom")

        # publishing position in posewithCov
        self.tool_pose.pose.pose.position.x = X[0][0]
        self.tool_pose.pose.pose.position.y = X[1][0]
        self.tool_pose.pose.pose.position.z = X[2][0]

        # publishing position in odom
        self.tool_state.pose.pose.position.x = X[0][0]
        self.tool_state.pose.pose.position.y = X[1][0]
        self.tool_state.pose.pose.position.z = X[2][0]

        quat = tf.transformations.quaternion_from_euler(X[6][0], X[7][0], X[8][0])

        # publishing orientation in posewithCov
        self.tool_pose.pose.pose.orientation.x = quat[0]
        self.tool_pose.pose.pose.orientation.y = quat[1]
        self.tool_pose.pose.pose.orientation.z = quat[2]
        self.tool_pose.pose.pose.orientation.w = quat[3]

        # publishing orientation in odom
        self.tool_state.pose.pose.orientation.x = quat[0]
        self.tool_state.pose.pose.orientation.y = quat[1]
        self.tool_state.pose.pose.orientation.z = quat[2]
        self.tool_state.pose.pose.orientation.w = quat[3]

        # publishing linear velocity in odom
        self.tool_state.twist.twist.linear.x = X[3][0]
        self.tool_state.twist.twist.linear.y = X[4][0]
        self.tool_state.twist.twist.linear.z = X[5][0]

        # publishing angular velocity in odom
        self.tool_state.twist.twist.angular.x = X[9][0]
        self.tool_state.twist.twist.angular.y = X[10][0]
        self.tool_state.twist.twist.angular.z = X[11][0]

        # Converting the covariance from 6x6 to 36x1 and publishing
        self.tool_pose.pose.covariance = P[:6, :6].reshape(1,36) [0]

        self.tool_state.pose.covariance = P[:6, :6].reshape(1,36) [0]
        self.tool_state.twist.covariance = P[6:, 6:].reshape(1,36) [0]

        # publishing messages
        self.kf_pose.publish(self.tool_pose)
        self.kf_state.publish(self.tool_state)
    def publishMeasurement(self, Z1, Z2):
        """
        Published Filtered Measurement
        """

        self.measurement.pose.pose.position.x = Z1[0][0]
        self.measurement.pose.pose.position.y = Z1[1][0]
        self.measurement.pose.pose.position.z = Z1[2][0]

        quat = tf.transformations.quaternion_from_euler(Z2[0][0], Z2[1][0], Z2[2][0])

        self.measurement.pose.pose.orientation.x = quat[0]
        self.measurement.pose.pose.orientation.x = quat[1]
        self.measurement.pose.pose.orientation.x = quat[2]
        self.measurement.pose.pose.orientation.x = quat[3]

        self.filtered_measurement.publish(self.measurement)

    def kalman(self):
        """
        Implementing the Kalman Filter
        """
        self.current_time = rospy.Time.now()
        self.delta_t = (self.current_time - self.previous_time).to_sec()

        if self.tag_pose.detections == [] and not self.first_tag_detection:
            rospy.logerr("First tag not detected")
            # time update
            self.previous_time = self.current_time

            # publishing zeros as prediction if no tags detected
            self.kf_state.publish(self.tool_state)
            self.kf_pose.publish(self.tool_pose)

            # returning because first tag not detected
            return

        elif self.tag_pose.detections != []:
            try:
                # Handling case when wrong tag is detected by the camera
                quaternion = (
                            self.tag_pose.detections[0].pose.pose.pose.orientation.x,
                            self.tag_pose.detections[0].pose.pose.pose.orientation.y,
                            self.tag_pose.detections[0].pose.pose.pose.orientation.z,
                            self.tag_pose.detections[0].pose.pose.pose.orientation.w)

                euler = tf.transformations.euler_from_quaternion(quaternion)


                # measurement vector update
                self.Z1 = np.array([self.tag_pose.detections[0].pose.pose.pose.position.x,
                                self.tag_pose.detections[0].pose.pose.pose.position.y,
                                self.tag_pose.detections[0].pose.pose.pose.position.z]).reshape(3,1)

                self.Z2 = np.array([euler[0], euler[1], euler[2]]).reshape(3,1)

                # Appending all measurements
                self.data_x.append(self.Z1[0][0])
                self.data_y.append(self.Z1[1][0])
                self.data_z.append(self.Z1[2][0])

                self.data_roll.append(self.Z2[0][0])
                self.data_pitch.append(self.Z2[1][0])
                self.data_yaw.append(self.Z2[2][0])

                # Applying IIR and updating measurement
                self.Z1 = np.array([utils.data_filter(self.data_x),
                                    utils.data_filter(self.data_y),
                                    utils.data_filter(self.data_z)]).reshape(3,1)

                self.Z2 = np.array([utils.data_filter(self.data_roll),
                                    utils.data_filter(self.data_pitch),
                                    utils.data_filter(self.data_yaw)]).reshape(3,1)

            except:
                rospy.logerr('Incorrect Tag Detected')
                rospy.logerr(self.tag_pose.detections)

                return

            if not self.first_tag_detection:

                # Once first tag detected we have seen the object once
                self.first_tag_detection = True
                
                # getting first transformation matrix from measurements
                R_0 = tf.transformations.euler_matrix(euler[0], euler[1], euler[2])

                self.R_t = R_0[:3, :3]
        # Motion Model for linear state vector components
        self.F1 = np.array([[1., 0., 0., self.delta_t, 0., 0.],
                            [0., 1., 0., 0., self.delta_t, 0.],
                            [0., 0., 1., 0., 0., self.delta_t],
                            [0., 0., 0., self.R_t[0][0], self.R_t[0][1], self.R_t[0][2]],
                            [0., 0., 0., self.R_t[1][0], self.R_t[1][1], self.R_t[1][2]],
                            [0., 0., 0., self.R_t[2][0], self.R_t[2][1], self.R_t[2][2]]])
        # Motion Model for angular state vector compoentsn
        self.F2 = np.array([[1., 0., 0., self.delta_t, 0., 0.],
                            [0., 1., 0., 0., self.delta_t, 0.],
                            [0., 0., 1., 0., 0., self.delta_t],
                            [0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0., 1.]])

        # Predictions based on the motion model
        self.QBar, self.P_2Bar = self.predictPose(self.Q, self.P_2, self.F2, self.Q2)
        self.pBar, self.P_1Bar = self.predictPose(self.p, self.P_1, self.F1, self.Q1)


        if self.tag_pose.detections != []:
            # If measurement is received, state is updated
            self.Q, self.P_2 = self.updateState(self.QBar, self.P_2Bar, self.Z2, self.H, self.R2)
            self.p, self.P_1 = self.updateState(self.pBar, self.P_1Bar, self.Z1, self.H, self.R1)
            # time update
            self.previous_time = self.current_time
        else:
            # Only prediction when no measurement received
            rospy.logwarn("No tags detected")
            self.Q = self.QBar
            self.P_2 = self.P_2Bar
            self.p = self.pBar
            self.P_1 = self.P_1Bar
        # Rotation Matrix update
        self.R_t = tf.transformations.euler_matrix(self.Q[0][0], self.Q[1][0], self.Q[2][0]) [:3, :3]

        # State Vector update
        self.X = np.append(self.p, self.Q).reshape(12,1)
        self.P = block_diag(self.P_1, self.P_2)

        # Publishing state and Covariance
        self.publishFilter(self.X, self.P)
        self.publishMeasurement(self.Z1, self.Z2)

if __name__ == "__main__":
    kf = KalmanFilter()
    rospy.spin()
