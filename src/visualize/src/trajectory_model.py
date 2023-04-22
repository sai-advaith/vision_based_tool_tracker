#!/usr/bin/env python

"""
trajectory_model.py

Publishes kalman filter prediction based on measurements from a fixed trajectory

It listens to the following messages:

Pose Message:
`/linear/path` (of type `geometry_msgs/PoseWithCovarianceStamped`)

AND

`/linear/state` (of type `nav_msgs/Odometry`)

It publishes a message:
`/kalman_filter/pose` (of type `geometry_msgs/PoseWithCovarianceStamped`)

`/kalman_filter/state` (of type `nav_msgs/Odometry`)
"""

# ROS
import rospy

# python libraries
import numpy as np
import math
from scipy.linalg import block_diag

# ROS messages and libraries
from nav_msgs.msg import Odometry
from apriltag_ros.msg import AprilTagDetectionArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from std_msgs.msg import Int8
import tf
from scipy.spatial.transform import Rotation as R

class TracjectoryModel(object):

    def __init__(self, namespace='trajectory_model'):
        """
        Initializing the parameters of the kalman filter
        """

        rospy.init_node("trajectory_model", anonymous=True)


        # broadcasting transform
        self.br = tf.TransformBroadcaster()

        # not publishing without receiving tag measurement
        self.received_trajectory_pose = False

        # Input Meassage
        self.trajectory_pose = PoseWithCovarianceStamped()

        # Output Message
        self.tool_state = Odometry()
        self.tool_pose = PoseWithCovarianceStamped()
        self.x_rot = Point()
        self.y_rot = Point()
        self.z_rot = Point()

        # interval between every message published
        self.OUTPUT_INTERVAL = rospy.get_param("~output_interval", default=1)

        # ROS publisher
        self.kf_state = rospy.Publisher("/kalman_filter/state", Odometry, queue_size=1)
        self.kf_pose = rospy.Publisher("/kalman_filter/pose", PoseWithCovarianceStamped, queue_size=1)
        self.x_vec = rospy.Publisher("/base/x", Point, queue_size=1)
        self.y_vec = rospy.Publisher("/base/y", Point, queue_size=1)
        self.z_vec = rospy.Publisher("/base/z", Point, queue_size=1)

        # Position initialization
        self.x = 0
        self.y = 0
        self.z = 0

        # Orientation Initialization
        self.roll = 0
        self.pitch = 0
        self.yaw = 0

        # Linear Velocity Initialization
        self.vel_x = 0
        self.vel_y = 0
        self.vel_z = 0

        # Angular Velocity Initialization
        self.vel_roll = np.pi/180.
        self.vel_pitch = np.pi/180.
        self.vel_yaw = np.pi/180.

        # Initializing sizes
        self.pose_size = 6
        self.twist_size = 6
        self.position_v_offset = 6

        # Tag Detected
        self.initial_R = False

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

        # Covariance matrix
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
        self.H = np.array( [[1, 0, 0, 0, 0, 0],
                            [0, 1, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0]])

        self.Z1 = np.array([self.x, self.y, self.z]).reshape(3,1)
        self.Z2 = np.array([self.roll, self.pitch, self.yaw]).reshape(3,1)

        # Orientation Vectors
        self.omega = np.array([self.vel_roll, self.vel_pitch, self.vel_yaw]).reshape(3,1)
        self.q = np.array([self.roll, self.pitch, self.yaw]).reshape(3,1)
        self.Q = np.array([self.roll, self.pitch, self.yaw, self.vel_roll, self.vel_pitch, self.vel_yaw]).reshape(6,1)
        self.QBar = np.zeros(self.Q.shape)
        self.current_time = rospy.Time.now()
        self.previous_time = rospy.Time.now()
        self.delta_t = 0.2

        # iterations
        self.count = 0
        self.publisherInterval = 3
        # timer for the output message
        rospy.Timer(
            rospy.Duration(self.OUTPUT_INTERVAL),
            lambda x: self.kalman()
        )

        # Subscribe to topics
        self.tag_subscriber = rospy.Subscriber("/linear/path", PoseWithCovarianceStamped, self.trajectory_receiver)

    def trajectory_receiver(self, msg):
        """
        Receives the Tag detections
        """
        if str(msg._type) == "geometry_msgs/PoseWithCovarianceStamped":
            self.received_trajectory_pose = True
            self.trajectory_pose = msg

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
            rospy.logerr('exception')
            x = X_bar
            P = P_bar

        return x,P

    def publishRotation(self, b1, b2, b3):
        """
        Publishing Rotation matrix directions
        """
        # Rotated X axis
        self.x_rot.x = b1[0][0]
        self.x_rot.y = b1[1][0]
        self.x_rot.z = b1[2][0]

        # Rotated Y axis
        self.y_rot.x = b2[0][0]
        self.y_rot.y = b2[1][0]
        self.y_rot.z = b2[2][0]

        # Rotated Z axis
        self.z_rot.x = b3[0][0]
        self.z_rot.y = b3[1][0]
        self.z_rot.z = b3[2][0]

        # publishing messages
        self.x_vec.publish(self.x_rot)
        self.y_vec.publish(self.y_rot)
        self.z_vec.publish(self.z_rot)

    def publishFilter(self, X, P):
        """
        Publishing the messages
        """
        self.tool_pose.header.frame_id = "camera"
        self.tool_state.header.frame_id = "camera"
        self.tool_state.child_frame_id = "odom"

        self.tool_pose.header.stamp = self.current_time
        self.tool_state.header.stamp = self.current_time

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
        
        for i in range(self.pose_size):
            
            for j in range(self.pose_size):

                self.tool_pose.pose.covariance[self.pose_size * i + j] = P[i][j]
                self.tool_state.pose.covariance[self.pose_size * i + j] = P[i][j]
        
        for i in range(self.twist_size):

            for j in range(self.twist_size):

                self.tool_state.twist.covariance[self.twist_size * i + j] = P[i + self.position_v_offset][j + self.position_v_offset]
        
        # publishing messages
        self.kf_pose.publish(self.tool_pose)
        self.kf_state.publish(self.tool_state)

    def kalman(self):
        """
        Implementing the Kalman Filter
        """
        self.current_time = rospy.Time.now()
        if not self.initial_R:
            quaternion = (self.trajectory_pose.pose.pose.orientation.x,
            self.trajectory_pose.pose.pose.orientation.y,
            self.trajectory_pose.pose.pose.orientation.z,
            self.trajectory_pose.pose.pose.orientation.w)

            # Once first tag detected we have seen the object once
            self.initial_R = True
            euler_measurement = tf.transformations.euler_from_quaternion(quaternion)
            # getting first transformation matrix from measurements
            R_0 = tf.transformations.euler_matrix(euler_measurement[0], euler_measurement[1], euler_measurement[2])

            self.R_t = R_0[:3, :3]

        # Motion Models
        self.F1 = np.array([[1., 0., 0., self.delta_t, 0., 0.],
                            [0., 1., 0., 0., self.delta_t, 0.],
                            [0., 0., 1., 0., 0., self.delta_t],
                            [0., 0., 0., self.R_t[0][0], self.R_t[0][1], self.R_t[0][2]],
                            [0., 0., 0., self.R_t[1][0], self.R_t[1][1], self.R_t[1][2]],
                            [0., 0., 0., self.R_t[2][0], self.R_t[2][1], self.R_t[2][2]]])

        self.F2 = np.array([[1., 0., 0., self.delta_t, 0., 0.],
                            [0., 1., 0., 0., self.delta_t, 0.],
                            [0., 0., 1., 0., 0., self.delta_t],
                            [0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0., 1.]])
        # Pose Predictions
        self.pBar, self.P_1Bar = self.predictPose(self.p, self.P_1, self.F1, self.Q1)
        self.QBar, self.P_2Bar = self.predictPose(self.Q, self.P_2, self.F2, self.Q2)
        
        # Update every 5 iterations
        if (self.count % self.publisherInterval == 0):
            self.Z1 = np.array([self.trajectory_pose.pose.pose.position.x,
                self.trajectory_pose.pose.pose.position.y,
                self.trajectory_pose.pose.pose.position.z]).reshape(3,1)

            quat = (self.trajectory_pose.pose.pose.orientation.x,
            self.trajectory_pose.pose.pose.orientation.y,
            self.trajectory_pose.pose.pose.orientation.z,
            self.trajectory_pose.pose.pose.orientation.w)

            euler =  tf.transformations.euler_from_quaternion(quat)
            self.Z2 = np.array([euler[0], euler[1], euler[2]]).reshape(3,1)

            self.Q, self.P_2 = self.updateState(self.QBar, self.P_2Bar, self.Z2, self.H, self.R2)
            self.p, self.P_1 = self.updateState(self.pBar, self.P_1Bar, self.Z1, self.H, self.R1)

        # Publish Predictions
        else:
            self.Q = self.QBar
            self.P_2 = self.P_2Bar
            self.p = self.pBar
            self.P_1 = self.P_1Bar

        # Rotation Matrix from the predicted/updated orientation
        self.R_t = tf.transformations.euler_matrix(self.Q[0][0], self.Q[1][0], self.Q[2][0]) [:3, :3]

        # Base vector of coordinate frame
        b1 = np.array([1., 0., 0.]).reshape(3,1)
        b2 = np.array([0., 1., 0.]).reshape(3,1)
        b3 = np.array([0., 0., 1.]).reshape(3,1)

        b1 = self.R_t.dot(b1)
        b2 = self.R_t.dot(b2)
        b3 = self.R_t.dot(b3)

        self.X = np.append(self.p, self.Q).reshape(12,1)
        self.P = block_diag(self.P_1, self.P_2)

        self.publishFilter(self.X, self.P)
        self.publishRotation(b1, b2, b3)

        self.previous_time = self.current_time
        self.count += 1

if __name__ == "__main__":
    kf = TracjectoryModel()
    rospy.spin()