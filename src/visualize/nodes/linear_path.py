#!/usr/bin/env python

"""
linear_path.py

Evaluates kalman filter by comparing its pose estimate to the linear path estimate

It publishes a message:
`/linear/path` (of type `geometry_msgs/PoseWithCovarianceStamped`)

`/linear/state` (of type `nav_msgs/Odometry`)

`/ground_truth/state` (of type `nav_msgs/Odometry`)
"""

import numpy as np
import rospy
import math
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
import tf
from itertools import cycle, chain

class LinearPath(object):
    def __init__(self, namespace='linear_path'):
        """
        Initializing the parameters of the linear path topic
        """
        rospy.init_node("linear_path", anonymous=True)

        # output message
        self.path_msg = PoseWithCovarianceStamped()
        self.state_msg = Odometry()
        self.ground_truth_msg = Odometry()

        # interval between every message published
        self.OUTPUT_INTERVAL = rospy.get_param("~output_interval", default=3)

         # output publisher
        self.path_output = rospy.Publisher("/linear/path", PoseWithCovarianceStamped, queue_size=0.1)
        self.path_state = rospy.Publisher("/linear/state", Odometry, queue_size=0.1)
        self.ground_truth = rospy.Publisher("/ground_truth/odom", Odometry, queue_size=0.1)

        self.i = 0
        self.phi = 0
        self.radius = 1

        self.position_noise = np.diag([0.0001, 0.0001, 0.0001]) ** 2
        self.orientation_noise = np.diag([0.001, 0.001, 0.001, 0.001]) ** 2
        self.velocity_noise = np.diag([0.001, 0.001, 0.001]) ** 2

        self.roll = self.phi
        self.pitch = self.phi
        self.yaw = self.phi

        self.current_time = rospy.Time.now()
        self.previous_time = rospy.Time.now()
        self.delta_t = 0.0

        self.v = 0.002
        self.omega = np.pi/180.

        self.current_position_x = self.i
        self.current_position_y = self.i
        self.current_position_z = self.i

        self.prev_position_x = self.i
        self.prev_position_y = self.i
        self.prev_position_z = self.i


        self.br = tf.TransformBroadcaster()
        # timer for the output message
        rospy.Timer(
            rospy.Duration(self.OUTPUT_INTERVAL),
            lambda x: self.path_publisher()
        )

    def path_publisher(self):
        """
        Publishes a path along circle in xy plane
        """
        self.current_time = rospy.Time.now()
        self.delta_t = (self.current_time - self.previous_time).to_sec()

        self.path_msg.header.stamp.secs = rospy.Time.now().to_sec()
        self.state_msg.header.stamp.secs = rospy.Time.now().to_sec()
        self.ground_truth_msg.header.stamp.secs = rospy.Time.now().to_sec()

        self.path_msg.header.frame_id = "map"
        
        self.roll = 0.
        self.pitch = 0.
        self.yaw = self.phi

        quat = tf.transformations.quaternion_from_euler(self.roll, self.pitch, self.yaw)

        self.ground_truth_msg.pose.pose.position.x += self.radius * np.cos(self.phi) * self.v
        self.ground_truth_msg.pose.pose.position.y += self.radius * np.sin(self.phi) * self.v
        self.ground_truth_msg.pose.pose.position.z = 0

        self.ground_truth_msg.pose.pose.orientation.x = quat[0]
        self.ground_truth_msg.pose.pose.orientation.y = quat[1]
        self.ground_truth_msg.pose.pose.orientation.z = quat[2]
        self.ground_truth_msg.pose.pose.orientation.w = quat[3]

        self.ground_truth_msg.twist.twist.linear.x = self.v
        self.ground_truth_msg.twist.twist.linear.y = 0.
        self.ground_truth_msg.twist.twist.linear.z = 0.
        
        self.path_msg.pose.pose.position.x = self.ground_truth_msg.pose.pose.position.x  + self.position_noise.dot(np.random.randn(3,1)) [0][0]
        self.path_msg.pose.pose.position.y = self.ground_truth_msg.pose.pose.position.y + self.position_noise.dot(np.random.randn(3,1)) [1][0]
        self.path_msg.pose.pose.position.z = self.ground_truth_msg.pose.pose.position.z + self.position_noise.dot(np.random.randn(3,1)) [2][0]

        self.path_msg.pose.pose.orientation.x = quat[0] + self.orientation_noise.dot(np.random.randn(4,1)) [0][0]
        self.path_msg.pose.pose.orientation.y = quat[1] + self.orientation_noise.dot(np.random.randn(4,1)) [1][0]
        self.path_msg.pose.pose.orientation.z = quat[2] + self.orientation_noise.dot(np.random.randn(4,1)) [2][0]
        self.path_msg.pose.pose.orientation.w = quat[3] + self.orientation_noise.dot(np.random.randn(4,1)) [3][0]

        self.state_msg.pose.pose.position.x = self.ground_truth_msg.pose.pose.position.x  + self.position_noise.dot(np.random.randn(3,1)) [0][0]
        self.state_msg.pose.pose.position.y = self.ground_truth_msg.pose.pose.position.y  + self.position_noise.dot(np.random.randn(3,1)) [1][0]
        self.state_msg.pose.pose.position.z = self.ground_truth_msg.pose.pose.position.z  + self.position_noise.dot(np.random.randn(3,1)) [2][0]

        self.state_msg.pose.pose.orientation.x = quat[0] + self.orientation_noise.dot(np.random.randn(4,1)) [0][0]
        self.state_msg.pose.pose.orientation.y = quat[1] + self.orientation_noise.dot(np.random.randn(4,1)) [1][0]
        self.state_msg.pose.pose.orientation.z = quat[2] + self.orientation_noise.dot(np.random.randn(4,1)) [2][0]
        self.state_msg.pose.pose.orientation.w = quat[3] + self.orientation_noise.dot(np.random.randn(4,1)) [3][0]

        self.br.sendTransform((self.i, self.i, 0),
                    tf.transformations.quaternion_from_euler(self.roll, self.pitch, self.yaw),
                    rospy.Time.now(),
                    "map",
                    "odom")

        self.current_position_x = self.path_msg.pose.pose.position.x
        self.current_position_y = self.path_msg.pose.pose.position.y
        self.current_position_z = self.path_msg.pose.pose.position.z
        
        self.state_msg.twist.twist.linear.x = self.v
        self.state_msg.twist.twist.linear.y = 0.
        self.state_msg.twist.twist.linear.z = 0.

        self.state_msg.twist.twist.angular.x = 0.
        self.state_msg.twist.twist.angular.y = 0.
        self.state_msg.twist.twist.angular.z = self.omega

        self.ground_truth_msg.twist.twist.angular.x = 0.
        self.ground_truth_msg.twist.twist.angular.y = 0.
        self.ground_truth_msg.twist.twist.angular.z = self.omega

        self.prev_position_x = self.current_position_x
        self.prev_position_y = self.current_position_y
        self.prev_position_z = self.current_position_z

        self.path_output.publish(self.path_msg)
        self.path_state.publish(self.state_msg)
        self.ground_truth.publish(self.ground_truth_msg)
        if (np.sin(self.phi) == 0.5):
            rospy.logerr('30 degrees')

        elif (np.sin(self.phi) == 1.0):
            rospy.logerr('quater complete')

        elif (np.cos(self.phi) == -1):
            rospy.logerr('semi circle')

        elif (self.phi >= 2*np.pi and np.cos(self.phi) == 1.0):
            rospy.logerr('complete circle')

        elif (np.cos(self.phi) == 1.0):
            rospy.logerr('starting circle')
        rospy.logerr(np.sin(self.phi))

        self.phi += self.omega * 0.2


    def normalizeAngle(self, phi):

        while (phi > np.pi):
            phi = phi - 2 * np.pi

        while (phi < - np.pi):
            phi = phi + 2 * np.pi

        return phi

if __name__ == "__main__":
    linear = LinearPath()
    rospy.spin()