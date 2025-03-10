#!/usr/bin/env python3
"""
ROS2 wrapper code taken from:
https://github.com/JMU-ROBOTICS-VIVA/ros2_aruco/tree/main

This node locates Aruco AR markers in images and publishes their ids and poses.

Subscriptions:
   /camera/image_raw (sensor_msgs.msg.Image)
   /camera/camera_info (sensor_msgs.msg.CameraInfo)

Published Topics:
    /aruco_poses (geometry_msgs.msg.PoseArray)
       Pose of all detected markers (suitable for rviz visualization)

    /aruco_markers (aruco_interfaces.msg.ArucoMarkers)
       Provides an array of all poses along with the corresponding
       marker ids.

    /aruco_image (sensor_msgs.msg.Image)
       Annotated image with marker locations and ids, with markers drawn on it

Parameters:
    marker_size - size of the markers in meters (default .065)
    aruco_dictionary_id - dictionary that was used to generate markers (default DICT_5X5_250)
    image_topic - image topic to subscribe to (default /camera/color/image_raw)
    camera_info_topic - camera info topic to subscribe to (default /camera/camera_info)
    camera_frame - camera optical frame to use (default "camera_depth_optical_frame")
    detected_markers_topic - topic to publish detected markers (default /aruco_markers)
    markers_visualization_topic - topic to publish markers visualization (default /aruco_poses)
    output_image_topic - topic to publish annotated image (default /aruco_image)

Author: Simone Giampà
Version: 2024-01-29

"""

# ROS1 imports
import rospy
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from aruco_interfaces.msg import ArucoMarkers

# Python imports
import numpy as np
import cv2

# Local imports for custom defined functions
from aruco_pose_estimation.utils import ARUCO_DICT
from aruco_pose_estimation.pose_estimation import pose_estimation


class ArucoNode():
    def __init__(self):
        self.initialize_parameters()

        # Make sure we have a valid dictionary id:
        try:
            dictionary_id = cv2.aruco.__getattribute__(self.dictionary_id_name)
            # check if the dictionary_id is a valid dictionary inside ARUCO_DICT values
            if dictionary_id not in ARUCO_DICT.values():
                raise AttributeError
        except AttributeError:
            rospy.logerr(
                "bad aruco_dictionary_id: {}".format(self.dictionary_id_name)
            )
            options = "\n".join([s for s in ARUCO_DICT])
            rospy.logerr("valid options: {}".format(options))

        # Set up subscriptions to the camera info and camera image topics

        # camera info topic for the camera calibration parameters
        self.info_sub = rospy.Subscriber(self.info_topic, CameraInfo, self.info_callback, queue_size=10)

        # select the type of input to use for the pose estimation
        if (bool(self.use_depth_input)):
            # use both rgb and depth image topics for the pose estimation

            # create a message filter to synchronize the image and depth image topics
            self.image_sub = message_filters.Subscriber(self.image_topic, Image)
            self.depth_image_sub = message_filters.Subscriber(self.depth_image_topic, Image)

            # create synchronizer between the 2 topics using message filters and approximate time policy
            # slop is the maximum time difference between messages that are considered synchronized
            self.synchronizer = message_filters.ApproximateTimeSynchronizer(
                [self.image_sub, self.depth_image_sub], queue_size=10, slop=0.05
            )
            self.synchronizer.registerCallback(self.rgb_depth_sync_callback)
        else:
            # rely only on the rgb image topic for the pose estimation

            # create a subscription to the image topic
            self.image_sub = rospy.Subscriber(
                self.image_topic, Image, self.image_callback, queue_size=10
            )

        # Set up publishers
        self.poses_pub = rospy. Publisher(self.markers_visualization_topic, PoseArray, queue_size=10)
        self.markers_pub = rospy.Publisher(self.detected_markers_topic, ArucoMarkers, queue_size=10)
        self.image_pub = rospy.Publisher(self.output_image_topic, Image, queue_size=10)

        # Set up fields for camera parameters
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        # code for updated version of cv2 (4.7.0)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)

        # old code version
        # self.aruco_dictionary = cv2.aruco.Dictionary_get(dictionary_id)
        # self.aruco_parameters = cv2.aruco.DetectorParameters_create()

        self.bridge = CvBridge()

    def info_callback(self, info_msg):
        self.info_msg = info_msg
        # get the intrinsic matrix and distortion coefficients from the camera info
        self.intrinsic_mat = np.reshape(np.array(self.info_msg.k), (3, 3))
        self.distortion = np.array(self.info_msg.d)

        rospy.loginfo("Camera info received.")
        rospy.loginfo("Intrinsic matrix: {}".format(self.intrinsic_mat))
        rospy.loginfo("Distortion coefficients: {}".format(self.distortion))
        rospy.loginfo("Camera frame: {}x{}".format(self.info_msg.width, self.info_msg.height))

        # Assume that camera parameters will remain the same...
        self.info_sub = None

    def image_callback(self, img_msg: Image):
        if self.info_msg is None:
            rospy.logwarn("No camera info has been received!")
            return

        # convert the image messages to cv2 format
        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="rgb8")

        # create the ArucoMarkers and PoseArray messages
        markers = ArucoMarkers()
        pose_array = PoseArray()

        # Set the frame id and timestamp for the markers and pose array
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        """
        # OVERRIDE: use calibrated intrinsic matrix and distortion coefficients
        self.intrinsic_mat = np.reshape([615.95431, 0., 325.26983,
                                         0., 617.92586, 257.57722,
                                         0., 0., 1.], (3, 3))
        self.distortion = np.array([0.142588, -0.311967, 0.003950, -0.006346, 0.000000])
        """
        
        # call the pose estimation function
        frame, pose_array, markers = pose_estimation(rgb_frame=cv_image, depth_frame=None,
                                                     aruco_detector=self.aruco_detector,
                                                     marker_size=self.marker_size, matrix_coefficients=self.intrinsic_mat,
                                                     distortion_coefficients=self.distortion, pose_array=pose_array, markers=markers)

        # if some markers are detected
        if len(markers.marker_ids) > 0:
            # Publish the results with the poses and markes positions
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        # publish the image frame with computed markers positions over the image
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

    def depth_image_callback(self, depth_msg: Image):
        if self.info_msg is None:
            rospy.logwarn("No camera info has been received!")
            return

    def rgb_depth_sync_callback(self, rgb_msg: Image, depth_msg: Image):

        # convert the image messages to cv2 format
        cv_depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        cv_image = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="rgb8")

        # create the ArucoMarkers and PoseArray messages
        markers = ArucoMarkers()
        pose_array = PoseArray()

        # Set the frame id and timestamp for the markers and pose array
        if self.camera_frame == "":
            markers.header.frame_id = self.info_msg.header.frame_id
            pose_array.header.frame_id = self.info_msg.header.frame_id
        else:
            markers.header.frame_id = self.camera_frame
            pose_array.header.frame_id = self.camera_frame

        markers.header.stamp = rgb_msg.header.stamp
        pose_array.header.stamp = rgb_msg.header.stamp

        # call the pose estimation function
        frame, pose_array, markers = pose_estimation(rgb_frame=cv_image, depth_frame=cv_depth_image,
                                                     aruco_detector=self.aruco_detector,
                                                     marker_size=self.marker_size, matrix_coefficients=self.intrinsic_mat,
                                                     distortion_coefficients=self.distortion, pose_array=pose_array, markers=markers)

        # if some markers are detected
        if len(markers.marker_ids) > 0:
            # Publish the results with the poses and markes positions
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers)

        # publish the image frame with computed markers positions over the image
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(frame, "rgb8"))

    def initialize_parameters(self):
        # Declare and read parameters from aruco_params.yaml
        self.marker_size = rospy.get_param("~marker_size", 0.0625)
        rospy.loginfo(f"Marker size: {self.marker_size}")
        self.dictionary_id_name = rospy.get_param("~aruco_dictionary_id", "DICT_5X5_250")
        rospy.loginfo(f"Marker type: {self.dictionary_id_name}")
        self.use_depth_input = rospy.get_param("~use_depth_input", True)
        rospy.loginfo(f"Use depth input: {self.use_depth_input}")
        self.image_topic = rospy.get_param("~image_topic", "/camera/image_raw")
        rospy.loginfo(f"Input image topic: {self.image_topic}")
        self.depth_image_topic = rospy.get_param("~depth_image_topic", "/camera/depth/image_raw")
        rospy.loginfo(f"Input depth image topic: {self.depth_image_topic}")
        self.info_topic = rospy.get_param("~camera_info_topic", "/camera/camera_info")
        rospy.loginfo(f"Image camera info topic: {self.info_topic}")
        self.camera_frame = rospy.get_param("~camera_frame", "")
        rospy.loginfo(f"Camera frame: {self.camera_frame}")
        self.detected_markers_topic = rospy.get_param("~detected_markers_topic", "/aruco_markers")
        self.markers_visualization_topic = rospy.get_param("~markers_visualization_topic", "/aruco_poses")
        self.output_image_topic = rospy.get_param("~output_image_topic", "/aruco_image")


def main():
    rospy.init_node('aruco_node', anonymous=True)
    aruco_node = ArucoNode()
    rospy.spin()
    rospy.shutdown()


if __name__ == "__main__":
    main()
