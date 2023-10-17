#!/usr/bin/env python3

# Author: Connor McGuile
# Latest author: Adarsh Jagan Sathyamoorthy
# Feel free to use in any way.

# A custom Dynamic Window Approach implementation for use with Turtlebot.
# Obstacles are registered by a front-mounted laser and stored in a set.
# If, for testing purposes or otherwise, you do not want the laser to be used,
# disable the laserscan subscriber and create your own obstacle set in main(),
# before beginning the loop. If you do not want obstacles, create an empty set.
# Implentation based off Fox et al.'s paper, The Dynamic Window Approach to
# Collision Avoidance (1997).

import rospy
import math
import numpy as np
from numpy.lib.stride_tricks import as_strided
from std_msgs.msg import Float32, Float32MultiArray
from cv_bridge import CvBridge, CvBridgeError

from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import OccupancyGrid, Odometry
import sensor_msgs.msg
from sensor_msgs.msg import LaserScan, CompressedImage
from tf.transformations import euler_from_quaternion
import time
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
import sys
import csv

# Headers for local costmap subscriber
from matplotlib import pyplot as plt
from matplotlib.path import Path
from PIL import Image

import sys
# OpenCV
# sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
# sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages')


class Config():
    # simulation parameters

    def __init__(self):
        
        # Robot parameters
        self.max_speed = 0.6     # [m/s]
        self.min_speed = 0.0     # [m/s]
        self.max_yawrate = 0.6   # [rad/s]
        self.max_accel = 1       # [m/ss]
        self.max_dyawrate = 3.2  # [rad/ss]
        
        self.v_reso = 0.30 #0.20              # [m/s]
        self.yawrate_reso = 0.3  # [rad/s]
        
        self.dt = 0.5  # [s]
        self.predict_time = 3.5 #2.0 #1.5  # [s]
        
        # 1===
        self.to_goal_cost_gain = 5.0       # lower = detour
        self.obs_cost_gain = 1.0           # Used with costmap
        self.speed_cost_gain = 0.1   # 0.1   # lower = faster
        # self.obs_cost_gain = 3.2            # lower z= fearless (used with scan)
        
        self.robot_radius = 0.6  # [m]
        self.x = 0.0
        self.y = 0.0
        self.v_x = 0.0
        self.v_y = 0.0
        self.w_z = 0.0
        self.goalX = 0.0006
        self.goalY = 0.0006
        self.th = 0.0
        self.r = rospy.Rate(20)

        self.x_odom_prev = 0.0
        self.y_odom_prev = 0.0
        self.theta_prev = 0.0
        self.x_odom_curr = 0.0
        self.y_odom_curr = 0.0
        self.theta_curr = 0.0

        self.collision_threshold = 0.3 # [m]

        # confidence = input("Enter Confidence Threshold : ")
        # self.conf_thresh = float(confidence)
        self.conf_thresh = 0.80

        # DWA output
        self.min_u = []

        self.stuck_status = False
        self.stuck_count = 0
        self.pursuing_safe_loc = False
        self.okay_locations = []
        self.stuck_locations = []


        # Costmap
        self.scale_percent = 300 # percent of original size
        self.costmap_shape = (200, 200)
        self.costmap_resolution = 0.05
        print("MIM Started!")
        # self.costmap_baselink_high = np.zeros(self.costmap_shape, dtype=np.uint8)
        # self.costmap_baselink_mid = np.zeros(self.costmap_shape, dtype=np.uint8)
        # self.costmap_baselink_low = np.zeros(self.costmap_shape, dtype=np.uint8)
        
        self.intensitymap_low = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.intensitymap_mid = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.intensitymap_high = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.intensitymap_low_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.intensitymap_mid_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.intensitymap_diff_inflated = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.kernel_size = (5, 5) # for glass #(11, 11) for everything else
        
        self.roi_shape = (80, 80)
        self.roi_prev = np.zeros(self.roi_shape, dtype=np.uint8)
        self.roi_curr = np.zeros(self.roi_shape, dtype=np.uint8)
        self.diff_curr = np.zeros(self.costmap_shape, dtype=np.uint8)
        self.flag = 0
        self.counter = 0
        
        self.intmap_rgb = cv2.cvtColor(self.intensitymap_low, cv2.COLOR_GRAY2RGB)
        self.obs_low_mid_high = np.argwhere(self.intensitymap_low > 150) # should be null set

        # For cost map clearing
        self.height_thresh = 75 #150
        self.intensity_thresh = 180
        self.alpha = 0.35
        
        # For on-field visualization
        self.plan_map_pub = rospy.Publisher("/planning_costmap", sensor_msgs.msg.Image, queue_size=10)
        self.viz_pub = rospy.Publisher("/viz_costmap", sensor_msgs.msg.Image, queue_size=10) 
        self.intensity_mid_pub = rospy.Publisher("/int_mid", sensor_msgs.msg.Image, queue_size=10)
        self.intensity_low_pub = rospy.Publisher("/int_low", sensor_msgs.msg.Image, queue_size=10)
        self.intensity_high_pub = rospy.Publisher("/int_high", sensor_msgs.msg.Image, queue_size=10) 
        self.intensity_diff_pub = rospy.Publisher("/int_diff", sensor_msgs.msg.Image, queue_size=10) 
        self.br = CvBridge()


    # Callback for Odometry
    def assignOdomCoords(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        rot_q = msg.pose.pose.orientation
        (roll,pitch,theta) = euler_from_quaternion ([rot_q.x,rot_q.y,rot_q.z,rot_q.w])
        # (roll,pitch,theta) = euler_from_quaternion ([rot_q.z, -rot_q.x, -rot_q.y, rot_q.w]) # used when lego-loam is used
        
        self.th = theta
        # print("Theta of body wrt odom:", self.th/0.0174533)

        # Get robot's current velocities
        self.v_x = msg.twist.twist.linear.x
        self.v_y = msg.twist.twist.linear.y
        self.w_z = msg.twist.twist.angular.z 
        # print("Robot's current velocities", [self.v_x, self.w_z])



    # Callback for goal from POZYX
    def target_callback(self, data):
        print("---------------Inside Goal Callback------------------------")

        radius = data.linear.x # this will be r
        theta = data.linear.y * 0.0174533 # this will be theta
        print("r and theta:",data.linear.x, data.linear.y)
        
        # Goal wrt robot frame        
        goalX_rob = radius * math.cos(theta)
        goalY_rob = radius * math.sin(theta)

        # Goal wrt odom frame (from where robot started)
        self.goalX =  self.x + goalX_rob*math.cos(self.th) - goalY_rob*math.sin(self.th)
        self.goalY = self.y + goalX_rob*math.sin(self.th) + goalY_rob*math.cos(self.th)
        
        # print("Self odom:",self.x, self.y)
        # print("Goals wrt odom frame:", self.goalX, self.goalY)

        # If goal is published as x, y coordinates wrt odom uncomment this
        # self.goalX = data.linear.x
        # self.goalY = data.linear.y

    

    def intensity_map_low_cb(self, data_low):

        # Obtain lower intensity map
        # data_low = rospy.wait_for_message('intensity_map_low', OccupancyGrid)
        int_low = np.reshape(data_low.data, (-1, int(math.sqrt(len(data_low.data)))))
        int_low = np.reshape(data_low.data, (int(math.sqrt(len(data_low.data))), -1))
        int_low = np.rot90(np.fliplr(int_low), 1, (1, 0))

        im_low_image = Image.fromarray(np.uint8(int_low))
        yaw_deg = 0 # Now cost map published wrt baselink
        im_low_pil = im_low_image.rotate(-yaw_deg)
        self.intensitymap_low = np.array(im_low_pil)
        self.intensitymap_low = np.rot90(np.uint8(self.intensitymap_low),2)

        # Inflation
        kernel = np.ones(self.kernel_size, np.uint8)
        dilated_low = cv2.dilate(self.intensitymap_low, kernel, iterations =1)

        self.intensitymap_low_inflated = np.array(dilated_low)
        self.intensity_low_pub.publish(self.br.cv2_to_imgmsg(self.intensitymap_low_inflated, encoding="mono8"))


    def intensity_map_high_cb(self, data_high):

        # Obtain higher intensity map
        int_high = np.reshape(data_high.data, (-1, int(math.sqrt(len(data_high.data)))))
        int_high = np.reshape(data_high.data, (int(math.sqrt(len(data_high.data))), -1))
        int_high = np.rot90(np.fliplr(int_high), 1, (1, 0))

        im_high_image = Image.fromarray(np.uint8(int_high))
        yaw_deg = 0 # Now cost map published wrt baselink
        im_high_pil = im_high_image.rotate(-yaw_deg)
        self.intensitymap_high = np.array(im_high_pil)
        self.intensitymap_high = np.rot90(np.uint8(self.intensitymap_high),2)

        # Inflation
        kernel = np.ones(self.kernel_size, np.uint8)
        dilated_high = cv2.dilate(self.intensitymap_high, kernel, iterations =1)

        self.intensitymap_high_inflated = np.array(dilated_high)
        self.intensity_high_pub.publish(self.br.cv2_to_imgmsg(self.intensitymap_high_inflated, encoding="mono8"))



    #Callback for MID intensity map
    def intensity_map_mid_cb(self, data):
        # print("Inside Mid intensity Map Callback!")

        # Mid Intensity Map
        int_mid = np.reshape(data.data, (-1, int(math.sqrt(len(data.data)))))
        int_mid = np.reshape(data.data, (int(math.sqrt(len(data.data))), -1))
        int_mid = np.rot90(np.fliplr(int_mid), 1, (1, 0))

        im_mid_image = Image.fromarray(np.uint8(int_mid))
        yaw_deg = 0 # Now cost map published wrt baselink
        im_mid_pil = im_mid_image.rotate(-yaw_deg)
        self.intensitymap_mid = np.array(im_mid_pil)
        self.intensitymap_mid = np.rot90(np.uint8(self.intensitymap_mid),2) # Rotating by 180 deg

        # Inflation
        kernel1 = np.ones(self.kernel_size,np.uint8)
        kernel2 = np.tril(kernel1, 3) # Bottom left triangle will be 1s
        kernel2 = np.triu(kernel2, -3) # Top right triangle will be 1s
        
        kernel1[:, 6:11] = 0 # inflates vertically
        
        #kernel = np.transpose(kernel2)
        #kernel = np.rot90(kernel2)

        kernel = kernel1

        #print(kernel)

        dilated_mid = cv2.dilate(self.intensitymap_mid, kernel, iterations =1)
        
        self.intensitymap_mid_inflated = np.array(dilated_mid)

        # Uncomment to use mid intensity map with markings 
        # self.intmap_rgb = cv2.cvtColor(self.intensitymap_mid_inflated, cv2.COLOR_GRAY2RGB)
        # # Robot location on costmap
        # rob_x = int(self.intmap_rgb.shape[0]/2)
        # rob_y = int(self.intmap_rgb.shape[1]/2)

        # # Visualization
        # # Mark the robot on intensitymap 
        # self.intmap_rgb = cv2.circle(self.intmap_rgb, (rob_x, rob_y), 4, (255, 0, 255), -1)
        

        #publish inflated intensity map as an image
        # self.intensity_mid_pub.publish(self.br.cv2_to_imgmsg(self.intensitymap_mid_inflated, encoding="mono8"))

        self.glass_detect()


    def glass_detect(self):
        # print("Inside Glass detection!")
        center_x = self.costmap_shape[0]/2
        center_y = self.costmap_shape[1]/2
        roi_side = self.roi_shape[0]/2
        
        if(self.flag == 0):
            self.x_odom_prev = self.x
            self.y_odom_prev = self.y
            self.theta_prev = self.th
            diff1 = self.intensitymap_mid - self.intensitymap_low
            self.diff_curr = diff1
            self.roi_prev = self.diff_curr[int(center_x-roi_side):int(center_x+roi_side), int(center_y-roi_side):int(center_y+roi_side)]

            self.flag = 1

        elif(self.flag == 1 and self.counter % 3 == 0 and (abs(self.v_x) > 0.1 or abs(self.v_y) > 0.1)):
        # elif(self.flag == 1 and self.counter % 3 == 0):
            t1 = time.time()
            self.counter = self.counter + 1

            self.x_odom_curr = self.x
            self.y_odom_curr = self.y
            self.theta_curr = self.th

            # Uninflated difference
            diff1 = self.intensitymap_mid - self.intensitymap_low
            # diff2 = (self.intensitymap_mid - self.intensitymap_high)

            self.diff_curr = diff1
            self.roi_curr = self.diff_curr[int(center_x-roi_side):int(center_x+roi_side), int(center_y-roi_side):int(center_y+roi_side)]
        
            # Transform previous ROI to current difference's frame
            x_rob_prev, y_rob_prev = odom_to_robot(self, np.array([self.x_odom_prev]), np.array([self.y_odom_prev]))
            cm_col, cm_row = robot_to_costmap(self, x_rob_prev, y_rob_prev)
            theta_diff = (self.theta_curr - self.theta_prev)*180/math.pi
            # print(cm_col, cm_row, theta_diff) 

            # row_tran = center_x - int(cm_row)
            # col_tran = center_y - int(cm_col)
            rotation = theta_diff
            # print(self.roi_prev.shape, int(cm_col), int(cm_row), rotation)

            roi_prev_image = Image.fromarray(np.uint8(self.roi_prev))
            roi_prev_pil = roi_prev_image.rotate(-rotation)
            transformed_roi = np.array(roi_prev_pil)
            transformed_roi[transformed_roi > 0] = 254
            

            self.diff_curr[int(int(cm_row)-roi_side):int(int(cm_row)+roi_side), int(int(cm_col-1)-roi_side):int(int(cm_col-1)+roi_side)] = self.diff_curr[int(int(cm_row)-roi_side):int(int(cm_row)+roi_side), int(int(cm_col-1)-roi_side):int(int(cm_col-1)+roi_side)] + transformed_roi
            # self.intensitymap_mid_inflated[int(int(cm_row)-roi_side):int(int(cm_row)+roi_side), int(int(cm_col-1)-roi_side):int(int(cm_col-1)+roi_side)] = self.intensitymap_mid_inflated[int(int(cm_row)-roi_side):int(int(cm_row)+roi_side), int(int(cm_col-1)-roi_side):int(int(cm_col-1)+roi_side)] + transformed_roi
            print("Max and min values are", np.max(self.diff_curr), np.min(self.diff_curr))
            
            diff = self.diff_curr 
            
            # Inflation
            kernel1 = np.ones(self.kernel_size,np.uint8)
            # kernel1[:, 6:11] = 0 # inflates vertically

            # kernel2 = np.tril(kernel1, 3) # Bottom left triangle will be 1s
            # kernel2 = np.triu(kernel2, -3) # Top right triangle will be 1s
            
            #kernel = np.transpose(kernel2)
            #kernel = np.rot90(kernel2)
            #kernel = kernel2

            kernel = kernel1

            print(kernel)
            
            dilated_diff = cv2.dilate(diff, kernel, iterations =1)
            
            self.intensitymap_diff_inflated = np.array(dilated_diff)
            self.intmap_rgb = cv2.cvtColor(self.intensitymap_diff_inflated, cv2.COLOR_GRAY2RGB)

            # Robot location on costmap
            rob_x = int(self.intmap_rgb.shape[0]/2)
            rob_y = int(self.intmap_rgb.shape[1]/2)

            # Visualization
            # Mark the robot on costmap 
            self.intmap_rgb = cv2.circle(self.intmap_rgb, (rob_x, rob_y), 3, (255, 0, 255), -1)

            # Publish
            # self.intensity_combo_pub.publish(self.br.cv2_to_imgmsg(intensitymap_diff_inflated, encoding="mono8"))
            # self.intensity_diff_pub.publish(self.br.cv2_to_imgmsg(self.intmap_rgb, encoding="bgr8"))

            self.x_odom_prev = self.x_odom_curr
            self.y_odom_prev = self.y_odom_curr
            self.theta_prev = self.theta_curr
            self.roi_prev = self.roi_curr

            t2 = time.time()
            print("Inference time", t2 - t1)


        else:
            self.counter = self.counter + 1




    def tall_obstacle_marker(self, rgb_image, centers):
        # Marking centers red = (0, 0, 255), or orange = (0, 150, 255)
        rgb_image[centers[:, 0], centers[:, 1], 0] = 0
        rgb_image[centers[:, 0], centers[:, 1], 1] = 0
        rgb_image[centers[:, 0], centers[:, 1], 2] = 255
        return rgb_image



class Obstacles():
    def __init__(self):
        # Set of coordinates of obstacles in view
        self.obst = set()
        self.collision_status = False

    # Custom range implementation to loop over LaserScan degrees with
    # a step and include the final degree
    def myRange(self,start,end,step):
        i = start
        while i < end:
            yield i
            i += step
        yield end


    # Callback for LaserScan
    def assignObs(self, msg, config):

        deg = len(msg.ranges)   # Number of degrees - varies in Sim vs real world
        self.obst = set()   # reset the obstacle set to only keep visible objects

        maxAngle = 360
        scanSkip = 1
        anglePerSlot = (float(maxAngle) / deg) * scanSkip
        angleCount = 0
        angleValuePos = 0
        angleValueNeg = 0
        self.collision_status = False
        for angle in self.myRange(0,deg-1,scanSkip):
            distance = msg.ranges[angle]

            if (distance < 0.05) and (not self.collision_status):
                self.collision_status = True
                # print("Collided")
                reached = False
                reset_robot(reached)

            if(angleCount < (deg / (2*scanSkip))):
                # print("In negative angle zone")
                angleValueNeg += (anglePerSlot)
                scanTheta = (angleValueNeg - 180) * math.pi/180.0


            elif(angleCount>(deg / (2*scanSkip))):
                # print("In positive angle zone")
                angleValuePos += anglePerSlot
                scanTheta = angleValuePos * math.pi/180.0
            # only record obstacles that are within 4 metres away

            else:
                scanTheta = 0

            angleCount += 1

            if (distance < 4):
                # angle of obstacle wrt robot
                # angle/2.844 is to normalise the 512 degrees in real world
                # for simulation in Gazebo, use angle/4.0
                # laser from 0 to 180


                objTheta =  scanTheta + config.th

                # round coords to nearest 0.125m
                obsX = round((config.x + (distance * math.cos(abs(objTheta))))*8)/8
                # determine direction of Y coord
                
                if (objTheta < 0):
                    obsY = round((config.y - (distance * math.sin(abs(objTheta))))*8)/8
                else:
                    obsY = round((config.y + (distance * math.sin(abs(objTheta))))*8)/8


                # add coords to set so as to only take unique obstacles
                self.obst.add((obsX,obsY))
                


# Model to determine the expected position of the robot after moving along trajectory
def motion(x, u, dt):
    # motion model
    # x = [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x[2] += u[1] * dt
    x[0] += u[0] * math.cos(x[2]) * dt
    x[1] += u[0] * math.sin(x[2]) * dt

    x[3] = u[0]
    x[4] = u[1]

    return x


# Determine the dynamic window from robot configurations
def calc_dynamic_window(x, config):

    # Dynamic window from robot specification
    Vs = [config.min_speed, config.max_speed,
          -config.max_yawrate, config.max_yawrate]

    # Dynamic window from motion model
    Vd = [x[3] - config.max_accel * config.dt,
          x[3] + config.max_accel * config.dt,
          x[4] - config.max_dyawrate * config.dt,
          x[4] + config.max_dyawrate * config.dt]

    #  [vmin, vmax, yawrate min, yawrate max]
    dw = [max(Vs[0], Vd[0]), min(Vs[1], Vd[1]),
          max(Vs[2], Vd[2]), min(Vs[3], Vd[3])]

    # print("Dynamic Window: ", dw)
    return dw


# Calculate a trajectory sampled across a prediction time
def calc_trajectory(xinit, v, y, config):

    x = np.array(xinit)
    traj = np.array(x)  # many motion models stored per trajectory
    time = 0
    while time <= config.predict_time:
        # store each motion model along a trajectory
        x = motion(x, [v, y], config.dt)
        traj = np.vstack((traj, x))
        time += config.dt # next sample

    return traj


# 3===
# Calculate trajectory, costings, and return velocities to apply to robot
def calc_final_input(x, u, dw, config, ob):

    xinit = x[:]
    min_cost = 10000.0
    config.min_u = u
    config.min_u[0] = 0.0
    
    yellow = (0, 255, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    orange = (0, 150, 255)

    count = 0
    # evaluate all trajectory with sampled input in dynamic window
    for v in np.arange(dw[0], dw[1] + config.v_reso/2, config.v_reso):
        for w in np.arange(dw[2], dw[3] + config.yawrate_reso/2, config.yawrate_reso):
            count = count + 1 
            
            traj = calc_trajectory(xinit, v, w, config)

            # calc costs with weighted gains
            to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(traj, config)
            speed_cost = config.speed_cost_gain * (config.max_speed - traj[-1, 3]) # end v should be as close to max_speed to have low cost
            obs_cost = config.obs_cost_gain * calc_obs_cost(traj, config)

            final_cost = to_goal_cost + obs_cost + speed_cost
            
            # print(count, "v,w = %.2f %.2f"% (v, w))
            # print("Goal cost = %.2f"% to_goal_cost, "veg_cost = %.2f"% veg_cost, "final_cost = %.2f"% final_cost)
            # print("Goal cost = %.2f"% to_goal_cost, "speed_cost = %.2f"% speed_cost, "veg_cost = %.2f"% veg_cost, "final_cost = %.2f"% final_cost)

            config.intmap_rgb = draw_traj(config, traj, yellow)

            # search minimum trajectory
            if min_cost >= final_cost:
                min_cost = final_cost
                config.min_u = [v, w]

    # print("Robot's current velocities", [config.v_x, config.w_z])
    # traj = calc_trajectory(xinit, config.v_x, config.w_z, config) # This leads to buggy visualization

    traj = calc_trajectory(xinit, config.min_u[0], config.min_u[1], config)
    to_goal_cost = config.to_goal_cost_gain * calc_to_goal_cost(traj, config)
    obs_cost_min = config.obs_cost_gain * calc_obs_cost(traj, config)
    print("min_u = %.2f %.2f"% (config.min_u[0], config.min_u[1]), "Goal cost = %.2f"% to_goal_cost, "Obs cost = %.2f"% obs_cost_min, "Min cost = %.2f"% min_cost)
    config.intmap_rgb = draw_traj(config, traj, green)

    # config.viz_pub.publish(config.br.cv2_to_imgmsg(config.intmap_rgb, encoding="bgr8"))
    
    return config.min_u


# Calculate goal cost via Pythagorean distance to robot
def calc_to_goal_cost(traj, config):
    
    # If-Statements to determine negative vs positive goal/trajectory position
    # traj[-1,0] is the last predicted X coord position on the trajectory
    if (config.goalX >= 0 and traj[-1,0] < 0):
        dx = config.goalX - traj[-1,0]
    elif (config.goalX < 0 and traj[-1,0] >= 0):
        dx = traj[-1,0] - config.goalX
    else:
        dx = abs(config.goalX - traj[-1,0])
    
    # traj[-1,1] is the last predicted Y coord position on the trajectory
    if (config.goalY >= 0 and traj[-1,1] < 0):
        dy = config.goalY - traj[-1,1]
    elif (config.goalY < 0 and traj[-1,1] >= 0):
        dy = traj[-1,1] - config.goalY
    else:
        dy = abs(config.goalY - traj[-1,1])

    # print("dx, dy", dx, dy)
    cost = math.sqrt(dx**2 + dy**2)
    # print("Cost: ", cost)
    return cost


def calc_obs_cost(traj, config):
    # print("Trajectory end-points wrt odom", traj[-1, 0], traj[-1, 1])

    # Convert traj points to robot frame
    x_end_odom = traj[-1, 0]
    y_end_odom = traj[-1, 1]

    # Trajectory approx mid-points
    x_mid_odom = traj[int(math.floor(len(traj)/2)), 0]
    y_mid_odom = traj[int(math.floor(len(traj)/2)), 1]

    # print(x_end_odom, x_mid_odom)
    # print("Odometry:", config.x, config.y)

    x_end_rob = (x_end_odom - config.x)*math.cos(config.th) + (y_end_odom - config.y)*math.sin(config.th)
    y_end_rob = -(x_end_odom - config.x)*math.sin(config.th) + (y_end_odom - config.y)*math.cos(config.th)
    x_mid_rob = (x_mid_odom - config.x)*math.cos(config.th) + (y_mid_odom - config.y)*math.sin(config.th)
    y_mid_rob = -(x_mid_odom - config.x)*math.sin(config.th) + (y_mid_odom - config.y)*math.cos(config.th)


    # int() and floor() behave differently with -ve numbers. int() is symmetric. 
    # cm_col = config.costmap_shape[0]/2 - math.floor(y_end_rob/config.costmap_resolution)
    # cm_row = config.costmap_shape[1]/2 - math.floor(x_end_rob/config.costmap_resolution)
    cm_col = config.costmap_shape[0]/2 - int(y_end_rob/config.costmap_resolution)
    cm_row = config.costmap_shape[1]/2 - int(x_end_rob/config.costmap_resolution)

    cm_mid_col = config.costmap_shape[0]/2 - int(y_mid_rob/config.costmap_resolution)
    cm_mid_row = config.costmap_shape[1]/2 - int(x_mid_rob/config.costmap_resolution)

    # print("End point coordinates", cm_col, cm_row)
    # print("Mid point coordinates", cm_mid_col, cm_mid_row)


    # !!! NOTE !!!: IN COSTMAP, VALUES SHOULD BE ACCESSED AS (ROW,COL). FOR VIZ, IT SHOULD BE (COL, ROW)! 
    # Sanity Check: Drawing end and mid points
    # config.intmap_rgb = cv2.circle(config.intmap_rgb, (int(cm_col), int(cm_row)), 1, (255, 255, 255), 1)
    # config.intmap_rgb = cv2.circle(config.intmap_rgb, (int(cm_mid_col), int(cm_mid_row)), 1, (0, 255, 0), 1)
    
    # print("Value at end-point = ", config.costmap_baselink[int(cm_row), int(cm_col)])
    # print("Max and min of costmap: ", np.max(config.costmap_baselink), np.min(config.costmap_baselink))

    # Cost which only considers trajectory end point
    # veg_cost = config.costmap_baselink_low[int(cm_row), int(cm_col)]
    
    # Cost which considers trajectory mid point and end point
    veg_cost = config.intensitymap_mid_inflated[int(cm_row), int(cm_col)] + config.intensitymap_mid_inflated[int(cm_mid_row), int(cm_mid_col)]
    # veg_cost = config.intensitymap_diff_inflated[int(cm_row), int(cm_col)] + config.intensitymap_diff_inflated[int(cm_mid_row), int(cm_mid_col)]

    return veg_cost



def draw_traj(config, traj, color):
    traj_array = np.asarray(traj)
    x_odom_list = np.asarray(traj_array[:, 0])
    y_odom_list = np.asarray(traj_array[:, 1])

    # print(x_odom_list.shape)

    x_rob_list, y_rob_list = odom_to_robot(config, x_odom_list, y_odom_list)
    cm_col_list, cm_row_list = robot_to_costmap(config, x_rob_list, y_rob_list)

    costmap_traj_pts = np.array((cm_col_list.astype(int), cm_row_list.astype(int))).T
    # print(costmap_traj_pts) 

    costmap_traj_pts = costmap_traj_pts.reshape((-1, 1, 2))
    config.intmap_rgb = cv2.polylines(config.intmap_rgb, [costmap_traj_pts], False, color, 1)
    
    return config.intmap_rgb




# NOTE: x_odom and y_odom are numpy arrays
def odom_to_robot(config, x_odom, y_odom):
    
    # print(x_odom.shape)
    # print(round(config.x, 2), round(config.y, 2))
    x_rob_odom_list = np.asarray([round(config.x, 2) for i in range(x_odom.shape[0])])
    y_rob_odom_list = np.asarray([round(config.y, 2) for i in range(y_odom.shape[0])])

    x_rob = (x_odom - x_rob_odom_list)*math.cos(config.th) + (y_odom - y_rob_odom_list)*math.sin(config.th)
    y_rob = -(x_odom - x_rob_odom_list)*math.sin(config.th) + (y_odom - y_rob_odom_list)*math.cos(config.th)
    # print("Trajectory end-points wrt robot:", x_rob, y_rob)

    return x_rob, y_rob


def robot_to_costmap(config, x_rob, y_rob):

    costmap_shape_list_0 = [config.costmap_shape[0]/2 for i in range(y_rob.shape[0])]
    costmap_shape_list_1 = [config.costmap_shape[1]/2 for i in range(x_rob.shape[0])]

    y_list = [math.floor(y/config.costmap_resolution) for y in y_rob]
    x_list = [math.floor(x/config.costmap_resolution) for x in x_rob]

    cm_col = np.asarray(costmap_shape_list_0) - np.asarray(y_list)
    cm_row = np.asarray(costmap_shape_list_1) - np.asarray(x_list)
    # print("Costmap coordinates of end-points: ", (int(cm_row), int(cm_col)))

    return cm_col, cm_row


# Begin DWA calculations
def dwa_control(x, u, config, ob):
    # Dynamic Window control

    dw = calc_dynamic_window(x, config)

    u = calc_final_input(x, u, dw, config, ob)

    return u


# Determine whether the robot has reached its goal
def atGoal(config, x):
    # check at goal
    if math.sqrt((x[0] - config.goalX)**2 + (x[1] - config.goalY)**2) <= config.robot_radius:
        return True
    return False




def main():
    print(__file__ + " start!!")
    
    config = Config()
    obs = Obstacles()

    subOdom = rospy.Subscriber("/spot/odometry", Odometry, config.assignOdomCoords)
    subLaser = rospy.Subscriber("/scan", LaserScan, obs.assignObs, config)
    subGoal = rospy.Subscriber('/target/position', Twist, config.target_callback)

    # Lidar Intensity Map subscriber
    rospy.Subscriber("/intensity_map_mid", OccupancyGrid, config.intensity_map_mid_cb)
    rospy.Subscriber("/intensity_map_low", OccupancyGrid, config.intensity_map_low_cb)
    rospy.Subscriber("/intensity_map_high", OccupancyGrid, config.intensity_map_high_cb)

    choice = input("Publish? 1 or 0")
    if(int(choice) == 1):
        pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
        print("Publishing to cmd_vel")
    else:
        pub = rospy.Publisher("/dont_publish", Twist, queue_size=1)
        print("Not publishing!")

    speed = Twist()
    
    # initial state [x(m), y(m), theta(rad), v(m/s), omega(rad/s)]
    x = np.array([config.x, config.y, config.th, 0.0, 0.0])
    
    # initial linear and angular velocities
    u = np.array([0.0, 0.0])


    # runs until terminated externally
    while not rospy.is_shutdown():

        # Initial
        if config.goalX == 0.0006 and config.goalY == 0.0006:
            # print("Initial condition")
            speed.linear.x = 0.0
            speed.angular.z = 0.0
            x = np.array([config.x, config.y, config.th, 0.0, 0.0])
        
        # Pursuing but not reached the goal
        elif (atGoal(config,x) == False): 

            u = dwa_control(x, u, config, obs.obst)

            x[0] = config.x
            x[1] = config.y
            x[2] = config.th
            x[3] = u[0]
            x[4] = u[1]
            speed.linear.x = x[3]
            speed.angular.z = x[4]


        # If at goal then stay there until new goal published
        else:
            print("Goal reached!")
            speed.linear.x = 0.0
            speed.angular.z = 0.0
            x = np.array([config.x, config.y, config.th, 0.0, 0.0])
        
        config.viz_pub.publish(config.br.cv2_to_imgmsg(config.intmap_rgb, encoding="bgr8"))
        pub.publish(speed)
        config.r.sleep()

    cv2.destroyAllWindows()



if __name__ == '__main__':
    rospy.init_node('dwa_costmap')
    main()
