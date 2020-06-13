import cv2
import sys
import time
import math
import numpy as np
import pybullet as p
import pybullet_data
from imageai.Detection.Custom import CustomObjectDetection
from scipy.spatial import distance
import sys

grid = np.zeros((10,10), dtype = int)

REMAP_INTERPOLATION = cv2.INTER_LINEAR

DEPTH_VISUALIZATION_SCALE = 2048


calibration = np.load("calibration.npz", allow_pickle=False)
imageSize = tuple(calibration["imageSize"])
leftMapX = calibration["leftMapX"]
leftMapY = calibration["leftMapY"]
leftROI = tuple(calibration["leftROI"])
rightMapX = calibration["rightMapX"]
rightMapY = calibration["rightMapY"]
rightROI = tuple(calibration["rightROI"])

CAMERA_WIDTH = 512
CAMERA_HEIGHT = 512


stereoMatcher = cv2.StereoBM_create()
stereoMatcher.setMinDisparity(16)
stereoMatcher.setNumDisparities(128)
stereoMatcher.setBlockSize(55)
stereoMatcher.setROI1(leftROI)
stereoMatcher.setROI2(rightROI)
stereoMatcher.setSpeckleRange(32)
stereoMatcher.setSpeckleWindowSize(45)


p.connect(p.GUI)  #or p.SHARED_MEMORY or p.DIRECT
p.setAdditionalSearchPath(pybullet_data.getDataPath())
plane = p.loadURDF("plane.urdf")
p.setGravity(0, 0, -10)
carpos = [19, 1, 0.1]
targetpos=[5, 5, 0.1]

# Loading Obstacles and Husky
b1 = p.loadURDF("base.urdf",[20,10,1],p.getQuaternionFromEuler([0,1.57079632,0]), useFixedBase = True)
b2 = p.loadURDF("base.urdf",[0,10,1],p.getQuaternionFromEuler([0,1.57079632,0]), useFixedBase = True)
b3 = p.loadURDF("base.urdf",[10,20,1],p.getQuaternionFromEuler([0,1.57079632,1.57079632]), useFixedBase = True)
b4 = p.loadURDF("base.urdf",[10,0,1],p.getQuaternionFromEuler([0,1.57079632,1.57079632]), useFixedBase = True)
r2r1 = p.loadURDF("r2d2.urdf",[15.2,12.5,0.5], useFixedBase = True)
r2r2 = p.loadURDF("r2d2.urdf",[7,16.5,0.5], useFixedBase = True)
r2r4 = p.loadURDF("r2d2.urdf",[3,3,0.5], useFixedBase = True)
teddy1 = p.loadURDF("teddy_large.urdf",[17.5,11.5,-0.7],p.getQuaternionFromEuler([1.57079632,0,0]), useFixedBase = True)
teddy2 = p.loadURDF("teddy_large.urdf",[6,1.8,-0.7],p.getQuaternionFromEuler([1.57079632,0,0]), useFixedBase = True)
sphere1 = p.loadURDF("sphere2red.urdf",[3,11,0.3],p.getQuaternionFromEuler([1.57079632,0,0]), useFixedBase = True)
sphere2 = p.loadURDF("sphere2red.urdf",[11,15,0.3],p.getQuaternionFromEuler([1.57079632,0,0]), useFixedBase = True)
sphere3 = p.loadURDF("sphere2red.urdf",[13,7,0.3],p.getQuaternionFromEuler([1.57079632,0,0]), useFixedBase = True)
#human1 = p.loadURDF("humanoid.urdf",[7,6.5,0.8],p.getQuaternionFromEuler([0,0,1.57079632]), useFixedBase = True)
car = p.loadURDF("husky/husky.urdf", carpos, p.getQuaternionFromEuler([0,0,1.57079632]))
car2 = p.loadURDF("husky/husky.urdf", targetpos, p.getQuaternionFromEuler([0,0,1.57079632]))
human2 = p.loadURDF("humanoid.urdf",[3,17,0.8],p.getQuaternionFromEuler([0,0,1.57079632]), useFixedBase = True)
duck1 = p.loadURDF("duck_vhacd.urdf",[17,17,0],p.getQuaternionFromEuler([1.57079632,0,0]), useFixedBase = True)
duck2 = p.loadURDF("duck_vhacd.urdf",[15,3,0],p.getQuaternionFromEuler([1.57079632,0,0]), useFixedBase = True)
cube1 = p.loadURDF("capsule.urdf",[7,13,0.25], useFixedBase = True)
cube2 = p.loadURDF("capsule.urdf",[9,9,0.25], useFixedBase = True)
numJoints = p.getNumJoints(car)
#chess = p.loadURDF("chess.urdf",[2,0,0], p.getQuaternionFromEuler([0,1.57079632,0]))
#base = p.loadURDF("base.urdf",[3,0,0], p.getQuaternionFromEuler([0,1.57079632,0]))

#Setting Constants
targetVel = 50  #rad/s
targetVelRev = -1
maxForce = 100 #Newton
#p.applyExternalForce(car,3,[100,0,0],)
targetVel1 = 10
targetVel2 = -10
targetVelS = 0
width = 512                               #Setting parameters for the camera image
height = 512

fov = 60
aspect = width / height
near = 0.02                                     #Near plane
far = 6
kp = 0.1
kd = 0.9
change = False  # Whether Lanes can be changed or not
j = 0   # Counter
prevOrNext = 1  # To determine whether the searcher should go to the next lane or the previous lane while searching
EPSILON = 0.004 # Maximum Possible Error while turning

p.resetDebugVisualizerCamera(10, 0, -89, [10,10,10])

# Loading Model for Detection
detector=CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath('detection_model-ex-005--loss-0004.657.h5')
detector.setJsonPath('detection_config.json')
detector.loadModel()


def mapp():
    '''
    To Initialize the depth map.
    '''
    img = np.zeros((400,400,3),dtype = int)
    for i in range(400):
        for j in range(400):
            if(grid[int(i/40)][int(j/40)] == 0):
                img[i][j] = np.array([255,255,255])
            if(grid[int(i/40)][int(j/40)] == 1):
                img[i][j] = np.array([51,153,255])                
            if(grid[int(i/40)][int(j/40)] == 2):
                img[i][j] = np.array([0,255,191])
    cv2.imwrite('map.png', img)          
	    

def depthAndDetect():
    '''
     To Create the Depth Map and to detect whether the object is the target or an obstacle.
    '''
    pos = p.getBasePositionAndOrientation(car)                              #Getting the position and orientation of the car
    cord = pos[0]
    ori = p.getEulerFromQuaternion(pos[1])
    unit = [0.45*math.cos(ori[2]),0.45*math.sin(ori[2]),0.20]                #Calculating a unit vector in direction of the car. The anle depends only upon YAW(z) as car is in xy plane
                             #Fixing the camera a little forward from the center of the car
    unit2 = [1*math.cos(ori[2]),1*math.sin(ori[2]),0.20] 
    see = [sum(x) for x in zip(cord,unit2)]                                #Setting the looking direction in direction of the car
    shift = [0.02*math.cos(ori[2]+1.57079632), 0.02*math.sin(ori[2]+1.57079632), 0]
    cam = [sum(x) for x in zip(cord,unit)]
    left = [sum(x) for x in zip(cam, shift)]
    right = [cam[0] - shift[0], cam[1] - shift[1], cam[2]]
    see_left = [see[0] + shift[0], see[1] + shift[1], see[2] + shift[2]]
    see_right = [see[0] - shift[0], see[1] - shift[1], see[2] - shift[2]]
    view_matrix_left = p.computeViewMatrix(left, see_left , [0,0,1])                  #Calculating the position of the camera
    pro_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)      #Getting the resolution matrix based on camera properties
    pre_image_left = p.getCameraImage(width, height, view_matrix_left, pro_matrix, shadow = True, renderer=p.ER_BULLET_HARDWARE_OPENGL)  #Getting the image from the camera
    view_matrix_right = p.computeViewMatrix(right, see_right , [0,0,1])                  #Calculating the position of the camera
    pre_image_right = p.getCameraImage(width, height, view_matrix_right, pro_matrix, shadow = True, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb_opengl = np.reshape(pre_image_right[2], (height, width, 4))
    cv2.imwrite('husky.png',rgb_opengl)
    detections=detector.detectObjectsFromImage(input_image='husky.png',output_image_path='husky_detection.png')

    # Checking whether the target husky was found
    if len(detections)==0:
        print('Husky not detected!')
        found = 0
    else:
        for detection in detections:
            if detection['name']=='husky':
                if detection['percentage_probability']>30:
                    print('Husky detected!')
                    img=cv2.imread('husky_detection.png')
                    img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
                    cv2.imshow("image",img)
                    found = 1
                    mapp()
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        sys.exit()
                        break
                else:
                    print("Husky not detected!")
                    found = 0
                
            else:
                print("Husky not detected!")
                found = 0
    if(found == 0):
        leftFrame = cv2.cvtColor(pre_image_left[2], cv2.COLOR_RGB2BGR)             #Image returned is in RGB format but opencv works with BGR so converting the colo scheme
        rightFrame = cv2.cvtColor(pre_image_right[2], cv2.COLOR_RGB2BGR)
        leftHeight, leftWidth = leftFrame.shape[:2]
        rightHeight, rightWidth = rightFrame.shape[:2]          
        fixedLeft = cv2.remap(leftFrame, leftMapX, leftMapY, REMAP_INTERPOLATION)
        fixedRight = cv2.remap(rightFrame, rightMapX, rightMapY, REMAP_INTERPOLATION)
        grayLeft = cv2.cvtColor(fixedLeft, cv2.COLOR_BGR2GRAY)
        grayRight = cv2.cvtColor(fixedRight, cv2.COLOR_BGR2GRAY)
        depth = stereoMatcher.compute(grayLeft, grayRight)
        cv2.imshow('left_remap', fixedLeft)
        cv2.imshow('right_remap', fixedRight)
        depth = depth / DEPTH_VISUALIZATION_SCALE
        ret, mask = cv2.threshold(depth, 0.13, 1, cv2.THRESH_BINARY)
        mask = mask[40:400,170:480]
        cv2.imshow('mask',mask)
        mask = cv2.resize(mask, (50,50))
        fraction = 0
        for x in mask:
            for y in x:
                if(y == 1):
                    fraction += 1
        fraction = fraction/(50*50)
        if(fraction>0.2):
            print("Object in front")
            cv2.destroyAllWindows()
            return 1
            
        else:
            print("No Object in front")
            cv2.destroyAllWindows()
            return 0
            
  
def Correct_Orientation():
    '''
    To correct the orientation error which occurs after turning.
    '''
    carorient=p.getEulerFromQuaternion(p.getBasePositionAndOrientation(car)[1])[2]
    print(carorient)
    if carorient>0:
        p.resetBasePositionAndOrientation(car,p.getBasePositionAndOrientation(car)[0],p.getQuaternionFromEuler([0,0,np.pi/2]))
        p.stepSimulation()
        
    else:
        p.resetBasePositionAndOrientation(car,p.getBasePositionAndOrientation(car)[0],p.getQuaternionFromEuler([0,0,-1*np.pi/2]))
        p.stepSimulation()

def Obs(orientation):
    '''
    Called when an obstacle is encountered. Has the capability to dodge the obstacle from the left side as well as the right side, as required.
    '''
    if orientation == 'left':
        rightTurn()
        forwardByN(1.8, possibleLaneChange = False)   #1
        leftTurn()
        lane_changed = forwardByN(3.3, possibleLaneChange = True) #2.5
        if lane_changed == True:
            lane_changed = False
            return
        leftTurn()
        forwardByN(1.8, possibleLaneChange = False)
        rightTurn()
    else:
        leftTurn()
        forwardByN(1.8, possibleLaneChange = False)
        rightTurn()
        lane_changed = forwardByN(3.3, possibleLaneChange = True)
        if lane_changed == True:
            lane_changed = False
            return
        rightTurn()
        forwardByN(1.8, possibleLaneChange = False)
        leftTurn()
    Correct_Orientation()
        
        
        

def rightTurn():
    '''
    To turn the bot rightwards.
    '''
    last_error = 0
    targetOrient = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(car)[1])[2] - math.pi / 2
    actual_target_angle = targetOrient if targetOrient >= -math.pi else targetOrient + 2 * math.pi
    current_angle = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(car)[1])[2]
    # print("Targeted Orienatation:", str(actual_target_angle))
    # print("Current Orientation:", str(current_angle))
    if current_angle < 0 and actual_target_angle > 0: # As pybullet angles range from -pi to +pi
        actual_target_angle = -math.pi
        while current_angle > -math.pi and current_angle < 0:
            error = current_angle - actual_target_angle
            targetVel = 6
            speed_correction = kp * error + kd * (error - last_error) # Applying PD from PID
            # print("1Targeted Orienatation:", str(actual_target_angle))
            # print("1Current Orientation:", str(current_angle))
            for joint in range(2,5,2):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel + speed_correction,force = maxForce)
            for joint in range(3,6,2):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = -1*targetVel - speed_correction,force = maxForce)
            current_angle = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(car)[1])[2]
            p.stepSimulation()
    
    actual_target_angle = targetOrient if targetOrient >= -math.pi else targetOrient + 2 * math.pi
    while abs(current_angle - actual_target_angle) > EPSILON:
        error = current_angle - actual_target_angle
        targetVel = 6
        speed_correction = kp * error + kd * (error - last_error) # Applying PD from PID
        # print("Targeted Orienatation:", str(actual_target_angle))
        # print("Current Orientation:", str(current_angle))
        for joint in range(2,5,2):
            p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel + speed_correction,force = maxForce)
        for joint in range(3,6,2):
            p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = -1*targetVel - speed_correction,force = maxForce)
        current_angle = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(car)[1])[2]
        p.stepSimulation()
    for joint in range(2, 6):
        p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL, targetVelocity = targetVelS, force = maxForce)
    print("Right Turn")
    return actual_target_angle





def leftTurn():
    '''
    To Turn the Bot Leftwards.
    '''
    last_error = 0
    targetOrient = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(car)[1])[2] + math.pi / 2
    actual_target_angle = targetOrient if targetOrient <= math.pi else targetOrient - 2 * math.pi
    current_angle = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(car)[1])[2]
    # print("Targeted Orienatation:", str(actual_target_angle))
    # print("Current Orientation:", str(current_angle))
    if current_angle > 0 and actual_target_angle < 0: # As pybullet angles range from -pi to +pi
        actual_target_angle = math.pi
        while current_angle < math.pi and current_angle > 0:
            error = actual_target_angle - current_angle
            targetVel = 6
            speed_correction = kp * error + kd * (error - last_error) # Applying PD from PID
            # print("1Targeted Orienatation:", str(actual_target_angle))
            # print("1Current Orientation:", str(current_angle))
            for joint in range(2,5,2):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = -1*targetVel - speed_correction,force = maxForce)
            for joint in range(3,6,2):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel + speed_correction,force = maxForce)
            current_angle = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(car)[1])[2]
            p.stepSimulation()

    actual_target_angle = targetOrient if targetOrient <= math.pi else targetOrient - 2 * math.pi
    while abs(current_angle - actual_target_angle) > EPSILON:
        error = actual_target_angle - current_angle
        targetVel = 6
        speed_correction = kp * error + kd * (error - last_error) # Applying PD from PID
        # print("Targeted Orienatation:", str(actual_target_angle))
        # print("Current Orientation:", str(current_angle))
        for joint in range(2,5,2):
            p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = -1*targetVel - speed_correction,force = maxForce)
        for joint in range(3,6,2):
            p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel + speed_correction,force = maxForce)
        current_angle = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(car)[1])[2]
        p.stepSimulation()
    for joint in range(2, 6):
        p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL, targetVelocity = targetVelS, force = maxForce)
    print("Left Turn")
    return actual_target_angle





def forwardByN(n, possibleLaneChange):
    '''
    To move forward by n blocks, provided to the function, along with a variable which mentions whether the lane can be changed during this motion.
    Lanes can be changed only if the motion entails moving closer to the end of the lane.
    '''
    print("Foward by N with possibleLaneChange = ", possibleLaneChange)
    global prevOrNext
    current_angle = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(car)[1])[2]
    if (current_angle < math.pi/4 and current_angle > -math.pi/4) or (current_angle < -3/4*math.pi) or (current_angle > 3/4*math.pi):
        carloc = p.getBasePositionAndOrientation(car)[0]
        carbox = [int(carloc[0]/2), int(carloc[1]/2)]
        center = [2*carbox[0]+1, 2*carbox[1]+1]
        while distance.euclidean([p.getBasePositionAndOrientation(car)[0][0], p.getBasePositionAndOrientation(car)[0][1]] , center) < n:
            if possibleLaneChange == True:
                curr_x = p.getBasePositionAndOrientation(car)[0][0]
                if curr_x > 9 or curr_x < -9:
                    if prevOrNext % 2 == 0:
                        changeLanes('next')
                    else:
                        changeLanes('prev')
                    return True
            targetVel = 5
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL, targetVelocity = targetVel, force = maxForce)
            p.stepSimulation()
    else:
        carloc = p.getBasePositionAndOrientation(car)[0]
        carbox = [int(carloc[0]/2), int(carloc[1]/2)]
        center = [2*carbox[0]+1, 2*carbox[1]+1]
        while distance.euclidean([p.getBasePositionAndOrientation(car)[0][0], p.getBasePositionAndOrientation(car)[0][1]] , center) < n:
            targetVel = 5
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL, targetVelocity = targetVel, force = maxForce)
            p.stepSimulation()
    for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVelS,force = maxForce)
    print("Forward by " + str(n) + " steps")
    return False




def changeLanes(lane):
    '''
    To change the lane when end of the lane is encountered.
    '''
    print("ChangeLanes called")
    global prevOrNext
    global change
    if lane == 'next':
        print("Going to Next Lane")
        rightTurn()
        obstacle = depthAndDetect()
        if obstacle == 0:
            forwardByN(1.8, possibleLaneChange = False)
            rightTurn()
        else:
            grid[carbox[0]-1][carbox[1]] = 2
            rightTurn()
            forwardByN(2, possibleLaneChange = False)
            leftTurn()
            forwardByN(1.8, possibleLaneChange = False)
            rightTurn()
    else:
        print("Going to Previous Lane")
        leftTurn()
        obstacle = depthAndDetect()
        if obstacle == 0:
            forwardByN(1.8, possibleLaneChange = False)
            leftTurn()
        else:
            grid[carbox[0]-1][carbox[1]] = 2
            leftTurn()
            forwardByN(2, possibleLaneChange = False)
            rightTurn()
            forwardByN(1.8, possibleLaneChange = False)
            leftTurn()
    prevOrNext += 1
    change = False
    Correct_Orientation()





def ObstacleEncountered(carbox, carori):
    '''
    Called when an obstacle is encountered.
    Calls Obs(orientation) after chechik whether the obstacle is skewed left of right with respect to the searcher.
    '''
    targetVel = 0
    for joint in range(2, 6):
        p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel,force = maxForce)
    if(carori > 0):
        if(carbox[0] > 8):
            orientation = 'right'
        elif(carbox[0] < 1):
            orientation = 'left'
        elif(grid[carbox[0]+1][carbox[1]] == 1):
            orientation = 'left'
        else:
            orientation = 'right'

    if(carori < 0):
        if(carbox[0] > 8):
            orientation = 'left'
        elif(carbox[0] < 1):
            orientation = 'right'
        elif(grid[carbox[0]+1][carbox[1]] == 1):
            orientation = 'right'
        else:
            orientation = 'left'        
    Obs(orientation)
            
            


    

while True:
    keys = p.getKeyboardEvents()
    # Up, Down, Left and Right keys are for manually controlling the bot
    # c is for preparing the depth map manually
    # h is for starting the autonomous part
    for k, v in keys.items():
        if (k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN)):
            targetVel = 10
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL, targetVelocity = targetVel,force = maxForce)
           
        
        if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED)):
            targetVel = 0
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel,force = maxForce)
            
          
        if (k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN)):
            targetVel = -3
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel,force = maxForce)
            
        
        if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED)):
            targetVel = 0
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel,force = maxForce)
            
            

        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN)):         #for left turn 
            targetVel = 10
            for joint in range(2,5,2):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = -1*targetVel,force = maxForce)
            for joint in range(3,6,2):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel,force = maxForce)


        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
            targetVel = 0
            for joint in range(2,6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel,force = maxForce)


        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN)):      #for right turn
            targetVel = 10
            for joint in range(2,5,2):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel,force = maxForce)
            for joint in range(3,6,2):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = -1*targetVel,force = maxForce)


        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
            targetVel = 0
            for joint in range(2,6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel,force = maxForce)


        if (k == ord('c') and (v & p.KEY_WAS_RELEASED)):
            depthAndDetect()
            print(p.getBasePositionAndOrientation(car)[1][2])
            mapp()
            
        
        if(k == ord('h') and (v & p.KEY_WAS_TRIGGERED)):
            targetVelh = 10
            br = 0
            while True:
                for joint in range(2, 6):
                    p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL, targetVelocity = targetVelh,force = maxForce)
                keys = p.getKeyboardEvents()
                for k, v in keys.items():
                    if k == ord('b') and (v & p.KEY_WAS_TRIGGERED):
                        br = 1
                        break
                if br == 1:
                    break
                carloc = p.getBasePositionAndOrientation(car)[0]
                carori = p.getBasePositionAndOrientation(car)[1][2]
                if carloc[1] < 16 and carloc[1] > 4:
                    j += 1
                    change = True
                    if j == 1:
                        print("Lanes can be changed Automatically Now")
                    if j == 1000: # To prevent Overflow
                        j = 2
                carbox = np.array([int(carloc[0]/2), int(carloc[1]/2)])
                center = np.array([2*carbox[0]+1, 2*carbox[1]+1])
                carpla = np.array([carloc[0], carloc[1]])                    
                if change == True:
                    if carloc[1] < 1.4 or carloc[1] > 18.6:
                        grid[carbox[0]][carbox[1]] = 1
                        if prevOrNext % 2 == 0:
                            changeLanes('next')
                        else:
                            changeLanes('prev')
                        j = 0        
                stop = 0
                if(carori > 0):
                    if(carpla[1] > center[1]):
                        stop = 1
                elif(carori < 0):
                    if(carpla[1] < center[1]):
                        stop = 1
                if(grid[carbox[0]][carbox[1]] == 0 and (distance.euclidean(carpla, center) > 0.5) and stop == 1):
                    for joint in range(2, 6):
                        p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL, targetVelocity = 0,force = maxForce)
                    grid[carbox[0]][carbox[1]] = 1
                    obj = depthAndDetect()
                    if(obj == 1):
                        if(carori > 0.97 and carori < 1):
                            grid[carbox[0]-1][carbox[1]] = 2
                        elif(carori > 0):
                            grid[carbox[0]][carbox[1]+1] = 2
                        elif(carori < 0):
                            grid[carbox[0]][carbox[1]-1] = 2
                        
                        ObstacleEncountered(carbox, carori)
                    
                
                        
                p.stepSimulation()
                time.sleep(1./240.)                
                    
                
        p.stepSimulation()
        time.sleep(1./240.)
