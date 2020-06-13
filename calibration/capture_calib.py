
import pybullet as p
import pybullet_data
import time
import math
import cv2
import numpy
p.connect(p.GUI)  #or p.SHARED_MEMORY or p.DIRECT
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, 0)
carpos = [0, 0, 0.1]

LEFT_PATH = "capture/left/{:06d}.jpg"
RIGHT_PATH = "capture/right/{:06d}.jpg"
frameId = 60

car = p.loadURDF("husky/husky.urdf", carpos)
numJoints = p.getNumJoints(car)
#cube = p.loadURDF("sphere2.urdf",[2,1,0])
chess = p.loadURDF("chess.urdf",[1.5,-0.2,0.7], p.getQuaternionFromEuler([0,1.57079632,0]))
base = p.loadURDF("base.urdf",[3,0,0], p.getQuaternionFromEuler([0,1.57079632,0]))
#door = p.loadURDF("r2d2.urdf", [2, -0.8, 0], p.getQuaternionFromEuler([0,0,-1.57079632]))
#cube = p.loadURDF("sphere2red.urdf",[2,0,-0.5])
#teddy = p.loadURDF("teddy_large.urdf",[1,0,1])
#teddy2 = p.loadURDF("teddy_large.urdf",[1,-3,1])
for joint in range(numJoints):
  print(p.getJointInfo(car, joint))
targetVel = 10  #rad/s
targetVelRev = -1
maxForce = 100 #Newton
#p.applyExternalForce(car,3,[100,0,0],)
targetVel1 = 4
targetVel2 = -4
targetVelS = 0


width = 512                               #Setting parameters for the camera image
height = 512

fov = 60
aspect = width / height
near = 0.02                                     #Near plane
far = 8                                         #Far plane




while (1):
    keys = p.getKeyboardEvents()
    print(p.getLinkState(car, 0))
    for k, v in keys.items():
        if (k == p.B3G_UP_ARROW and (v & p.KEY_IS_DOWN)):
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL, targetVelocity =targetVel,force = maxForce)
           
            p.stepSimulation()
            time.sleep(1./240.)
        if (k == p.B3G_UP_ARROW and (v & p.KEY_WAS_RELEASED)):
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVelS,force = maxForce)
            
            p.stepSimulation()
            time.sleep(1./240.)          
        if (k == p.B3G_DOWN_ARROW and (v & p.KEY_IS_DOWN)):
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVelRev,force = maxForce)
            
            p.stepSimulation()
        if (k == p.B3G_DOWN_ARROW and (v & p.KEY_WAS_RELEASED)):
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVelS,force = maxForce)
            p.stepSimulation()
            time.sleep(1./240.)           
        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_IS_DOWN)):
            for joint in [2,4]:
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel2,force = maxForce)
            for joint in [3,5]:
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel1,force = maxForce)
            p.stepSimulation()
            time.sleep(1./240.)
            
        if (k == p.B3G_LEFT_ARROW and (v & p.KEY_WAS_RELEASED)):
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVelS,force = maxForce)
            
            p.stepSimulation()
            time.sleep(1./240.)
            
        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_IS_DOWN)):
            for joint in [2,4]:
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel1,force = maxForce)
            for joint in [3,5]:
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVel2,force = maxForce)
            p.stepSimulation()
            time.sleep(1./240.)

            
        if (k == p.B3G_RIGHT_ARROW and (v & p.KEY_WAS_RELEASED)):
            for joint in range(2, 6):
                p.setJointMotorControl2(car, joint, p.VELOCITY_CONTROL,targetVelocity = targetVelS,force = maxForce)
            
            p.stepSimulation()
            time.sleep(1./240.)



        if (k == ord('r') and (v & p.KEY_IS_DOWN)):
            p.applyExternalTorque(car, -1, [0,0,10], flags = p.LINK_FRAME)
            p.stepSimulation()
            time.sleep(1./240.)
        	
            
            
        if (k == ord('r') and (v & p.KEY_WAS_RELEASED)):
            p.applyExternalTorque(car, -1, [0,0,0], flags = p.LINK_FRAME)
            p.stepSimulation()
            time.sleep(1./240.)
            
        if (k == ord('a') and (v & p.KEY_IS_DOWN)):
            targetVel = targetVel + 1
            targetVel1 = targetVel1 + 0.5
            targetVel2 = targetVel2 - 0.5
            targetVelRev = targetVelRev - 1
            while(1):
                br = 0
                keys = p.getKeyboardEvents()
                for k, v in keys.items():
                    if(k == ord('a') and (v & p.KEY_WAS_RELEASED)):
                        br = 1
                        break
                if(br == 1):
                    break
                


        if (k == ord('c') and (v & p.KEY_WAS_RELEASED)):
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
            print(see_left)
            print(see_right)
            pro_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)      #Getting the resolution matrix based on camera properties
            pre_image_left = p.getCameraImage(width, height, view_matrix_left, pro_matrix, shadow = True, renderer=p.ER_BULLET_HARDWARE_OPENGL)  #Getting the image from the camera
            view_matrix_right = p.computeViewMatrix(right, see_right , [0,0,1])                  #Calculating the position of the camera
            pre_image_right = p.getCameraImage(width, height, view_matrix_right, pro_matrix, shadow = True, renderer=p.ER_BULLET_HARDWARE_OPENGL)  #Getting the image from the camera

            image_left = cv2.cvtColor(pre_image_left[2], cv2.COLOR_RGB2BGR)             #Image returned is in RGB format but opencv works with BGR so converting the colo scheme
            image_right = cv2.cvtColor(pre_image_right[2], cv2.COLOR_RGB2BGR)
            
            cv2.imshow('Husky_left', image_left)                                        #Displaying the image
            cv2.imshow('Husky_right', image_right)
            if cv2.waitKey(0) & 0xFF == ord('q'):
                cv2.imwrite(LEFT_PATH.format(frameId), image_left)
                cv2.imwrite(RIGHT_PATH.format(frameId), image_right)
                frameId += 1
                cv2.destroyAllWindows()
            elif cv2.waitKey(0) & 0xFF == ord('w'):
                cv2.destroyAllWindows()              



p.getContactPoints(car)

p.disconnect()
