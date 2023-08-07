import numpy as np
import math
import cv2
from numpy import dot, sqrt
from PIL import Image

def trignometry_for_distance(a, b):
    return math.sqrt(((b[0] - a[0]) * (b[0] - a[0])) +\
                     ((b[1] - a[1]) * (b[1] - a[1])))

def align_face(raw_face, left_eye, right_eye):
    right_eye_x = right_eye[0]
    right_eye_y = right_eye[1]

    left_eye_x = left_eye[0]
    left_eye_y = left_eye[1]

    # finding rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate image direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    a = trignometry_for_distance(left_eye, point_3rd)
    b = trignometry_for_distance(right_eye, point_3rd)
    c = trignometry_for_distance(right_eye, left_eye)
    cos_a = (b*b + c*c - a*a)/(2*b*c)
    angle = (np.arccos(cos_a) * 180) / math.pi

    if direction == -1:
        angle = 90 - angle

    # Rotate the face image
    rows, cols = raw_face.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle * direction, 1)
    rotated_image = cv2.warpAffine(raw_face, rotation_matrix, (cols, rows))

    # Fill the background with white color
    mask = np.all(rotated_image == [0, 0, 0], axis=-1)
    rotated_image[mask] = [255, 255, 255]

    # Save the resulting image
    # cv2.imwrite("rotated_filled_image.jpg", rotated_image)
    return rotated_image

def draw_box(image, boxes, color=(125, 255, 125), thickness = 10):
    """Draw square boxes on image"""
    edge_pixel = 20
    for box in boxes:
        # cv2.rectangle(image,
        #                 (int(box[0]), int(box[1])),
        #                 (int(box[2]), int(box[3])), color, 3)
        # Top-left
        cv2.line(image, (int(box[0]), int(box[1])), (int(box[0] + edge_pixel), int(box[1])), color, thickness)
        cv2.line(image, (int(box[0]), int(box[1])), (int(box[0]), int(box[1] + edge_pixel)), color, thickness)
        # Top-right
        cv2.line(image, (int(box[2]), int(box[1])), (int(box[2] - edge_pixel), int(box[1])), color, thickness)
        cv2.line(image, (int(box[2]), int(box[1])), (int(box[2]), int(box[1] + edge_pixel)), color, thickness)
        # Bottom-right
        cv2.line(image, (int(box[2]), int(box[3])), (int(box[2] - edge_pixel), int(box[3])), color, thickness)
        cv2.line(image, (int(box[2]), int(box[3])), (int(box[2]), int(box[3] - edge_pixel)), color, thickness)
        # # Bottom-left
        cv2.line(image, (int(box[0]), int(box[3])), (int(box[0] + edge_pixel), int(box[3])), color, thickness)
        cv2.line(image, (int(box[0]), int(box[3])), (int(box[0]), int(box[3] - edge_pixel)), color, thickness)

def draw_landmark(image, landmarks, color=(125, 255, 125)):
    """Draw landmarks on image"""
    # temp_img = image.copy()
    for index, idI in enumerate(landmarks):
        if index == 34 or index == 38 or index == 92 or index == 88:
            cv2.circle(image, (int(landmarks[index][0]), int(landmarks[index][1])), 2, (0, 0, 255), -1)  
        else:
            cv2.circle(image, (int(landmarks[index][0]), int(landmarks[index][1])), 1, color, -1)  
        # if (index % 2) == 0: 
        #     cv2.circle(temp_img, (int(landmarks[index][0]), int(landmarks[index][1])), 2, (0, 0, 255), -1)  
        # else:
        #     cv2.circle(temp_img, (int(landmarks[index][0]), int(landmarks[index][1])), 1, color, -1)  
        # cv2.imwrite("{}.jpg".format(index), temp_img)

def estimatePose(frame, landmarks):
    """Calculate poses"""
    size = frame.shape #(height, width, color_channel)
    image_points = np.array([
                            (landmarks[86][0], landmarks[86][1]),     # Nose tip
                            (landmarks[0][0], landmarks[0][1]),       # Chin
                            (landmarks[35][0], landmarks[35][1]),     # Left eye left corner
                            (landmarks[93][0], landmarks[93][1]),     # Right eye right corne
                            (landmarks[52][0], landmarks[52][1]),     # Left Mouth corner
                            (landmarks[61][0], landmarks[61][1])      # Right mouth corner
                        ], dtype="double")
    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])

    # Camera internals
 
    center = (size[1]/2, size[0]/2)
    focal_length = center[0] / np.tan(60/2 * np.pi / 180)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )

    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)#, flags=cv2.CV_ITERATIVE)

    
    axis = np.float32([[500,0,0], 
                          [0,500,0], 
                          [0,0,500]])
                          
    imgpts, jac = cv2.projectPoints(axis, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    modelpts, jac2 = cv2.projectPoints(model_points, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))
    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6] 

    
    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]


    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    # return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (image_points[0][0], image_points[0][1]), image_points
    return str(int(roll)), str(int(pitch)), str(int(yaw)), image_points