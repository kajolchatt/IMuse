import cv2 as cv
import mediapipe as mp
import time
import utils
import math
import numpy as np
import webbrowser

# Constants for redirection
BLINK_THRESHOLD_HAPPY = 2
BLINK_THRESHOLD_ROMANTIC = 3
BLINK_THRESHOLD_ENERGETIC = 4

# Function to check conditions and recommend music and redirect
def recommendAndRedirect(total_blinks, eye_position_left, eye_position_right):
    if total_blinks == BLINK_THRESHOLD_HAPPY:
        # Recommend happy music for one blink
        print("Recommend happy music.")
        webbrowser.open("https://www.youtube.com/watch?v=Cc_cNEjAh_Y&list=PL8U7gDbfLksNOQ-IbN_jfC9DVQYt4xXTo")
    elif total_blinks == BLINK_THRESHOLD_ROMANTIC:
        # Recommend romantic music for two blinks
        print("Recommend romantic music.")
        webbrowser.open("https://www.youtube.com/watch?v=ElZfdU54Cp8&list=PL0Z67tlyTaWphlJ8dod2fSFGmBlUW_KJJ")
    elif total_blinks == BLINK_THRESHOLD_ENERGETIC:
        # Recommend energetic music for three blinks
        print("Recommend energetic music.")
        webbrowser.open("https://www.youtube.com/watch?v=1xYZeDReUz4&list=PLy2B5nO_wrWhrdnSDFg9pEIIMbp6izokS")
    if eye_position_left == 'LEFT':
        # Recommend sad music for left eye movement
        print("Recommend sad music.")
        webbrowser.open("https://www.youtube.com/watch?v=SBWYGGDYmhg&list=PLHuHXHyLu7BGi-vR7X6j_xh_Tt9wy7pNA")
    elif eye_position_right == 'RIGHT':
        # Recommend sad music for right eye movement
        print("Recommend sad music.")
        webbrowser.open("https://www.youtube.com/watch?v=SBWYGGDYmhg&list=PLHuHXHyLu7BGi-vR7X6j_xh_Tt9wy7pNA")

# Variables
frame_counter = 0
CEF_COUNTER = 0
TOTAL_BLINKS = 0

# Constants
CLOSED_EYES_FRAME = 3
FONTS = cv.FONT_HERSHEY_COMPLEX

# Face boundary indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Lips indices for landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]

# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# Right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

map_face_mesh = mp.solutions.face_mesh
camera = cv.VideoCapture(0)

# Landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]
    return mesh_coord

# Euclidean distance function
def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blink ratio function
def blinkRatio(img, landmarks, right_indices, left_indices):
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)
    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio

# Eyes extractor function
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    dim = gray.shape
    mask = np.zeros(dim, dtype=np.uint8)

    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    eyes = cv.bitwise_and(gray, gray, mask=mask)
    eyes[mask == 0] = 155

    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    return cropped_right, cropped_left

# Eyes position estimator function
def positionEstimator(cropped_eye):
    h, w = cropped_eye.shape
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9, 9), 0)
    median_blur = cv.medianBlur(gaussain_blur, 3)
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    piece = int(w / 3)
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece + piece]
    left_piece = threshed_eye[0:h, piece + piece:w]

    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)
    return eye_position, color

# Pixel counter function
def pixelCounter(first_piece, second_piece, third_piece):
    right_part = np.sum(first_piece == 0)
    center_part = np.sum(second_piece == 0)
    left_part = np.sum(third_piece == 0)

    eye_parts = [right_part, center_part, left_part]
    max_index = eye_parts.index(max(eye_parts))

    pos_eye = ''
    if max_index == 0:
        pos_eye = "RIGHT"
        color = [utils.BLACK, utils.GREEN]
    elif max_index == 1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index == 2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye = "Closed"
        color = [utils.WHITE, utils.BLACK]
    return pos_eye, color

# Initialize mediapipe FaceMesh
with map_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

    # Start time
    start_time = time.time()
    
    # Main loop for video processing
    while True:
        frame_counter += 1
        
        # Capture frame-by-frame
        ret, frame = camera.read()
        if not ret:
            break
        
        # Resize frame for processing (if necessary)
        frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
        frame_height, frame_width = frame.shape[:2]
        
        # Convert frame to RGB for Mediapipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        
        # Process frame with FaceMesh
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Detect landmarks and draw on frame
            mesh_coords = landmarksDetection(frame, results, draw=True)
            
            # Calculate blink ratio
            ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
            cv.putText(frame, f'Blink Ratio: {round(ratio, 2)}', (30, 30), FONTS, 0.7, (0, 255, 0), 2)
            
            # Detect blinks and recommend music
            if ratio > 5.5:
                CEF_COUNTER += 1
            else:
                if CEF_COUNTER > CLOSED_EYES_FRAME:
                    TOTAL_BLINKS += 1
                    # Call music recommendation function based on blink count and eye position
                    recommendAndRedirect(TOTAL_BLINKS, eye_position_left, eye_position_right)
                CEF_COUNTER = 0
            
            # Display total blinks on frame
            cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (30, 60), FONTS, 0.7, (0, 255, 0), 2)
            
            # Extract eyes and estimate position
            right_eye, left_eye = eyesExtractor(frame, [mesh_coords[p] for p in RIGHT_EYE], [mesh_coords[p] for p in LEFT_EYE])
            
            if right_eye is not None and left_eye is not None:
                eye_position_right, color_right = positionEstimator(right_eye)
                eye_position_left, color_left = positionEstimator(left_eye)
                
                cv.putText(frame, f'Right Eye: {eye_position_right}', (30, 90), FONTS, 0.7, (0, 255, 0), 2)
                cv.putText(frame, f'Left Eye: {eye_position_left}', (30, 120), FONTS, 0.7, (0, 255, 0), 2)
        
        # Display FPS on frame
        cv.putText(frame, f'FPS: {round(frame_counter / (time.time() - start_time), 1)}', (30, 150), FONTS, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv.imshow('Frame', frame)

        # Check for 'q' key press to exit
        key = cv.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break

# Release resources
camera.release()
cv.destroyAllWindows()
