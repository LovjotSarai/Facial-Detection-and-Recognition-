# facial_and_hand_recognition.py

import cv2
import numpy as np
from database_connector import create_db_connection, download_images

# Replace with your actual database info
host = "your_host"
user = "your_username"
password = "your_password"
database = "your_database_name"

def hand_gesture_recognition(frame, background):
    """Detect hand using skin color segmentation and count the number of fingers raised."""
    # Convert image to YCrCb
    image_ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCR_CB)
    
    # Define range for skin color
    min_YCrCb = np.array([0, 133, 77], np.uint8)
    max_YCrCb = np.array([255, 173, 127], np.uint8)
    
    # Find region with skin color
    mask = cv2.inRange(image_ycrcb, min_YCrCb, max_YCrCb)
    
    # Do morphological transformations
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    
    # Find contours
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find contour of max area(hand)
    if contours:
        cnt = max(contours, key=lambda x: cv2.contourArea(x))
        
        # Approximate the contour a little
        epsilon = 0.0005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        
        # Make convex hull around hand
        hull = cv2.convexHull(cnt)
        
        # Define area of hull and area of hand
        area_hull = cv2.contourArea(hull)
        area_cnt = cv2.contourArea(cnt)
      
        # Find the percentage of area not covered by hand in convex hull
        area_ratio = ((area_hull - area_cnt) / area_cnt) * 100
    
        # Find the defects in convex hull with respect to hand
        hull = cv2.convexHull(approx, returnPoints=False)
        defects = cv2.convexityDefects(approx, hull)
        
        # Count defects
        l = 0
        
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(approx[s][0])
            end = tuple(approx[e][0])
            far = tuple(approx[f][0])
            pt = (100, 180)
            
            # Find length of all sides of triangle
            a = np.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
            b = np.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
            c = np.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
            s = (a + b + c) / 2
            ar = np.sqrt(s * (s - a) * (s - b) * (s - c))
            
            # Distance between point and convex hull
            d = (2 * ar) / a
            
            # Apply cosine rule
            angle = np.arccos((b**2 + c**2 - a**2) / (2 * b * c)) * 57
            
            # Ignore angles > 90 and ignore points very close to convex hull(they generally come due to noise)
            if angle <= 90 and d > 30:
                l += 1
                cv2.circle(frame, far, 3, [255, 0, 0], -1)
            
            # Draw lines around hand
            cv2.line(frame, start, end, [0, 255, 0], 2)
            
        l += 1
        
        # Print number of fingers
        font = cv2.FONT_HERSHEY_SIMPLEX
        if l == 1:
            if area_cnt < 2000:
                cv2.putText(frame, 'Put hand in the box', (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
            else:
                if area_ratio < 12:
                    cv2.putText(frame, '0', (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                elif area_ratio < 17.5:
                    cv2.putText(frame, 'Best of luck', (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    
                else:
                    cv2.putText(frame, str(l), (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
                    
        elif l > 1:
            cv2.putText(frame, str(l), (0, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
    
    return frame

def facial_recognition(download_images_flag=False):
    """Perform facial recognition and hand gesture recognition."""
    if download_images_flag:
        connection = create_db_connection(host, user, password, database)
        download_images(connection)
    
    face_cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_path)
    video_capture = cv2.VideoCapture(0)
    background = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Call the hand gesture recognition function and get the processed frame
        frame_with_gestures = hand_gesture_recognition(frame, background)

        cv2.imshow('Video', frame_with_gestures)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    facial_recognition(download_images_flag=True)
