import cv2
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

cap = cv2.VideoCapture("single.mp4")

left_counter = 0
right_counter = 0
left_stage = None
right_stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = cv2.resize(image, (600, 500))

        try:
            landmarks = results.pose_landmarks.landmark
            
            # Left arm landmarks
            left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            left_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            # Calculate angle for left arm
            left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
            
            cv2.putText(image, f'Left Angle: {int(left_angle)}', 
                        tuple(np.multiply(left_elbow, [500, 600]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 153, 255), 1, cv2.LINE_AA)

            # Right arm landmarks
            right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                              landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
            right_elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
            right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

            # Calculate angle for right arm
            right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
            
            cv2.putText(image, f'Right Angle: {int(right_angle)}', 
                        tuple(np.multiply(right_elbow, [500, 600]).astype(int)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (102, 153, 255), 1, cv2.LINE_AA)
            
            # Left arm push-up counting
            if left_angle > 160:
                left_stage = "down"
            if left_angle < 30 and left_stage == 'down':
                left_stage = "up"
                left_counter += 1

            # Right arm push-up counting
            if right_angle > 160:
                right_stage = "down"
            if right_angle < 30 and right_stage == 'down':
                right_stage = "up"
                right_counter += 1

        except:
            pass

        cv2.putText(image, f'Left Reps: {left_counter}', (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 204, 102), 1, cv2.LINE_AA)
        cv2.putText(image, f'Right Reps: {right_counter}', (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 102), 1, cv2.LINE_AA)

        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
