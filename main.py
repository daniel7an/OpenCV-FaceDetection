import cv2
import mediapipe as mp
import time

# Initializing Face Detection Classes
mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
faceDetection = mpFaceDetection.FaceDetection(0.75)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
pTime = 0
while True:
    ret, frame = cap.read()
    # Converting BGR To RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Processing Frame
    results = faceDetection.process(rgb_frame)
    if results.detections:
        for id, detection in enumerate(results.detections):
            # Function For Drawing Rectangle Automaticly
            #mpDraw.draw_detection(frame, detection)
            # Drawing Rectangle Manually
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, ic = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (bbox), (255, 0, 255), 2)
            cv2.putText(frame, f'{int(detection.score[0]*100)}%', (bbox[0], bbox[1]-20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)


    # Calculating Capture FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(frame, f'FPS: {str(int(fps))}', (10, 130), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    if ret:
        cv2.imshow('Frame', frame)
    else:
        print(str(ret))

    key = cv2.waitKey(20)
    if key == 81 or key == 137:
        break
cap.release()
cv2.destroyAllWindows()




