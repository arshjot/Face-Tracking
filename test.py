import cv2
from face_tracking import FaceTracking
from config import Config

config = Config
face = FaceTracking()

input_ = cv2.VideoCapture(0) if config.use_webcam else cv2.VideoCapture(config.video_path)

# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (int(input_.get(3)), int(input_.get(4))))

while True:
    # We get a new frame from the webcam
    ret, frame = input_.read()
    if ret:

        # We send this frame to GazeTracking to analyze it
        face.refresh(frame)

        frame = face.annotated_frame()
        eyes_text, face_check_text, face_away_text = "", "", ""

        if face.eyes_are_blinking():
            eyes_text = "Eyes: Blinking"
        elif face.eyes_are_right():
            eyes_text = "Eyes: Right"
        elif face.eyes_are_left():
            eyes_text = "Eyes: Left"
        elif face.eyes_are_center():
            eyes_text = "Eyes: Center"

        if face.face_check():
            face_check_text = face.face_check()
        if face.face_is_looking_away():
            face_away_text = "Looking Away"

        cv2.putText(frame, face_check_text, (300, 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 128), 2)
        cv2.putText(frame, face_away_text, (300, 60), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 128), 2)

        cv2.putText(frame, eyes_text, (15, 20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 2)

        if face.face_located:
            yaw = str(round(face.yaw_predicted, 0))
            pitch = str(round(face.pitch_predicted, 0))
            roll = str(round(face.roll_predicted, 0))
            cv2.putText(frame, "Yaw  : " + yaw + " deg", (15, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 2)
            cv2.putText(frame, "Pitch: " + pitch + " deg", (15, 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 2)
            cv2.putText(frame, "Roll : " + roll + " deg", (15, 110), cv2.FONT_HERSHEY_DUPLEX, 0.8, (147, 58, 31), 2)

        # out.write(frame)
        cv2.imshow("Demo", frame)

        if cv2.waitKey(1) == 27:
            break

    else:
        break

input_.release()
cv2.destroyAllWindows()
# out.release()