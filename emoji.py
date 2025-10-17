import mediapipe as mp  # hand tracking & face mesh
import cv2  # for image processing
import numpy as np

# Haar Cascades for smile detection (keep for smile-trigger)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

# MediaPipe Setup
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

# MediaPipe Face Mesh (for beautiful reliable tracking)
mpFaceMesh = mp.solutions.face_mesh
face_mesh = mpFaceMesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Load emote images
emote = cv2.imread("goblin.webp", cv2.IMREAD_UNCHANGED)
emote1 = cv2.imread("images.png", cv2.IMREAD_UNCHANGED)
emote2 = cv2.imread("cheer.webp", cv2.IMREAD_UNCHANGED)
emote3 = cv2.imread("images.jpg", cv2.IMREAD_UNCHANGED)
emote4 = cv2.imread("laugh.jpg", cv2.IMREAD_UNCHANGED)

if emote is None or emote1 is None or emote2 is None or emote3 is None or emote4 is None:
    print("Error: Could not load emote images! Make sure the files exist!")
    exit()
emote = cv2.resize(emote, (150, 150))
emote1 = cv2.resize(emote1, (150, 150))
emote2 = cv2.resize(emote2, (150, 150))
emote3 = cv2.resize(emote3, (150, 150))
emote4 = cv2.resize(emote4, (150, 150))

def peace_sign(multi_hand_landmarks, height):
    if len(multi_hand_landmarks) != 1:
        return False
    for hand_landmarks in multi_hand_landmarks:
        index_tip = hand_landmarks.landmark[8].y * height
        index_base = hand_landmarks.landmark[5].y * height
        middle_tip = hand_landmarks.landmark[12].y * height
        middle_base = hand_landmarks.landmark[9].y * height
        ring_tip = hand_landmarks.landmark[16].y * height
        ring_base = hand_landmarks.landmark[13].y * height
        pinky_tip = hand_landmarks.landmark[20].y * height
        pinky_base = hand_landmarks.landmark[17].y * height
        index_up = index_tip < index_base
        middle_up = middle_tip < middle_base
        ring_down = ring_tip > ring_base
        pinky_down = pinky_tip > pinky_base
        if index_up and middle_up and ring_down and pinky_down:
            return True
    return False

def both_index_fingers_up(multi_hand_landmarks, height):
    if len(multi_hand_landmarks) != 2:
        return False
    count_up = 0
    for hand_landmarks in multi_hand_landmarks:
        index_tip = hand_landmarks.landmark[8].y * height
        index_base = hand_landmarks.landmark[5].y * height
        if index_tip < index_base:
            count_up += 1
    return count_up == 2

def both_thumbs_up(multi_hand_landmarks, height):
    if len(multi_hand_landmarks) != 2:
        return False
    thumbs_up = 0
    for hand_landmarks in multi_hand_landmarks:
        thumb_tip = hand_landmarks.landmark[4].y * height
        wrist = hand_landmarks.landmark[0].y * height
        if thumb_tip < wrist:
            thumbs_up += 1
    return thumbs_up == 2

def hands_on_cheeks(multi_hand_landmarks, width, height):
    if len(multi_hand_landmarks) != 2:
        return False
    cheek_zone_width = int(width * 0.15)
    cheek_zone_height = int(height * 0.25)
    left_cheek_x = int(width * 0.18)
    right_cheek_x = int(width * 0.82)
    cheek_y = int(height * 0.48)
    cheeks_covered = 0
    fingers_to_check = [4, 8]
    for hand_landmarks in multi_hand_landmarks:
        for idx in fingers_to_check:
            x = int(hand_landmarks.landmark[idx].x * width)
            y = int(hand_landmarks.landmark[idx].y * height)
            if (left_cheek_x < x < left_cheek_x + cheek_zone_width and
                cheek_y < y < cheek_y + cheek_zone_height):
                cheeks_covered += 1
                break
            if (right_cheek_x < x < right_cheek_x + cheek_zone_width and
                cheek_y < y < cheek_y + cheek_zone_height):
                cheeks_covered += 1
                break
    return cheeks_covered >= 2

def overlay_emote(frame, emote_img):
    h, w = frame.shape[:2]
    eh, ew = emote_img.shape[:2]
    x = (w - ew) // 2
    y = (h - eh) // 2
    if h == 0 or w == 0 or eh == 0 or ew == 0 or y < 0 or x < 0 or y + eh > h or x + ew > w:
        return frame
    if len(emote_img.shape) == 3 and emote_img.shape[2] == 4:
        alpha = emote_img[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+eh, x:x+ew, c] = (
                alpha * emote_img[:, :, c] +
                (1 - alpha) * frame[y:y+eh, x:x+ew, c]
            )
    else:
        frame[y:y+eh, x:x+ew] = emote_img[:, :, :3]
    return frame

a = cv2.VideoCapture(0)
if not a.isOpened():
    print("Error: Could not open webcam!")
    exit()

show_emote = False
emote_timer = 0
emote_img_to_show = None
emote_duration = 60

print("🎮 Clash Royale Gesture Emote Tool Started!")
print("👐 Put both hands on cheeks, show both thumbs up, or both index fingers up to trigger an emote")
print("😄 Smile to trigger laugh emote!")
print("❌ Press 'q' to quit")

while True:
    success, img = a.read()
    if not success or img is None:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    smile_detected = False
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = smile_cascade.detectMultiScale(roi_gray, 1.8, 20)
        if len(smiles) > 0:
            smile_detected = True
            break

    if smile_detected:
        show_emote = True
        emote_timer = emote_duration
        emote_img_to_show = emote4

    height, width = img.shape[:2]
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # -------- FACE MESH VISUALIZATION BLOCK --------
    results_face_mesh = face_mesh.process(imgRGB)
    if results_face_mesh.multi_face_landmarks:
        for face_landmarks in results_face_mesh.multi_face_landmarks:
            # Blue mesh lines + bold pink dots
            mpDraw.draw_landmarks(
                img, face_landmarks, mpFaceMesh.FACEMESH_TESSELATION,
                mpDraw.DrawingSpec(color=(20, 150, 255), thickness=1),
                mpDraw.DrawingSpec(color=(255, 20, 150), thickness=2, circle_radius=2)
            )
            # Big yellow/cyan dots at key places (left cheek, right cheek, chin, nose tip, etc.)
            for idx in [1, 33, 61, 199, 263, 291]:
                x = int(face_landmarks.landmark[idx].x * width)
                y = int(face_landmarks.landmark[idx].y * height)
                cv2.circle(img, (x, y), 7, (0, 255, 255), -1)
    # -------- END FACE MESH BLOCK --------

    result = hands.process(imgRGB)

    # Cheek zones
    cheek_zone_width = int(width * 0.15)
    cheek_zone_height = int(height * 0.25)
    left_cheek_x = int(width * 0.18)
    right_cheek_x = int(width * 0.82)
    cheek_y = int(height * 0.48)
    cv2.rectangle(img, (left_cheek_x, cheek_y),
                  (left_cheek_x + cheek_zone_width, cheek_y + cheek_zone_height), (0, 255, 0), 2)
    cv2.rectangle(img, (right_cheek_x, cheek_y),
                  (right_cheek_x + cheek_zone_width, cheek_y + cheek_zone_height), (0, 255, 0), 2)

    # Hand landmark drawing with style
    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS,
                mpDraw.DrawingSpec(color=(208, 84, 0), thickness=2, circle_radius=5),
                mpDraw.DrawingSpec(color=(84, 208, 0), thickness=2, circle_radius=7))
            for idx in [4, 8, 12, 16, 20]:
                x = int(handLms.landmark[idx].x * width)
                y = int(handLms.landmark[idx].y * height)
                cv2.circle(img, (x, y), 10, (0, 255, 255), -1)

        # Gesture priorities
        if hands_on_cheeks(result.multi_hand_landmarks, width, height):
            show_emote = True
            emote_timer = emote_duration
            emote_img_to_show = emote
        elif both_thumbs_up(result.multi_hand_landmarks, height):
            show_emote = True
            emote_timer = emote_duration
            emote_img_to_show = emote1
        elif both_index_fingers_up(result.multi_hand_landmarks, height):
            show_emote = True
            emote_timer = emote_duration
            emote_img_to_show = emote2
        elif peace_sign(result.multi_hand_landmarks, height):
            show_emote = True
            emote_timer = emote_duration
            emote_img_to_show = emote3

    if show_emote and emote_timer > 0 and emote_img_to_show is not None:
        img = overlay_emote(img, emote_img_to_show)
        emote_timer -= 1
        if emote_timer <= 0:
            show_emote = False
            emote_img_to_show = None

    if result.multi_hand_landmarks:
        hands_count = len(result.multi_hand_landmarks)
        cv2.putText(img, f"Hands detected: {hands_count}", (10, 70),
                    cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 2)

    if img is not None and img.shape[0] > 0 and img.shape[1] > 0:
        cv2.imshow("Clash Royale Emote Tool", img)
    else:
        print("Invalid frame: cannot display")
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

a.release()
cv2.destroyAllWindows()
print("👋 Thanks for using the Clash Royale Emote Tool!")
