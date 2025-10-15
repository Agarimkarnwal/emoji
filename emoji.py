import mediapipe as mp #used for hand tracking
import cv2 #used for image processing
import numpy as np

# Initialize MediaPipe
mpHands = mp.solutions.hands #basically this 
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

# Load emote images
emote = cv2.imread("goblin.webp", cv2.IMREAD_UNCHANGED)
emote1 = cv2.imread("images.png", cv2.IMREAD_UNCHANGED)
emote2 = cv2.imread("cheer.webp", cv2.IMREAD_UNCHANGED)
emote3=cv2.imread("images.jpg", cv2.IMREAD_UNCHANGED)

if emote is None or emote1 is None or emote2 is None or emote3 is None:
    print("Error: Could not load emote images! Make sure the files exist!")
    exit()

# Resize emotes for better display
emote = cv2.resize(emote, (150, 150))
emote1 = cv2.resize(emote1, (150, 150))
emote2 = cv2.resize(emote2, (150, 150))
emote3=cv2.resize(emote3,(150,150))

def peace_sign(multi_hand_landmarks,height):
    if len(multi_hand_landmarks)!=1:
        return False
    fingers_up=0
    for hand_landmarks in multi_hand_landmarks:
        index_tip=hand_landmarks.landmark[8].y*height
        index_base=hand_landmarks.landmark[5].y*height
        middle_finger_tip=hand_landmarks.landmark[12].y*height
        middle_finger_base=hand_landmarks.landmark[9].y*height
        ring_finger_tip=hand_landmarks.landmark[16].y*height
        ring_finger_base=hand_landmarks.landmark[13].y*height
        pinky_finger_tip=hand_landmarks.landmark[20].y*height
        pinky_finger_base=hand_landmarks.landmark[17].y*height

        index_up=index_tip<index_base
        middle_up=middle_finger_tip<middle_finger_base
        ring_down=ring_finger_tip>ring_finger_base
        pinky_down=pinky_finger_tip>pinky_finger_base

        if index_up and middle_up and ring_down and pinky_down:
            return True
        



        if middle_finger_tip<middle_finger_base:
            fingers_up+=1
        
    return fingers_up==2



def both_index_fingers_up(multi_hand_landmarks, height):
    if len(multi_hand_landmarks) != 2:
        return False
    count_up = 0
    for hand_landmarks in multi_hand_landmarks:
        index_tip_y = hand_landmarks.landmark[8].y * height
        index_base_y = hand_landmarks.landmark[5].y * height
        if index_tip_y < index_base_y:  # Tip is higher, finger is up
            count_up += 1
    return count_up == 2

def both_thumbs_up(multi_hand_landmarks, height):
    if len(multi_hand_landmarks) != 2:
        return False
    thumbs_up = 0
    for hand_landmarks in multi_hand_landmarks:
        thumb_tip_y = hand_landmarks.landmark[4].y * height
        wrist_y = hand_landmarks.landmark[0].y * height
        if thumb_tip_y < wrist_y:  # Tip above wrist, thumb up
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
    if y + eh > h or x + ew > w:
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
print("❌ Press 'q' to quit")

while True:
    success, img = a.read()
    if not success:
        print("Failed to grab frame")
        break
    height, width = img.shape[:2]
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)

    cheek_zone_width = int(width * 0.15)
    cheek_zone_height = int(height * 0.25)
    left_cheek_x = int(width * 0.18)
    right_cheek_x = int(width * 0.82)
    cheek_y = int(height * 0.48)
    cv2.rectangle(img, (left_cheek_x, cheek_y),
                  (left_cheek_x + cheek_zone_width, cheek_y + cheek_zone_height),
                  (0,255,0), 2)
    cv2.rectangle(img, (right_cheek_x, cheek_y),
                  (right_cheek_x + cheek_zone_width, cheek_y + cheek_zone_height),
                  (0,255,0), 2)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
        # Priority: cheeks > thumbs up > index fingers>peace sign
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
        elif peace_sign(result.multi_hand_landmarks,height):
            show_emote = True
            emote_timer = emote_duration
            emote_img_to_show = emote3

    # Only ONE emote shown per frame (priority order)
    if show_emote and emote_timer > 0 and emote_img_to_show is not None:
        img = overlay_emote(img, emote_img_to_show)
        emote_timer -= 1
        if emote_timer <= 0:
            show_emote = False
            emote_img_to_show = None

    if result.multi_hand_landmarks:
        hands_count = len(result.multi_hand_landmarks)
        cv2.putText(img, f"Hands detected: {hands_count}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv2.imshow("Clash Royale Emote Tool", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

a.release()
cv2.destroyAllWindows()
print("👋 Thanks for using the Clash Royale Emote Tool!")
