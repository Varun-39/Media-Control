import cv2
import mediapipe as mp
import time
import math
import threading
from pynput.keyboard import Key, Controller

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
GESTURE_COOLDOWN = 1.0

# ------------------ One Euro Filter ------------------
class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.01):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    def smoothing_factor(self, dt, cutoff):
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def filter(self, t, x):
        dt = t - self.t_prev
        if dt <= 0:
            return self.x_prev

        dx = (x - self.x_prev) / dt
        cutoff = self.min_cutoff + self.beta * abs(dx)
        a = self.smoothing_factor(dt, cutoff)

        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev = x_hat
        self.t_prev = t
        return x_hat

# ------------------ Threaded Camera ------------------
class ThreadedCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(3, CAMERA_WIDTH)
        self.cap.set(4, CAMERA_HEIGHT)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        threading.Thread(target=self.update, daemon=True).start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

# ------------------ Main ------------------
# ------------------ Main ------------------
mp_hands = mp.solutions.hands
# Optimize for Jetson Nano: model_complexity=0 (Lite), min_detection_confidence=0.5
hands = mp_hands.Hands(
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
keyboard = Controller()
camera = ThreadedCamera()

vol_filter = OneEuroFilter(time.time(), 0.5)
previous_y = 0.5
last_gesture_time = 0

# Performance tracking
prev_frame_time = 0
new_frame_time = 0

print("✋ Open = Play | ✊ Fist = Pause | ✌️ Peace = Mute | 🤏 Pinch = Volume")

while True:
    success, frame = camera.read()
    if not success:
        break
    
    # Measure Latency: Start
    process_start_time = time.time()

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    h, w, _ = frame.shape
    current_time = time.time()

    # Calculate FPS
    new_frame_time = current_time
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps_text = f"FPS: {int(fps)}"
    
    # Calculate Processing Latency (ms)
    latency = (time.time() - process_start_time) * 1000
    latency_text = f"Lat: {int(latency)}ms"

    # Display Performance Metrics
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, latency_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            lm = hand.landmark

            # -------- Finger detection --------
            fingers = []
            for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
                fingers.append(lm[tip].y < lm[pip].y)

            thumb = lm[4].x < lm[3].x
            fingers.insert(0, thumb)

            total = sum(fingers)

            # -------- Pinch --------
            pinch_dist = math.hypot(lm[4].x - lm[8].x, lm[4].y - lm[8].y)
            hand_scale = math.hypot(lm[0].x - lm[5].x, lm[0].y - lm[5].y)
            pinch_ratio = pinch_dist / hand_scale

            # -------- Volume Control --------
            if pinch_ratio < 0.4:
                y = lm[8].y
                smooth_y = vol_filter.filter(current_time, y)
                delta = previous_y - smooth_y

                if abs(delta) > 0.015:
                    if delta > 0:
                        keyboard.tap(Key.media_volume_up)
                        cv2.putText(frame, "VOL UP", (50,50), 1, 1.5, (0,255,0),2)
                    else:
                        keyboard.tap(Key.media_volume_down)
                        cv2.putText(frame, "VOL DOWN", (50,50), 1, 1.5, (0,0,255),2)
                    previous_y = smooth_y

            # -------- Gestures --------
            elif current_time - last_gesture_time > GESTURE_COOLDOWN:

                if total == 5:
                    keyboard.tap(Key.media_play_pause)
                    last_gesture_time = current_time
                    cv2.putText(frame, "PLAY", (50,50), 1, 2, (0,255,0),2)

                elif total == 0:
                    keyboard.tap(Key.media_play_pause)
                    last_gesture_time = current_time
                    cv2.putText(frame, "PAUSE", (50,50), 1, 2, (0,0,255),2)

                elif fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
                    keyboard.tap(Key.media_volume_mute)
                    last_gesture_time = current_time
                    cv2.putText(frame, "MUTE", (50,50), 1, 2, (255,255,0),2)

    cv2.imshow("Touchless Media Controller", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

camera.stop()
cv2.destroyAllWindows()
