# Ngưỡng Eye Aspect Ratio để phát hiện nhắm mắt
EAR_THRESHOLD = 0.25

# Số frame liên tiếp mắt nhắm để coi là buồn ngủ (3 giây với 30fps = 90 frames)
EAR_CONSEC_FRAMES = 90

# Số frame nhìn lệch liên tiếp để coi là mất tập trung (15 giây = 450 frames)
GAZE_CONSEC_FRAMES = 450

# Ngưỡng độ lệch hướng nhìn (góc độ)
GAZE_THRESHOLD = 25

# CẤU HÌNH CAMERA
CAMERA_INDEX = 0
FRAME_WIDTH = 1024
FRAME_HEIGHT = 600
FPS = 30

ALERT_SOUND_PATH = "alert.wav"

# Thời gian delay giữa các lần cảnh báo (giây)
ALERT_COOLDOWN = 5

LOG_FILE = "logs/activity_log.csv"
REPORT_OUTPUT = "reports/"

# Confidence threshold cho face detection
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

# Chỉ số landmarks cho mắt trái
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]

# Chỉ số landmarks cho mắt phải
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# Chỉ số cho mũi (dùng tính hướng nhìn)
NOSE_TIP_INDEX = 1
CHIN_INDEX = 152
LEFT_EYE_CORNER_INDEX = 33
RIGHT_EYE_CORNER_INDEX = 263
LEFT_MOUTH_INDEX = 61
RIGHT_MOUTH_INDEX = 291
