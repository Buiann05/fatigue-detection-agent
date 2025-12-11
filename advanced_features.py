import cv2
import numpy as np
from datetime import datetime, timedelta


class YawnDetector:
    
    # Phát hiện ngáp dựa trên khoảng cách miệng
    
    def __init__(self, threshold=0.6):
        self.threshold = threshold
        self.yawn_counter = 0
        self.total_yawns = 0
    
    def calculate_mouth_aspect_ratio(self, mouth_landmarks):
        """
        Tính MAR (Mouth Aspect Ratio)
        
        MAR = (||p2-p8|| + ||p3-p7|| + ||p4-p6||) / (2 * ||p1-p5||)
        """
        # Tính khoảng cách dọc
        vertical_1 = np.linalg.norm(mouth_landmarks[1] - mouth_landmarks[7])
        vertical_2 = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[6])
        vertical_3 = np.linalg.norm(mouth_landmarks[3] - mouth_landmarks[5])
        
        # Tính khoảng cách ngang
        horizontal = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[4])
        
        mar = (vertical_1 + vertical_2 + vertical_3) / (2.0 * horizontal)
        return mar
    
    def detect_yawn(self, mouth_landmarks):
        
        # Phát hiện ngáp
        
        mar = self.calculate_mouth_aspect_ratio(mouth_landmarks)
        
        is_yawning = False
        
        if mar > self.threshold:
            self.yawn_counter += 1
            if self.yawn_counter > 15:  
                is_yawning = True
                self.total_yawns += 1
                self.yawn_counter = 0
        else:
            self.yawn_counter = 0
        
        return is_yawning


class HeadPoseEstimator:
    
    # Ước lượng nghiêng và quay đầu
    
    def __init__(self):
        self.model_points = np.array([
            (0.0, 0.0, 0.0),             # Mũi
            (0.0, -330.0, -65.0),        # Cằm
            (-225.0, 170.0, -135.0),     # Mắt trái
            (225.0, 170.0, -135.0),      # Mắt phải
            (-150.0, -150.0, -125.0),    # Miệng trái
            (150.0, -150.0, -125.0)      # Miệng phải
        ])
    
    def estimate_pose(self, face_landmarks, img_shape):
        
        # Ước lượng góc quay đầu
        
        # Lấy các landmarks quan trọng
        image_points = np.array([
            face_landmarks.landmark[1],    # Mũi
            face_landmarks.landmark[152],  # Cằm
            face_landmarks.landmark[33],   # Mắt trái
            face_landmarks.landmark[263],  # Mắt phải
            face_landmarks.landmark[61],   # Miệng trái
            face_landmarks.landmark[291]   # Miệng phải
        ])
        
        # Chuyển sang pixel coordinates
        h, w = img_shape[:2]
        image_points_2d = np.array([
            [p.x * w, p.y * h] for p in image_points
        ], dtype=np.float64)
        
        # Camera matrix (ước lượng)
        focal_length = w
        center = (w/2, h/2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # Distortion coefficients (giả sử không có distortion)
        dist_coeffs = np.zeros((4, 1))
        
        # Giải PnP
        success, rotation_vector, translation_vector = cv2.solvePnP(
            self.model_points,
            image_points_2d,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        
        if not success:
            return 0, 0, 0
        
        # Chuyển rotation vector sang rotation matrix
        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        
        # Tính Euler angles
        pose_mat = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
        
        pitch, yaw, roll = euler_angles.flatten()[:3]
        
        return pitch, yaw, roll


class BreakReminder:
    
    # Nhắc nhở nghỉ giải lao định kỳ
    
    def __init__(self, interval_minutes=25):
        
        # interval_minutes: số phút giữa các lần nhắc (25)

        self.interval = timedelta(minutes=interval_minutes)
        self.last_break_time = datetime.now()
        self.breaks_taken = 0
    
    def check_break_time(self):
        
        # Kiểm tra xem đã đến giờ nghỉ chưa
        
        elapsed = datetime.now() - self.last_break_time
        
        if elapsed >= self.interval:
            return True
        return False
    
    def take_break(self):
        
        # Đánh dấu đã nghỉ
        
        self.last_break_time = datetime.now()
        self.breaks_taken += 1
    
    def get_time_until_break(self):
        
        # Tính thời gian còn lại đến giờ nghỉ
        
        elapsed = datetime.now() - self.last_break_time
        remaining = self.interval - elapsed
        
        if remaining.total_seconds() < 0:
            return timedelta(0)
        
        return remaining


class FocusScoreCalculator:
    
    # Tính điểm tập trung theo thời gian
    
    def __init__(self, window_size=60):
        
        # window_size: kích thước cửa sổ tính điểm (giây)

        self.window_size = window_size
        self.events = []  # [(timestamp, is_focused)]
    
    def add_event(self, is_focused):
        
        self.events.append((datetime.now(), is_focused))
        
        # Xóa events cũ hơn window
        cutoff = datetime.now() - timedelta(seconds=self.window_size)
        self.events = [(t, f) for t, f in self.events if t > cutoff]
    
    def get_focus_score(self):
        
        # Tính điểm tập trung (0-100)
        
        if not self.events:
            return 100.0
        
        focused_count = sum(1 for _, is_focused in self.events if is_focused)
        score = (focused_count / len(self.events)) * 100
        
        return score
    
    def get_grade(self):
        
        # Chuyển điểm sang xếp loại
        
        score = self.get_focus_score()
        
        if score >= 95:
            return "A+"
        elif score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"