from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy.spatial import distance as dist
import cv2
import os
from datetime import datetime

def calculate_ear(eye_landmarks):
    """
    Tính Eye Aspect Ratio (EAR)
    
    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    
    Với eye_landmarks là 6 điểm theo thứ tự:
    p1: góc mắt trái
    p2, p3: điểm trên mí trên
    p4: góc mắt phải
    p5, p6: điểm dưới mí dưới
    
    Args:
        eye_landmarks: numpy array shape (6, 2) chứa tọa độ 6 điểm
        
    Returns:
        float: giá trị EAR
    """
    # Tính khoảng cách dọc
    vertical_1 = dist.euclidean(eye_landmarks[1], eye_landmarks[5])
    vertical_2 = dist.euclidean(eye_landmarks[2], eye_landmarks[4])
    
    # Tính khoảng cách ngang
    horizontal = dist.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    # Tính EAR
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    
    return ear


def extract_eye_landmarks(face_landmarks, eye_indices, img_width, img_height):
    """
    Trích xuất tọa độ landmarks của mắt
    
    Args:
        face_landmarks: MediaPipe face landmarks
        eye_indices: list các chỉ số landmarks của mắt
        img_width: chiều rộng ảnh
        img_height: chiều cao ảnh
        
    Returns:
        numpy array: tọa độ các điểm (n, 2)
    """
    landmarks = []
    for idx in eye_indices:
        landmark = face_landmarks.landmark[idx]
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        landmarks.append([x, y])
    
    return np.array(landmarks, dtype=np.int32)


def calculate_gaze_direction(face_landmarks, img_width, img_height):
    """
    Tính hướng nhìn dựa trên vị trí các landmarks
    
    Args:
        face_landmarks: MediaPipe face landmarks
        img_width: chiều rộng ảnh
        img_height: chiều cao ảnh
        
    Returns:
        tuple: (horizontal_angle, vertical_angle)
    """
    # Lấy các điểm quan trọng
    nose_tip = face_landmarks.landmark[1]
    chin = face_landmarks.landmark[152]
    left_eye = face_landmarks.landmark[33]
    right_eye = face_landmarks.landmark[263]
    
    # Chuyển sang tọa độ pixel
    nose_x = nose_tip.x * img_width
    left_eye_x = left_eye.x * img_width
    right_eye_x = right_eye.x * img_width
    
    # Tính điểm giữa hai mắt
    eye_center_x = (left_eye_x + right_eye_x) / 2
    
    # Tính góc ngang (horizontal angle)
    # Nếu mũi lệch sang trái -> nhìn trái (âm)
    # Nếu mũi lệch sang phải -> nhìn phải (dương)
    horizontal_diff = nose_x - eye_center_x
    face_width = abs(right_eye_x - left_eye_x)
    horizontal_angle = (horizontal_diff / face_width) * 50 if face_width > 0 else 0
    
    # Tính góc dọc (vertical angle) - đơn giản hóa
    nose_y = nose_tip.y * img_height
    chin_y = chin.y * img_height
    vertical_angle = (nose_y - chin_y) / img_height * 50
    
    return horizontal_angle, vertical_angle


def is_looking_away(horizontal_angle, vertical_angle, threshold=25):
    """
    Kiểm tra xem người dùng có đang nhìn lệch không
    
    Args:
        horizontal_angle: góc ngang
        vertical_angle: góc dọc
        threshold: ngưỡng góc
        
    Returns:
        bool: True nếu nhìn lệch
    """
    return abs(horizontal_angle) > threshold or abs(vertical_angle) > threshold


def draw_eye_landmarks(frame, landmarks, color=(0, 255, 0)):
    """
    Vẽ landmarks của mắt lên frame
    
    Args:
        frame: khung hình
        landmarks: tọa độ các điểm
        color: màu vẽ (BGR)
    """
    for point in landmarks:
        cv2.circle(frame, tuple(point), 2, color, -1)
    
    # Vẽ đường viền mắt
    cv2.polylines(frame, [landmarks], True, color, 1)


def create_status_overlay(frame, status_text, color, position=(10, 30)):
    """
    Tạo overlay hiển thị trạng thái
    
    Args:
        frame: khung hình
        status_text: text hiển thị
        color: màu chữ
        position: vị trí hiển thị
    """
    cv2.putText(frame, status_text, position, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def ensure_dir(directory):
    """
    Đảm bảo thư mục tồn tại, nếu không thì tạo mới
    
    Args:
        directory: đường dẫn thư mục
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_timestamp():
    """
    Lấy timestamp hiện tại
    
    Returns:
        str: timestamp định dạng YYYY-MM-DD HH:MM:SS
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_duration(seconds):
    """
    Định dạng thời gian từ giây sang HH:MM:SS
    
    Args:
        seconds: số giây
        
    Returns:
        str: chuỗi định dạng
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def _load_font(font_size):
    # Hàm nội bộ để load font tránh lặp code
    try:
        # Thử load font Arial (Windows/Linux phổ biến)
        return ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            # Thử đường dẫn mặc định của Linux nếu cần
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        except:
            # Fallback về mặc định
            return ImageFont.load_default()

def get_text_dimensions(text, font_size=32):
    
    # Tính toán kích thước chính xác của text (width, height)
    
    font = _load_font(font_size)
    
    # Sử dụng getbbox để lấy khung bao quanh text
    # getbbox trả về (left, top, right, bottom)
    bbox = font.getbbox(text)
    
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    # Cộng thêm một chút height cho các ký tự có dấu (như ê, ắ) hoặc đuôi (g, y)
    return width, height + 5

def put_text_vietnamese(img, text, position, font_size=32, color=(0, 0, 255)):
    
    # Hàm vẽ tiếng Việt lên ảnh OpenCV
    
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = _load_font(font_size)
    
    rgb_color = (color[2], color[1], color[0]) 
    draw.text(position, text, font=font, fill=rgb_color)
    
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)