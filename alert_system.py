import pygame
import time
import os
import cv2
import numpy as np
from utils import put_text_vietnamese, get_text_dimensions

class AlertSystem:
    
    # Hệ thống cảnh báo âm thanh và hiển thị
    
    def __init__(self, sound_path=None, cooldown=5):
        """
            sound_path: đường dẫn file âm thanh
            cooldown: thời gian delay giữa các lần cảnh báo (giây)
        """
        self.sound_path = sound_path
        self.cooldown = cooldown
        self.last_alert_time = 0
        
        # Khởi tạo pygame mixer cho âm thanh
        pygame.mixer.init()
        
        # Tạo file âm thanh mặc định nếu chưa có
        if sound_path and not os.path.exists(sound_path):
            self._create_default_sound()
        
        # Load âm thanh
        if sound_path and os.path.exists(sound_path):
            try:
                self.alert_sound = pygame.mixer.Sound(sound_path)
            except:
                print(f"Không thể load file âm thanh: {sound_path}")
                self.alert_sound = None
        else:
            self.alert_sound = None
    
    def _create_default_sound(self):
        
        # Tạo file âm thanh 
        
        try:
            import numpy as np
            from scipy.io import wavfile
            
            # Tạo âm thanh beep 1000Hz trong 0.5 giây
            sample_rate = 44100
            duration = 0.5
            frequency = 1000
            
            t = np.linspace(0, duration, int(sample_rate * duration))
            audio = np.sin(2 * np.pi * frequency * t)
            
            # Fade in/out để tránh tiếng click
            fade_samples = int(sample_rate * 0.01)
            audio[:fade_samples] *= np.linspace(0, 1, fade_samples)
            audio[-fade_samples:] *= np.linspace(1, 0, fade_samples)
            
            # Chuyển sang int16
            audio = (audio * 32767).astype(np.int16)
            
            # Lưu file
            wavfile.write(self.sound_path, sample_rate, audio)
            print(f"Đã tạo file âm thanh mặc định: {self.sound_path}")
            
        except Exception as e:
            print(f"Không thể tạo file âm thanh: {e}")
    
    def can_alert(self):
        
        # Kiểm tra xem có thể phát cảnh báo không (dựa trên cooldown)
        
        current_time = time.time()
        if current_time - self.last_alert_time >= self.cooldown:
            return True
        return False
    
    def play_sound(self):

        if self.alert_sound and self.can_alert():
            try:
                self.alert_sound.play()
                self.last_alert_time = time.time()
            except Exception as e:
                print(f"⚠️ Lỗi khi phát âm thanh: {e}")
    
    def trigger_alert(self, alert_type="general"):
            
        # alert_type: loại cảnh báo (drowsy, distracted, general)
        
        if self.can_alert():
            self.play_sound()
            return True
        return False
    
    def reset_cooldown(self):
        
        self.last_alert_time = 0


class VisualAlertOverlay:

    # Overlay hiển thị cảnh báo trực quan trên màn hình
    
    def __init__(self, frame_width, frame_height):
        
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.alerts = []
        
    def add_alert(self, message, color=(0, 0, 255), duration=2.0):
       
        alert = {
            'message': message,
            'color': color,
            'start_time': time.time(),
            'duration': duration
        }
        self.alerts.append(alert)
    
    def render(self, frame):
        
        current_time = time.time()
        active_alerts = []

        if not self.alerts:
            return 
        
        font_size = 30 # Cố định cỡ chữ
        padding_x = 20 # Khoảng đệm ngang
        padding_y = 10 # Khoảng đệm dọc

        for i, alert in enumerate(self.alerts):
            elapsed = current_time - alert['start_time']
            
            if elapsed < alert['duration']:
                # Vị trí hiển thị (xếp chồng các cảnh báo)
                y_center = 100 + (i * 60) # Tăng khoảng cách giữa các dòng lên 60
                
                # Lấy chiều rộng và cao thực tế của đoạn text
                text_w, text_h = get_text_dimensions(alert['message'], font_size=font_size)
                
                # Tính toán tọa độ khung nền
                center_x = self.frame_width // 2
                
                # Tọa độ trái-trên (Top-Left) và phải-dưới (Bottom-Right) của khung
                x1 = center_x - (text_w // 2) - padding_x
                y1 = y_center - (text_h // 2) - padding_y
                x2 = center_x + (text_w // 2) + padding_x
                y2 = y_center + (text_h // 2) + padding_y

                # Vẽ background màu đen
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
                
                # Vẽ viền màu (optional - cho đẹp)
                cv2.rectangle(frame, (x1, y1), (x2, y2), alert['color'], 2)
                
                # Vẽ text tiếng Việt
                # Lưu ý vị trí vẽ text phải trừ đi nửa chiều rộng/cao để căn giữa
                text_pos = (center_x - text_w // 2, y_center - text_h // 2 - 5) # -5 để căn chỉnh visual
                
                frame[:] = put_text_vietnamese(
                    frame, 
                    alert['message'], 
                    text_pos,
                    font_size=font_size, 
                    color=alert['color']
                )
                
                active_alerts.append(alert)
        
        self.alerts = active_alerts