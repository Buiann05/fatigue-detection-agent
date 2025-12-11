import cv2
import mediapipe as mp
import numpy as np
import time

from utils import (
    calculate_ear, 
    extract_eye_landmarks, 
    calculate_gaze_direction,
    is_looking_away,
    draw_eye_landmarks,
    create_status_overlay
)

import config


class FatigueDetector:
    
    # Detector chính để phát hiện mệt mỏi và mất tập trung
    
    def __init__(self):

        # Khởi tạo MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE
        )
        
        # Bộ đếm frames
        self.eye_closed_counter = 0
        self.gaze_away_counter = 0
        
        # Trạng thái
        self.is_drowsy = False
        self.is_distracted = False
        
        # Lịch sử EAR để làm mượt
        self.ear_history = []
        self.history_size = 5
        
        # Thời gian
        self.last_drowsy_time = 0
        self.last_distracted_time = 0
        
        # Thống kê
        self.total_blinks = 0
        self.last_blink_time = 0
    
    def process_frame(self, frame):
        
        # Xử lý một frame
        
        # Chuyển BGR sang RGB cho MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Xử lý
        results = self.face_mesh.process(rgb_frame)
        
        # Khởi tạo dict kết quả
        detections = {
            'face_detected': False,
            'ear': 0.0,
            'is_drowsy': False,
            'is_distracted': False,
            'gaze_angle': (0, 0),
            'blink_detected': False
        }
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            detections['face_detected'] = True
            
            # Lấy kích thước frame
            img_h, img_w = frame.shape[:2]
            
            # 1. PHÁT HIỆN BUỒN NGỦ (qua EAR)
            ear = self._calculate_ear_from_landmarks(face_landmarks, img_w, img_h)
            detections['ear'] = ear
            
            # Thêm vào lịch sử và làm mượt
            self.ear_history.append(ear)
            if len(self.ear_history) > self.history_size:
                self.ear_history.pop(0)
            
            smoothed_ear = np.mean(self.ear_history)
            
            # Kiểm tra nhắm mắt
            if smoothed_ear < config.EAR_THRESHOLD:
                self.eye_closed_counter += 1
            else:
                # Phát hiện chớp mắt
                if self.eye_closed_counter > 2 and self.eye_closed_counter < 20:
                    current_time = time.time()
                    if current_time - self.last_blink_time > 0.5:  # Tránh đếm trùng
                        self.total_blinks += 1
                        detections['blink_detected'] = True
                        self.last_blink_time = current_time
                
                self.eye_closed_counter = 0
            
            # Xác định buồn ngủ
            if self.eye_closed_counter >= config.EAR_CONSEC_FRAMES:
                self.is_drowsy = True
                detections['is_drowsy'] = True
                self.last_drowsy_time = time.time()
            else:
                self.is_drowsy = False
            
            # Phát hiện mất tập trung (qua hướng nhìn)
            h_angle, v_angle = calculate_gaze_direction(face_landmarks, img_w, img_h)
            detections['gaze_angle'] = (h_angle, v_angle)
            
            if is_looking_away(h_angle, v_angle, config.GAZE_THRESHOLD):
                self.gaze_away_counter += 1
            else:
                self.gaze_away_counter = 0
            
            if self.gaze_away_counter >= config.GAZE_CONSEC_FRAMES:
                self.is_distracted = True
                detections['is_distracted'] = True
                self.last_distracted_time = time.time()
            else:
                self.is_distracted = False
            
            # Vẽ landmarks lên frame
            self._draw_detections(frame, face_landmarks, img_w, img_h, detections)
        
        return frame, detections
    
    def _calculate_ear_from_landmarks(self, face_landmarks, img_w, img_h):

        # Tính EAR từ face landmarks
        
        # Trích xuất landmarks mắt trái
        left_eye = extract_eye_landmarks(
            face_landmarks, 
            config.LEFT_EYE_INDICES, 
            img_w, img_h
        )
        
        # Trích xuất landmarks mắt phải
        right_eye = extract_eye_landmarks(
            face_landmarks, 
            config.RIGHT_EYE_INDICES, 
            img_w, img_h
        )
        
        # Tính EAR cho mỗi mắt
        left_ear = calculate_ear(left_eye)
        right_ear = calculate_ear(right_eye)
        
        # Trung bình
        ear = (left_ear + right_ear) / 2.0
        
        return ear
    
    def _draw_detections(self, frame, face_landmarks, img_w, img_h, detections):
        
        # Vẽ các detection lên frame
        
        # Vẽ landmarks mắt
        left_eye = extract_eye_landmarks(
            face_landmarks, 
            config.LEFT_EYE_INDICES, 
            img_w, img_h
        )
        right_eye = extract_eye_landmarks(
            face_landmarks, 
            config.RIGHT_EYE_INDICES, 
            img_w, img_h
        )
        
        # Màu dựa trên trạng thái
        eye_color = config.COLOR_RED if detections['is_drowsy'] else config.COLOR_GREEN
        draw_eye_landmarks(frame, left_eye, eye_color)
        draw_eye_landmarks(frame, right_eye, eye_color)
        
        # Vẽ thông tin lên góc trên trái
        y_offset = 30
        
        # EAR value
        cv2.putText(frame, f"EAR: {detections['ear']:.3f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, config.COLOR_WHITE, 2)
        y_offset += 25
        
        # Trạng thái mắt
        eye_status = "DROWSY!" if detections['is_drowsy'] else "Awake"
        eye_color_text = config.COLOR_RED if detections['is_drowsy'] else config.COLOR_GREEN
        cv2.putText(frame, f"Eyes: {eye_status}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, eye_color_text, 2)
        y_offset += 25
        
        # Hướng nhìn
        h_angle, v_angle = detections['gaze_angle']
        cv2.putText(frame, f"Gaze: H={h_angle:.1f} V={v_angle:.1f}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, config.COLOR_WHITE, 2)
        y_offset += 25
        
        # Trạng thái tập trung
        focus_status = "DISTRACTED!" if detections['is_distracted'] else "Focused"
        focus_color = config.COLOR_RED if detections['is_distracted'] else config.COLOR_GREEN
        cv2.putText(frame, f"Focus: {focus_status}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, focus_color, 2)
        y_offset += 25
        
        # Số lần chớp mắt
        cv2.putText(frame, f"Blinks: {self.total_blinks}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, config.COLOR_WHITE, 2)
    
    def reset_counters(self):
        
        self.eye_closed_counter = 0
        self.gaze_away_counter = 0
        self.total_blinks = 0
    
    def get_stats(self):
        
        return {
            'total_blinks': self.total_blinks,
            'is_drowsy': self.is_drowsy,
            'is_distracted': self.is_distracted,
            'eye_closed_frames': self.eye_closed_counter,
            'gaze_away_frames': self.gaze_away_counter
        }
    
    def cleanup(self):
        
        self.face_mesh.close()