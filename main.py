import cv2
import time
import sys
import numpy as np  

import config
from detector import FatigueDetector
from alert_system import AlertSystem, VisualAlertOverlay
from logger import ActivityLogger, SessionReporter
from utils import format_duration

from advanced_features import (
    YawnDetector, 
    HeadPoseEstimator, 
    BreakReminder, 
    FocusScoreCalculator
)

class FatigueDetectionAgent:

    # AI Agent giám sát mệt mỏi và mất tập trung
    
    def __init__(self):
        print("Khởi động AI Agent Giám sát Mệt mỏi...")
        
        # Khởi tạo các components cơ bản
        self.detector = FatigueDetector()
        self.alert_system = AlertSystem(
            sound_path=config.ALERT_SOUND_PATH,
            cooldown=config.ALERT_COOLDOWN
        )
        self.logger = ActivityLogger(log_file=config.LOG_FILE)
        self.reporter = SessionReporter(self.logger)
        
        self.yawn_detector = YawnDetector(threshold=0.6)
        self.head_pose_estimator = HeadPoseEstimator()
        self.break_reminder = BreakReminder(interval_minutes=25) 
        self.focus_score_calc = FocusScoreCalculator(window_size=60)
        
        # Camera
        self.camera = None
        self.is_running = False
        
        # Thời gian
        self.start_time = None
        self.last_log_time = 0
        self.log_interval = 60  # Ghi log mỗi 60 giây
        
        print("Khởi tạo thành công!")
    
    def initialize_camera(self):
        
        # Khởi tạo camera
        
        print(f"Đang khởi động camera {config.CAMERA_INDEX}...")
        
        self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
        
        if not self.camera.isOpened():
            print("Không thể mở camera!")
            return False
        
        # Cài đặt resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.camera.set(cv2.CAP_PROP_FPS, config.FPS)
        
        # Kiểm tra
        ret, frame = self.camera.read()
        if not ret:
            print("Không thể đọc frame từ camera!")
            return False
        
        print(f"Camera đã sẵn sàng ({frame.shape[1]}x{frame.shape[0]})")
        
        # Khởi tạo visual overlay
        self.visual_overlay = VisualAlertOverlay(frame.shape[1], frame.shape[0])
        
        return True
    
    def _extract_mouth_points(self, face_landmarks, width, height):
        
        indices = [78, 81, 13, 311, 308, 402, 14, 178] 
        
        points = []
        for idx in indices:
            lm = face_landmarks.landmark[idx]
            points.append([lm.x * width, lm.y * height])
        
        return np.array(points, dtype=np.float64)

    def run(self):
        if not self.initialize_camera():
            return
        
        self.is_running = True
        self.start_time = time.time()
        
        print("\n" + "="*60)
        print("AI AGENT ĐANG HOẠT ĐỘNG")
        print("="*60)
        print("Nhấn 'q' để thoát")
        print("Nhấn 's' để xem thống kê")
        print("Nhấn 'r' để reset bộ đếm")
        print("="*60 + "\n")
        
        fps_counter = 0
        fps_start_time = time.time()
        current_fps = 0
        
        try:
            while self.is_running:
                # Đọc frame
                ret, frame = self.camera.read()
                
                if not ret:
                    print("Không thể đọc frame!")
                    break
                
                # Lật frame (mirror)
                frame = cv2.flip(frame, 1)
                
                # Xử lý detection cơ bản (Mắt, EAR, Gaze)
                processed_frame, detections = self.detector.process_frame(frame)
                                
                # Lấy face landmarks từ detector
                face_landmarks = getattr(self.detector, 'current_landmarks', None)
                
                if detections['face_detected'] and face_landmarks:
                    h, w = frame.shape[:2]
                    
                    # Phát hiện ngáp
                    try:
                        # Trích xuất điểm miệng
                        mouth_landmarks = self._extract_mouth_points(face_landmarks, w, h)
                        is_yawning = self.yawn_detector.detect_yawn(mouth_landmarks)
                        
                        if is_yawning:
                            print("🥱 Phát hiện ngáp!")
                            self.visual_overlay.add_alert("PHÁT HIỆN NGÁP!", config.COLOR_YELLOW, duration=2.0)
                            # Ghi log sự kiện ngáp
                            self.logger.log_event('yawn', duration=1.0, details="Mouth Aspect Ratio High")
                    except Exception as e:
                        pass # Bỏ qua lỗi tính nếu mặt bị khuất

                    # 2. Ước lượng quay và nghiêng đầu
                    pitch, yaw, roll = self.head_pose_estimator.estimate_pose(face_landmarks, frame.shape)
                    
                    # Hiển thị thông số Pose
                    pose_text = f"Head: P={pitch:.0f} Y={yaw:.0f} R={roll:.0f}"
                    cv2.putText(processed_frame, pose_text, (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Cảnh báo nếu cúi đầu quá thấp (ngủ gật) hoặc quay đi quá nhiều
                    if abs(pitch) > 25 or abs(yaw) > 30:
                        detections['is_distracted'] = True # Ghi đè trạng thái mất tập trung
                
                # Tính điểm tập trung
                # Nếu không buồn ngủ và không mất tập trung -> Đang tập trung
                is_focused = not (detections['is_drowsy'] or detections['is_distracted'])
                self.focus_score_calc.add_event(is_focused)
                
                score = self.focus_score_calc.get_focus_score()
                grade = self.focus_score_calc.get_grade()
                
                cv2.putText(processed_frame, f"Focus Score: {score:.0f}% ({grade})",
                           (10, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Nhắc nhở nghỉ giải lao 
                if self.break_reminder.check_break_time():
                    self.visual_overlay.add_alert(
                        "⏰ Đã 25 phút! Hãy nghỉ giải lao 5 phút",
                        config.COLOR_BLUE,
                        duration=10.0
                    )
                    self.alert_system.play_sound()
                    self.break_reminder.take_break() 

                # Xử lý cảnh báo (Drowsy/Distracted từ detector gốc + logic mới)
                self._handle_alerts(detections)
                
                # Ghi log định kỳ
                current_time = time.time()
                if current_time - self.last_log_time >= self.log_interval:
                    self._log_periodic_status(detections)
                    self.last_log_time = current_time
                
                # Vẽ visual alerts
                self.visual_overlay.render(processed_frame)
                
                # Hiển thị thông tin runtime
                self._draw_runtime_info(processed_frame, current_fps)
                
                # Hiển thị
                cv2.imshow("AI Agent - Fatigue & Focus Monitor", processed_frame)
                
                # Tính FPS
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    current_fps = fps_counter
                    fps_counter = 0
                    fps_start_time = time.time()
                
                # Xử lý phím
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nĐang dừng AI Agent...")
                    break
                elif key == ord('s'):
                    self._show_stats()
                elif key == ord('r'):
                    self.detector.reset_counters()
                    print("Đã reset bộ đếm")
        
        except KeyboardInterrupt:
            print("\nNgắt bởi người dùng")
        
        finally:
            self.cleanup()
    
    def _handle_alerts(self, detections):

        # Xử lý cảnh báo dựa trên detections

        if detections['is_drowsy']:
            if self.alert_system.trigger_alert('drowsy'):
                self.visual_overlay.add_alert(
                    "CẢNH BÁO: BẠN ĐANG BUỒN NGỦ!",
                    config.COLOR_RED,
                    duration=3.0
                )
                self.logger.log_event(
                    'drowsy',
                    duration=self.detector.eye_closed_counter / config.FPS,
                    details=f"EAR={detections['ear']:.3f}"
                )
                print(f"[CẢNH BÁO] Phát hiện buồn ngủ - EAR: {detections['ear']:.3f}")
        
        if detections['is_distracted']:
            if self.alert_system.trigger_alert('distracted'):
                self.visual_overlay.add_alert(
                    "CẢNH BÁO: BẠN ĐANG MẤT TẬP TRUNG!",
                    config.COLOR_YELLOW,
                    duration=3.0
                )
                h_angle, v_angle = detections['gaze_angle']
                self.logger.log_event(
                    'distracted',
                    duration=self.detector.gaze_away_counter / config.FPS,
                    details=f"Gaze=({h_angle:.1f}, {v_angle:.1f})"
                )
                print(f"[CẢNH BÁO] Phát hiện mất tập trung - Góc: {h_angle:.1f}°")
    
    def _log_periodic_status(self, detections):

        # Ghi log định kỳ trạng thái

        if not detections['is_drowsy'] and not detections['is_distracted']:
            self.logger.log_event(
                'focused',
                duration=self.log_interval,
                details=f"EAR={detections['ear']:.3f}, Blinks={self.detector.total_blinks}, Score={self.focus_score_calc.get_focus_score():.1f}"
            )
    
    def _draw_runtime_info(self, frame, fps):
        # Vẽ thông tin runtime lên frame

        # Thời gian chạy
        elapsed = time.time() - self.start_time
        runtime_text = f"Runtime: {format_duration(elapsed)}"
        
        # FPS
        fps_text = f"FPS: {fps}"
        
        # Vẽ nền
        cv2.rectangle(frame, (0, frame.shape[0] - 60), (300, frame.shape[0]), (0, 0, 0), -1)
        
        # Vẽ text
        cv2.putText(frame, runtime_text, (10, frame.shape[0] - 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1)
        cv2.putText(frame, fps_text, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_WHITE, 1)
    
    def _show_stats(self):

        # Hiển thị thống kê

        print("\n" + "="*60)
        print("THỐNG KÊ HIỆN TẠI")
        print("="*60)
        
        stats = self.logger.get_session_stats()
        detector_stats = self.detector.get_stats()
        
        print(f"Thời gian: {format_duration(stats['session_duration'])}")
        print(f"Tập trung: {stats['focused_time']/60:.1f} phút ({stats['focused_percentage']:.1f}%)")
        print(f"Điểm tập trung hiện tại: {self.focus_score_calc.get_focus_score():.1f} ({self.focus_score_calc.get_grade()})")
        print(f"Buồn ngủ: {stats['drowsy_count']} lần")
        print(f"Mất tập trung: {stats['distracted_count']} lần")
        print(f"Ngáp: {self.yawn_detector.total_yawns} lần")
        print(f"Chớp mắt: {detector_stats['total_blinks']} lần")
        print(f"Tổng cảnh báo: {stats['total_alerts']} lần")
        print("="*60 + "\n")
    
    def cleanup(self):

        # Dọn dẹp tài nguyên
        
        print("\nĐang dọn dẹp...")
        
        # Tạo báo cáo cuối session
        print("\nTạo báo cáo phiên...")
        report = self.reporter.generate_text_report()
        print(report)
        
        # Lưu báo cáo
        self.reporter.save_report()
        
        # Dọn dẹp
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        self.detector.cleanup()
        
        print("Đã dọn dẹp xong. Tạm biệt!")


def main():

    print("""
    AI AGENT GIÁM SÁT MỆT MỎI & MẤT TẬP TRUNG
    """)
    
    # Khởi tạo và chạy agent
    agent = FatigueDetectionAgent()
    agent.run()


if __name__ == "__main__":
    main()