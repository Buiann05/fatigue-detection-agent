import csv
import os
from datetime import datetime
import pandas as pd
from utils import ensure_dir, get_timestamp


class ActivityLogger:
    
    # Ghi log các sự kiện trong quá trình giám sát
    
    def __init__(self, log_file="logs/activity_log.csv"):
        
        self.log_file = log_file
        
        # Đảm bảo thư mục tồn tại
        ensure_dir(os.path.dirname(log_file))
        
        # Tạo file mới nếu chưa có
        if not os.path.exists(log_file):
            self._create_log_file()
        
        # Thống kê session
        self.session_start = datetime.now()
        self.session_stats = {
            'focused_time': 0,
            'drowsy_count': 0,
            'distracted_count': 0,
            'total_alerts': 0
        }
    
    def _create_log_file(self):
        
        with open(self.log_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp',
                'session_id',
                'event_type',
                'duration',
                'details'
            ])
    
    def log_event(self, event_type, duration=0, details=""):
        
        # Ghi log một sự kiện
        
        session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        timestamp = get_timestamp()
        
        # Cập nhật thống kê
        if event_type == 'drowsy':
            self.session_stats['drowsy_count'] += 1
            self.session_stats['total_alerts'] += 1
        elif event_type == 'distracted':
            self.session_stats['distracted_count'] += 1
            self.session_stats['total_alerts'] += 1
        elif event_type == 'focused':
            self.session_stats['focused_time'] += duration
        
        # Ghi vào file
        try:
            with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([
                    timestamp,
                    session_id,
                    event_type,
                    f"{duration:.2f}",
                    details
                ])
        except Exception as e:
            print(f"Lỗi khi ghi log: {e}")
    
    def get_session_stats(self):
        
        # Lấy thống kê session hiện tại
        
        total_time = (datetime.now() - self.session_start).total_seconds()
        
        return {
            'session_duration': total_time,
            'focused_time': self.session_stats['focused_time'],
            'focused_percentage': (self.session_stats['focused_time'] / total_time * 100) if total_time > 0 else 0,
            'drowsy_count': self.session_stats['drowsy_count'],
            'distracted_count': self.session_stats['distracted_count'],
            'total_alerts': self.session_stats['total_alerts']
        }
    
    def load_logs(self, session_id=None):
        
        # Load logs từ file
        
        try:
            df = pd.read_csv(self.log_file)
            
            if session_id:
                df = df[df['session_id'] == session_id]
            
            return df
        except Exception as e:
            print(f"⚠️ Lỗi khi load logs: {e}")
            return pd.DataFrame()
    
    def get_all_sessions(self):
        
        # Lấy danh sách tất cả các session
        
        try:
            df = pd.read_csv(self.log_file)
            return df['session_id'].unique().tolist()
        except:
            return []


class SessionReporter:
    
    # Tạo báo cáo chi tiết cho session
    
    def __init__(self, logger):
        
        self.logger = logger
    
    def generate_text_report(self):
        
        # Tạo báo cáo dạng text
        
        stats = self.logger.get_session_stats()
        
        report = f"""
         BÁO CÁO PHIÊN GIÁM SÁT                      

Ngày: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}
Thời gian phiên: {stats['session_duration']/60:.1f} phút

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

THỐNG KÊ TỔNG QUAN:

Thời gian tập trung: {stats['focused_time']/60:.1f} phút ({stats['focused_percentage']:.1f}%)
Số lần buồn ngủ: {stats['drowsy_count']} lần
Số lần mất tập trung: {stats['distracted_count']} lần
Tổng cảnh báo: {stats['total_alerts']} lần

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ĐÁNH GIÁ:
"""
        # Đánh giá
        if stats['focused_percentage'] >= 90:
            report += "Xuất sắc! Bạn tập trung rất tốt.\n"
        elif stats['focused_percentage'] >= 75:
            report += "Tốt! Bạn duy trì sự tập trung khá ổn.\n"
        elif stats['focused_percentage'] >= 60:
            report += "Trung bình. Cần cải thiện sự tập trung.\n"
        else:
            report += "Cần cải thiện! Bạn bị mất tập trung nhiều.\n"
        
        report += "\n╚════════════════════════════════════════════════════════╝\n"
        
        return report
    
    def save_report(self, output_path="reports/session_report.txt"):
        
        # Lưu báo cáo ra file
        
        ensure_dir(os.path.dirname(output_path))
        
        report = self.generate_text_report()
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Đã lưu báo cáo: {output_path}")
        except Exception as e:
            print(f"Lỗi khi lưu báo cáo: {e}")
