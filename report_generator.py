import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from datetime import datetime
import os

from utils import ensure_dir


class ReportGenerator:
    
    # Tạo các biểu đồ và báo cáo trực quan
    
    def __init__(self, log_file="logs/activity_log.csv"):
       
        self.log_file = log_file
        self.output_dir = "reports/"
        ensure_dir(self.output_dir)
    
    def load_session_data(self, session_id):
        
        # Load dữ liệu của một session
        
        try:
            df = pd.read_csv(self.log_file)
            df = df[df['session_id'] == session_id]
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
        except Exception as e:
            print(f"Lỗi khi load dữ liệu: {e}")
            return pd.DataFrame()
    
    def generate_timeline_chart(self, session_id, output_path=None):
        
        # Tạo biểu đồ timeline các sự kiện
        
        df = self.load_session_data(session_id)
        
        if df.empty:
            print("Không có dữ liệu để vẽ biểu đồ")
            return
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Màu cho từng loại sự kiện
        colors = {
            'focused': 'green',
            'drowsy': 'red',
            'distracted': 'orange',
            'alert': 'purple'
        }
        
        # Vẽ timeline
        for event_type in df['event_type'].unique():
            event_data = df[df['event_type'] == event_type]
            
            ax.scatter(
                event_data['timestamp'], 
                [event_type] * len(event_data),
                c=colors.get(event_type, 'blue'),
                s=100,
                alpha=0.6,
                label=event_type.capitalize()
            )
        
        # Định dạng
        ax.set_xlabel('Thời gian', fontsize=12, fontweight='bold')
        ax.set_ylabel('Loại sự kiện', fontsize=12, fontweight='bold')
        ax.set_title(f'Timeline Sự kiện - Session {session_id}', 
                    fontsize=14, fontweight='bold')
        
        # Format trục x
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        plt.xticks(rotation=45)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Lưu
        if output_path is None:
            output_path = os.path.join(self.output_dir, f'timeline_{session_id}.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ timeline: {output_path}")
        plt.close()
    
    def generate_pie_chart(self, session_id, output_path=None):
        
        # Tạo biểu đồ tròn phân bố thời gian
        
        df = self.load_session_data(session_id)
        
        if df.empty:
            print("Không có dữ liệu")
            return
        
        # Tính tổng thời gian cho mỗi loại
        duration_by_type = df.groupby('event_type')['duration'].sum()
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        explode = [0.05 if x == 'focused' else 0 for x in duration_by_type.index]
        
        wedges, texts, autotexts = ax.pie(
            duration_by_type,
            labels=duration_by_type.index,
            autopct='%1.1f%%',
            startangle=90,
            colors=colors[:len(duration_by_type)],
            explode=explode,
            shadow=True
        )
        
        # Định dạng text
        for text in texts:
            text.set_fontsize(12)
            text.set_fontweight('bold')
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        
        ax.set_title(f'Phân bố thời gian theo trạng thái\nSession {session_id}',
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Lưu
        if output_path is None:
            output_path = os.path.join(self.output_dir, f'pie_chart_{session_id}.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Đã lưu biểu đồ tròn: {output_path}")
        plt.close()
    
    def generate_bar_chart(self, session_id, output_path=None):
        
        # Tạo biểu đồ cột số lượng sự kiện
        
        df = self.load_session_data(session_id)
        
        if df.empty:
            print("Không có dữ liệu")
            return
        
        # Đếm số lượng từng loại sự kiện
        event_counts = df['event_type'].value_counts()
        
        # Tạo figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']
        bars = ax.bar(
            event_counts.index,
            event_counts.values,
            color=colors[:len(event_counts)],
            alpha=0.7,
            edgecolor='black'
        )
        
        # Thêm giá trị lên đầu cột
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                f'{int(height)}',
                ha='center',
                va='bottom',
                fontsize=12,
                fontweight='bold'
            )
        
        ax.set_xlabel('Loại sự kiện', fontsize=12, fontweight='bold')
        ax.set_ylabel('Số lượng', fontsize=12, fontweight='bold')
        ax.set_title(f'Số lượng sự kiện theo loại\nSession {session_id}',
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Lưu
        if output_path is None:
            output_path = os.path.join(self.output_dir, f'bar_chart_{session_id}.png')
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Đã lưu biểu đồ cột: {output_path}")
        plt.close()
    
    def generate_complete_report(self, session_id):
        
        # Tạo báo cáo đầy đủ với tất cả biểu đồ
        
        print(f"\nĐang tạo báo cáo đầy đủ cho session {session_id}...")
        
        self.generate_timeline_chart(session_id)
        self.generate_pie_chart(session_id)
        self.generate_bar_chart(session_id)
        
        print(f"\nHoàn thành! Báo cáo được lưu trong thư mục: {self.output_dir}")


def main():
    
    # Script chạy độc lập để tạo báo cáo
    
    import sys
    
    if len(sys.argv) < 2:
        print("Sử dụng: python report_generator.py <session_id>")
        print("\nĐể xem danh sách sessions:")
        
        try:
            df = pd.read_csv("logs/activity_log.csv")
            sessions = df['session_id'].unique()
            print("\nCác session có sẵn:")
            for s in sessions:
                print(f"  - {s}")
        except:
            print("  Không tìm thấy file log")
        
        sys.exit(1)
    
    session_id = sys.argv[1]
    
    generator = ReportGenerator()
    generator.generate_complete_report(session_id)


if __name__ == "__main__":
    main()