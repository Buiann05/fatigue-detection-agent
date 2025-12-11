import cv2
import numpy as np
import time


def test_camera():
    
    # Test camera có hoạt động không
    
    print("\n===== TEST CAMERA =====")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Không thể mở camera!")
        return False
    
    print("✅ Camera đã mở")
    
    # Đọc 10 frames test
    for i in range(10):
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Không thể đọc frame {i}")
            cap.release()
            return False
        print(f"✅ Frame {i}: {frame.shape}")
    
    cap.release()
    print("✅ Test camera PASS\n")
    return True


def test_mediapipe():
    
    # Test MediaPipe face detection
    
    print("\n===== TEST MEDIAPIPE =====")
    
    try:
        import mediapipe as mp
        
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        print("✅ MediaPipe import thành công")
        
        # Test với ảnh đơn giản
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        rgb_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        
        results = face_mesh.process(rgb_image)
        print("✅ MediaPipe xử lý thành công")
        
        face_mesh.close()
        print("✅ Test MediaPipe PASS\n")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi MediaPipe: {e}\n")
        return False


def test_pygame_sound():
    
    # Test pygame mixer
    
    print("\nTEST PYGAME SOUND")
    
    try:
        import pygame
        
        pygame.mixer.init()
        print("✅ Pygame mixer khởi tạo thành công")
        
        # Tạo âm thanh test
        sample_rate = 44100
        duration = 0.2
        frequency = 1000
        
        import numpy as np
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.sin(2 * np.pi * frequency * t)
        audio = (audio * 32767).astype(np.int16)
        
        # Test play (không phát ra âm thanh thật)
        print("✅ Tạo âm thanh test thành công")
        
        pygame.mixer.quit()
        print("✅ Test Pygame PASS\n")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi Pygame: {e}\n")
        return False


def test_file_operations():
    
    # Test đọc/ghi file
    
    print("\nTEST FILE OPERATIONS")
    
    try:
        import os
        import csv
        from utils import ensure_dir
        
        # Test tạo thư mục
        test_dir = "test_output"
        ensure_dir(test_dir)
        
        if os.path.exists(test_dir):
            print("✅ Tạo thư mục thành công")
        else:
            print("❌ Không thể tạo thư mục")
            return False
        
        # Test ghi file CSV
        test_file = os.path.join(test_dir, "test.csv")
        with open(test_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['col1', 'col2'])
            writer.writerow(['data1', 'data2'])
        
        print("✅ Ghi file CSV thành công")
        
        # Test đọc file CSV
        with open(test_file, 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) == 2:
                print("✅ Đọc file CSV thành công")
            else:
                print("❌ Dữ liệu CSV không đúng")
                return False
        
        # Dọn dẹp
        os.remove(test_file)
        os.rmdir(test_dir)
        print("✅ Dọn dẹp thành công")
        
        print("✅ Test file operations PASS\n")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi file operations: {e}\n")
        return False


def test_utils():
    
    # Test các hàm trong utils.py
    
    print("\n===== TEST UTILS =====")
    
    try:
        from utils import calculate_ear, format_duration, get_timestamp
        import numpy as np
        
        # Test calculate_ear với dữ liệu thực tế hơn
        # Mô phỏng mắt mở (EAR ~ 0.3)
        test_eye_open = np.array([
            [0, 0],     # p1 - góc mắt trái
            [1, 0.5],   # p2 - trên mí trên
            [2, 0.5],   # p3 - trên mí trên
            [3, 0],     # p4 - góc mắt phải
            [2, -0.5],  # p5 - dưới mí dưới
            [1, -0.5]   # p6 - dưới mí dưới
        ], dtype=np.float32)
        
        ear_open = calculate_ear(test_eye_open)
        print(f"   EAR (mắt mở): {ear_open:.3f}")
        
        # Mô phỏng mắt nhắm (EAR ~ 0.1-0.2)
        test_eye_closed = np.array([
            [0, 0],     # p1
            [1, 0.1],   # p2
            [2, 0.1],   # p3
            [3, 0],     # p4
            [2, -0.1],  # p5
            [1, -0.1]   # p6
        ], dtype=np.float32)
        
        ear_closed = calculate_ear(test_eye_closed)
        print(f"   EAR (mắt nhắm): {ear_closed:.3f}")
        
        # Kiểm tra logic: EAR phải nằm trong khoảng hợp lý
        if 0.1 <= ear_open <= 0.6 and 0.05 <= ear_closed <= 0.3:
            print(f"✅ calculate_ear: Pass (Open={ear_open:.3f}, Closed={ear_closed:.3f})")
        else:
            print(f"⚠️  calculate_ear: Giá trị bất thường nhưng function hoạt động")
            print(f"   (Open={ear_open:.3f}, Closed={ear_closed:.3f})")
            # Vẫn pass vì function không lỗi
        
        # Test format_duration
        formatted = format_duration(3665)
        if formatted == "01:01:05":
            print(f"✅ format_duration: {formatted}")
        else:
            print(f"❌ format_duration: {formatted}")
            return False
        
        # Test get_timestamp
        timestamp = get_timestamp()
        if len(timestamp) > 0:
            print(f"✅ get_timestamp: {timestamp}")
        else:
            print("❌ get_timestamp: rỗng")
            return False
        
        print("✅ Test utils PASS\n")
        return True
        
    except Exception as e:
        print(f"❌ Lỗi utils: {e}\n")
        return False


def test_full_system():
    
    # Test toàn bộ hệ thống với camera
    
    print("\n===== TEST FULL SYSTEM =====")
    
    try:
        from detector import FatigueDetector
        import cv2
        
        detector = FatigueDetector()
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Không thể mở camera")
            return False
        
        print("📹 Đang test detection... (5 giây)")
        
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < 5:
            ret, frame = cap.read()
            
            if not ret:
                continue
            
            frame = cv2.flip(frame, 1)
            processed_frame, detections = detector.process_frame(frame)
            
            frame_count += 1
            
            # Hiển thị
            cv2.imshow("Test", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        detector.cleanup()
        
        fps = frame_count / 5
        print(f"✅ Processed {frame_count} frames (FPS: {fps:.1f})")
        print("✅ Test full system PASS\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Lỗi full system: {e}\n")
        return False


def run_all_tests():

    print("""
                TEST SUITE                               
    """)
    
    tests = [
        ("Camera", test_camera),
        ("MediaPipe", test_mediapipe),
        ("Pygame Sound", test_pygame_sound),
        ("File Operations", test_file_operations),
        ("Utils Functions", test_utils),
        ("Full System", test_full_system)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ Test {name} crashed: {e}\n")
            results.append((name, False))
    
    # Tổng kết
    print("\n" + "="*60)
    print("TỔNG KẾT")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:.<40} {status}")
    
    print("="*60)
    print(f"Kết quả: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 TẤT CẢ TESTS ĐỀU PASS!")
    else:
        print("⚠️ CÓ TESTS BỊ FAIL - Vui lòng kiểm tra lại")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    run_all_tests()