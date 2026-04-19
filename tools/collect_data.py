import cv2
import numpy as np
import os
import pandas as pd
from src.detection.hand_detector import HolisticDetector

class DataCollector:
    def __init__(self, person_id, save_dir='data/raw/collected', target_samples=30):
        self.detector = HolisticDetector()
        self.person_id = person_id
        self.save_dir = save_dir
        self.target_samples = target_samples
        
        vocab_path = 'data/vocab.csv'
        self.vocab_df = pd.read_csv(vocab_path)
        self.words = self.vocab_df['word'].tolist()
        self.current_idx = 0
        
        self.is_recording = False
        self.current_sequence = []
        self.recorded_counts = {}
        
        self._init_save_dirs()
        self._load_recorded_counts()
    
    def _init_save_dirs(self):
        for word in self.words:
            word_dir = os.path.join(self.save_dir, word)
            os.makedirs(word_dir, exist_ok=True)
    
    def _load_recorded_counts(self):
        for word in self.words:
            word_dir = os.path.join(self.save_dir, word)
            if os.path.exists(word_dir):
                files = [f for f in os.listdir(word_dir) if f.startswith(self.person_id) and f.endswith('.npy')]
                self.recorded_counts[word] = len(files)
            else:
                self.recorded_counts[word] = 0
    
    def _get_next_index(self, word):
        word_dir = os.path.join(self.save_dir, word)
        os.makedirs(word_dir, exist_ok=True)
        existing_files = [f for f in os.listdir(word_dir) if f.startswith(self.person_id) and f.endswith('.npy')]
        return len(existing_files) + 1
    
    def _save_sequence(self, word, sequence):
        if len(sequence) < 15:
            return False, f"序列太短（{len(sequence)}帧），至少需要15帧"
        
        if len(sequence) > 150:
            start = (len(sequence) - 150) // 2
            sequence = sequence[start:start + 150]
        
        sequence_array = np.array(sequence)
        
        index = self._get_next_index(word)
        file_name = f'{self.person_id}_{index:03d}.npy'
        save_path = os.path.join(self.save_dir, word, file_name)
        np.save(save_path, sequence_array)
        
        self.recorded_counts[word] = self.recorded_counts.get(word, 0) + 1
        
        return True, f"已保存: {save_path} (形状: {sequence_array.shape})"
    
    def _draw_ui(self, frame, status_text=""):
        h, w = frame.shape[:2]
        
        if self.is_recording:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
            cv2.putText(frame, f"RECORDING: {len(self.current_sequence)} frames", 
                       (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 255, 0), 2)
        
        word = self.words[self.current_idx]
        category = self.vocab_df.iloc[self.current_idx]['category']
        
        cv2.putText(frame, f"词: {word} ({category})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(frame, f"进度: {self.current_idx + 1}/{len(self.words)}", (10, 65), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        
        recorded = self.recorded_counts.get(word, 0)
        cv2.putText(frame, f"已录: {recorded}/{self.target_samples}", (w - 200, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        
        if status_text:
            cv2.putText(frame, status_text, (w // 2 - 200, h // 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(frame, "[SPACE]录制 [N]下一个 [P]上一个 [Q]统计退出 [ESC]直接退出",
                   (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def _draw_warning(self, frame, message):
        h, w = frame.shape[:2]
        cv2.putText(frame, message, (w // 2 - 150, h // 2 + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        return frame
    
    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("错误：无法打开摄像头")
            return
        
        print(f"开始数据采集，当前录制人: {self.person_id}")
        print(f"词汇表共 {len(self.words)} 个词，每个词目标录制 {self.target_samples} 次")
        print("操作说明：")
        print("  [空格] 开始/停止录制")
        print("  [N] 下一个词")
        print("  [P] 上一个词")
        print("  [Q] 显示统计后退出")
        print("  [ESC] 直接退出")
        
        status_message = ""
        status_timer = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            results = self.detector.detect(frame)
            landmarks = self.detector.get_landmarks(results, frame.shape)
            frame = self.detector.draw_landmarks(frame, results)
            
            has_hand = not np.all(landmarks[:126] == 0)
            
            if self.is_recording:
                if has_hand:
                    self.current_sequence.append(landmarks)
                else:
                    if status_timer <= 0:
                        status_message = "警告：检测不到手！"
                        status_timer = 30
            
            if status_timer > 0:
                frame = self._draw_warning(frame, status_message)
                status_timer -= 1
            
            frame = self._draw_ui(frame, status_message if status_timer > 0 else "")
            
            cv2.imshow('Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:
                break
            elif key == ord(' '):
                if self.is_recording:
                    self.is_recording = False
                    if len(self.current_sequence) > 0:
                        success, msg = self._save_sequence(self.words[self.current_idx], self.current_sequence)
                        print(msg)
                        if success:
                            status_message = f"保存成功！({len(self.current_sequence)}帧)"
                            status_timer = 60
                        else:
                            status_message = msg
                            status_timer = 90
                        self.current_sequence = []
                    else:
                        status_message = "未录制到有效数据"
                        status_timer = 60
                else:
                    self.is_recording = True
                    self.current_sequence = []
                    status_message = "开始录制..."
                    status_timer = 30
            elif key == ord('n') or key == ord('N'):
                if self.current_idx < len(self.words) - 1:
                    self.current_idx += 1
                    status_message = f"切换到: {self.words[self.current_idx]}"
                    status_timer = 60
            elif key == ord('p') or key == ord('P'):
                if self.current_idx > 0:
                    self.current_idx -= 1
                    status_message = f"切换到: {self.words[self.current_idx]}"
                    status_timer = 60
            elif key == ord('q') or key == ord('Q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        self._print_statistics()
        
        self.detector.close()
    
    def _print_statistics(self):
        print("\n" + "=" * 50)
        print("录制统计")
        print("=" * 50)
        
        total_recorded = 0
        total_target = len(self.words) * self.target_samples
        
        for word in self.words:
            recorded = self.recorded_counts.get(word, 0)
            total_recorded += recorded
            category = self.vocab_df[self.vocab_df['word'] == word]['category'].values[0]
            status = "✓" if recorded >= self.target_samples else "✗"
            print(f"  {status} {word} ({category}): {recorded}/{self.target_samples}")
        
        print("-" * 50)
        print(f"总计: {total_recorded}/{total_target} ({total_recorded/total_target*100:.1f}%)")
        print("=" * 50)

def main():
    print("=" * 50)
    print("聆心手语数据采集工具")
    print("=" * 50)
    
    person_id = input("请输入录制人ID: ").strip()
    if not person_id:
        print("错误：录制人ID不能为空")
        return
    
    collector = DataCollector(person_id=person_id)
    collector.run()

if __name__ == '__main__':
    main()