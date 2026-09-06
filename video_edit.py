import cv2
import numpy as np

video_path = "A001_P228_G001_H120.mp4"
target_frames = 13
target_size = (224, 224)

cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

indices = np.linspace(0, total_frames - 1, target_frames).round().astype(int)

frames = []
for idx in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok:
        continue

    frame = cv2.resize(frame, target_size)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame)

cap.release()

frames = np.stack(frames)  # [13, 224, 224, 3]

print("fps:", fps)
print("total frames:", total_frames)
print("sample indices:", indices)
print("sample csv frameNum:", indices + 1)
print("output shape:", frames.shape)