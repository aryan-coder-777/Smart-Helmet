import cv2
import sys
import time

print("--- OpenCV VideoCapture Diagnostics ---")

# Try camera indexes
for index in [0, 1, 2]:
    print(f"\nTesting camera index: {index}")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Index {index} could NOT be opened.")
        continue
    
    print(f"Index {index} opened successfully.")
    
    # Try setting standard resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    print("Reading 5 frames to check capture...")
    for i in range(5):
        ret, frame = cap.read()
        if ret:
            print(f"  Frame {i+1}: Success, shape={frame.shape}")
        else:
            print(f"  Frame {i+1}: Failed to read frame")
        time.sleep(0.1)
        
    cap.release()

print("\nDiagnostics complete.")
