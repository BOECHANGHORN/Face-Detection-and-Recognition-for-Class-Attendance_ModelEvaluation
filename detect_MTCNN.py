from facenet_pytorch import MTCNN
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw
import os
import time

path_str_list = ['test_base', 'test_nolook', 'test_specs', 'test_mask', 'test_hat_specs', 'test_3ppl', 'test_dim']
# Change path here
path_str = path_str_list[0]

if not os.path.exists(f"assets/output/MTCNN_{path_str}"):
    os.makedirs(f"assets/output/MTCNN_{path_str}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(keep_all=True, device=device)

video = cv2.VideoCapture(f'assets/videos/{path_str}.mp4')
frames = []
total_faces = 0

start_time = time.time()  # Start timestamp

while True:
    success, frame = video.read()
    if not success:
        break
    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))

frames_tracked = []
for i, frame in enumerate(frames):
    print('\rProcessing frame: {}'.format(i + 1), end='')

    # Detect faces
    boxes, _ = mtcnn.detect(frame)
    if boxes is not None:
        num_faces = len(boxes)
        total_faces += num_faces

        # Print total number of faces in current frame
        print(f', current face: {num_faces}')

        # Crop and save faces
        for j, box in enumerate(boxes):
            # Crop face
            face = frame.crop(box.tolist())
            # Save face as image
            filename = f"assets/output/MTCNN_{path_str}/frame{i + 1}_face{j + 1}.jpg"  # Frame number and face number
            face.save(filename)

        # Draw faces
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

        # Add to frame list
        frames_tracked.append(frame_draw.resize((640, 360), Image.Resampling.BILINEAR))
    else:
        print(', no faces detected.')

end_time = time.time()  # End timestamp
total_time = end_time - start_time  # Total time spent

print(f'\nDone. Total faces detected: {total_faces}')
print(f'Total time spent: {total_time:.2f} seconds')

# Save tracked video
dim = frames_tracked[0].size
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_tracked = cv2.VideoWriter(f'assets/output/MTCNN_{path_str}.mp4', fourcc, 25.0, dim)
for frame in frames_tracked:
    video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
video_tracked.release()
