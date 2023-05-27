import cv2
import numpy as np
from PIL import Image, ImageDraw
import os
import time

path_str_list = ['test_base', 'test_nolook', 'test_specs', 'test_mask', 'test_hat_specs', 'test_3ppl', 'test_dim']

for test in range(7):
    # Change path here
    path_str = path_str_list[test]

    if not os.path.exists(f"assets/output/HC_{path_str}"):
        os.makedirs(f"assets/output/HC_{path_str}")

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

        # Convert frame to grayscale
        gray = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2GRAY)

        # Detect faces using Haar Cascade
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        num_faces = len(faces)
        total_faces += num_faces

        # Print total number of faces in current frame
        print(f', current face: {num_faces}')

        # Crop and save faces
        for j, (x, y, w, h) in enumerate(faces):
            # Crop face
            face = frame.crop((x, y, x + w, y + h))
            # Save face as image
            filename = f"assets/output/HC_{path_str}/frame{i + 1}_face{j + 1}.jpg"  # Frame number and face number
            face.save(filename)

            # Draw faces
            draw = ImageDraw.Draw(frame)
            draw.rectangle([(x, y), (x + w, y + h)], outline=(255, 0, 0), width=6)

        # Add to frame list
        frames_tracked.append(frame.resize((640, 360), Image.Resampling.BILINEAR))

    end_time = time.time()  # End timestamp
    total_time = end_time - start_time  # Total time spent

    print(f'\nDone. Total faces detected: {total_faces}')
    print(f'Total time spent: {total_time:.2f} seconds')

    # Save tracked video
    dim = frames_tracked[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_tracked = cv2.VideoWriter(f'assets/output/HC_{path_str}.mp4', fourcc, 25.0, dim)
    for frame in frames_tracked:
        video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video_tracked.release()
