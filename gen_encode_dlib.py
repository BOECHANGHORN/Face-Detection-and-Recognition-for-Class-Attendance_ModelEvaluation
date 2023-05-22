import pickle

import cv2
import face_recognition
import os


def get_encodings(image_list):
    encode_list = []
    for image in image_list:
        image = cv2.resize(image, (0, 0), None, 0.25, 0.25)
        # Convert color space from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Calculate the 128-dimensional face encodings for the first face detected
        encode = face_recognition.face_encodings(image)
        if encode:
            encode_list.append(encode[0])

    return encode_list


# Define the paths of the image directories and corresponding names
image_paths = {
    'Boe': 'assets/images/Boe',
    'Eugene': 'assets/images/Eugene',
    'Chang': 'assets/images/Chang'
}

# Collect the image paths and names from the directories
image_files = []
names = []
for name, path in image_paths.items():
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            image_files.append(os.path.join(path, filename))
            names.append(name)

# Process the images and compute the embeddings
embeddings = []
for i, image_file in enumerate(image_files):
    print(f'Processing image {i + 1}/{len(image_files)}: {image_file}')

    image = cv2.imread(image_file)

    face_encodings = get_encodings([image])

    if face_encodings:
        print('Face detected')
        embeddings.extend(face_encodings)
    else:
        print('No face detected')

# Define the path for saving the pickle file
output_file = 'assets/embedding/dlib_embeddings.pkl'

# Save the embeddings as a list of [name, embeddings] for each entry
embeddings_list = []
for i in range(len(names)):
    embeddings_list.append([names[i], embeddings[i]])

# Save the embeddings list as a pickled file
with open(output_file, 'wb') as f:
    pickle.dump(embeddings_list, f)
