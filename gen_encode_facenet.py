import pickle

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from PIL import Image
import numpy as np
import os


workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=15,
    thresholds=[0.4, 0.5, 0.5], factor=0.709, post_process=True,
    device=device
)

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

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
aligned = []
for i, image_file in enumerate(image_files):
    print(f'Processing image {i+1}/{len(image_files)}: {image_file}')
    x = torch.tensor(np.array(Image.open(image_file).convert('RGB')))
    x_aligned, prob = mtcnn(x, return_prob=True)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)

aligned = torch.stack(aligned).to(device)
embeddings = resnet(aligned).detach().cpu()

# Define the path for saving the pickle file
output_file = 'assets/embedding/facenet_embeddings.pkl'

# Save the embeddings as a list of [name, embeddings] for each entry
embeddings_list = []
for i in range(len(names)):
    embeddings_list.append([names[i], embeddings[i].numpy()])

# Save the embeddings list as a pickled file
with open(output_file, 'wb') as f:
    pickle.dump(embeddings_list, f)
