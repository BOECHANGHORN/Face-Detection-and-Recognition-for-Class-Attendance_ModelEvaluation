import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import pickle
import os
from PIL import Image
from prettytable import PrettyTable
import time

path_str_list = ['test_base', 'test_nolook', 'test_specs', 'test_mask', 'test_hat_specs', 'test_3ppl', 'test_dim']

# Change path here
path_str = path_str_list[5]

print("Facenet\n")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load embeddings from pickle file
with open('assets/embedding/facenet_embeddings.pkl', 'rb') as f:
    embeddings_list = pickle.load(f)

with open(f'assets/groundtruth/{path_str}.pkl', 'rb') as f:
    groundtruth = pickle.load(f)


# Load all images from 'cropped_faces' directory and sort by frame number
images_dir = f'assets/output/HOG_{path_str}'
image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
image_files = sorted(image_files, key=lambda x: int(x.split('_')[0][5:]))

# Initialize variables for TP, FP, and FN
TP = 0
FP = 0
FN = 0

# Create a table to store the results
table = PrettyTable()
table.field_names = ["No", "Image Name", "Predicted Name", "Distance", "Time (s)", "Real Name", "Correct"]

start_time = time.time()  # Start timestamp
inner_time = start_time

correct = 0

# Process each image and find the closest match in embeddings list
for i, filename in enumerate(image_files):
    image_path = os.path.join(images_dir, filename)

    # Load image and get embedding
    img = Image.open(image_path).convert('RGB')
    img = img.resize((160, 160))
    img = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255
    embedding = resnet(img).detach().cpu()

    # Find the closest match in embeddings list
    min_dist = float('inf')
    match_name = 'None'
    distances = []  # initialize list to store distances
    for name, embedding_in_list in embeddings_list:
        dist = (embedding - embedding_in_list).norm().item()
        distances.append(dist)  # append distance to list
        if dist < min_dist and dist < 0.8:
            min_dist = dist
            match_name = name

    # Print the result
    print(f"{filename}: {match_name} : {min_dist}")

    # Calculate time spent for processing the image
    elapsed_time = time.time() - inner_time

    # Get the real name from the groundtruth
    real_name = groundtruth[i][1]

    # Determine if the prediction is correct
    is_correct = match_name == real_name

    # Increment TP, FP, and FN counts based on the comparison
    if is_correct:
        TP += 1
        correct += 1
    elif match_name != 'None' and not is_correct:
        FP += 1
    elif match_name == 'None' and not is_correct:
        FN += 1

    # Add the result to the table
    table.add_row([i + 1, filename, match_name, min_dist, format(elapsed_time, ".2f"), real_name, is_correct])

    inner_time = time.time()

# Print the table
print(table)

# End timestamp
end_time = time.time()

# Calculate total time spent and print it
total_time = end_time - start_time
print(f"Total time spent: {total_time:.2f} seconds\n")

print("Facenet\n")

# Calculate and print total accuracy
accuracy = correct / len(image_files)
print(f"{correct} correct out of {len(image_files)}")
print(f"Total accuracy: {accuracy: 4f}")

# Calculate precision, recall, and F1 score
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print(f"Total precision: {precision: 4f}")
print(f"Total recall: {recall: 4f}")
print(f"Total f1 score: {f1_score: 4f}")
