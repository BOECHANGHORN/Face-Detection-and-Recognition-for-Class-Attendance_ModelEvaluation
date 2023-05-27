import json

import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
import pickle
import os
from PIL import Image
from prettytable import PrettyTable
import time

path_str_list = ['test_base', 'test_nolook', 'test_specs', 'test_mask', 'test_hat_specs', 'test_3ppl', 'test_dim']

for test in range(7):
    # Change path here
    path_str = path_str_list[test]

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
    TN = 0
    FP = 0
    FN = 0

    # Create a table to store the results
    table = PrettyTable()
    table.field_names = ["No", "Image Name", "Predicted Name", "Distance", "Time (s)", "Real Name", "Correct"]

    start_time = time.time()  # Start timestamp
    inner_time = start_time

    table_list = []

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
        elif match_name != 'None' and not is_correct:
            FP += 1
        elif match_name == 'None' and not is_correct:
            FN += 1
        else:
            TN += 1     # TN is included in TP

        # Add the result to the table
        table.add_row([i + 1, filename, match_name, min_dist, format(elapsed_time, ".2f"), real_name, is_correct])
        table_list.append([i + 1, filename, match_name, min_dist, format(elapsed_time, ".2f"), real_name, is_correct])

        inner_time = time.time()

    # Print the table
    print(table)

    # End timestamp
    end_time = time.time()

    # Calculate total time spent and print it
    total_time = end_time - start_time
    print(f"Total time spent: {total_time:.2f} seconds\n")

    print("Facenet\n")
    print(path_str + '\n')

    # Calculate and print total accuracy
    accuracy = (TP + TN) / max((TP + FP + TN + FN), 1)
    print(f"{TP} correct out of {len(image_files)}")
    print(f"Total accuracy: {accuracy: 4f}")

    # Calculate precision, recall, and F1 score
    precision = TP / max((TP + FP), 1)
    recall = TP / max((TP + FN), 1)
    f1_score = 2 * (precision * recall) / max((precision + recall), 1)

    print(f"Total precision: {precision: 4f}")
    print(f"Total recall: {recall: 4f}")
    print(f"Total f1 score: {f1_score: 4f}")

    output_data = {
        "Model Name": "Facenet",
        "Dataset Name": path_str,
        "Total Time Spent": total_time,
        "Correct": TP,
        "Total Images": len(image_files),
        "Total Accuracy": accuracy,
        "Total Precision": precision,
        "Total Recall": recall,
        "Total F1 Score": f1_score,
        "Table Data": table_list
    }

    # Create the directory if it doesn't exist
    output_dir = "assets/model_result"
    os.makedirs(output_dir, exist_ok=True)

    # Save the output data as JSON
    output_file = f"{output_dir}/facenet_{path_str}.json"
    with open(output_file, "w") as f:
        json.dump(output_data, f)
