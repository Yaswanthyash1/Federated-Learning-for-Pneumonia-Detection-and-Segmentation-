import os
import shutil
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
import cv2

# --- Partitioning Methods ---

def dirichlet_partition(dataset, num_partitions, alpha=0.5):
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        indices = dataset.indices
    else:
        original_dataset = dataset
        indices = np.arange(len(dataset))
    targets = np.array(original_dataset.data["Target"])[indices]
    num_classes = len(set(targets))

    indices_by_class = {i: np.where(targets == i)[0].tolist() for i in range(num_classes)}

    partitions = [[] for _ in range(num_partitions)]
    
    for c in range(num_classes):
        np.random.shuffle(indices_by_class[c])
        proportions = np.random.dirichlet([alpha] * num_partitions)
        proportions = (proportions * len(indices_by_class[c])).astype(int)
        proportions[-1] = len(indices_by_class[c]) - sum(proportions[:-1])

        start = 0
        for i in range(num_partitions):
            end = start + proportions[i]
            partitions[i].extend([indices[idx] for idx in indices_by_class[c][start:end]])
            start = end

    return partitions

def stratified_partition(dataset, num_partitions):
    if isinstance(dataset, Subset):
        original_dataset = dataset.dataset
        indices = dataset.indices
    else:
        original_dataset = dataset
        indices = np.arange(len(dataset))

    targets = np.array(original_dataset.data["Target"])[indices]
    num_classes = len(set(targets))

    indices_by_class = {i: np.where(targets == i)[0].tolist() for i in range(num_classes)}

    partitions = [[] for _ in range(num_partitions)]
    
    for c in range(num_classes):
        np.random.shuffle(indices_by_class[c])
        partition_size = len(indices_by_class[c]) // num_partitions
        for i in range(num_partitions):
            start = i * partition_size
            end = (i + 1) * partition_size if i < num_partitions - 1 else len(indices_by_class[c])
            partitions[i].extend([indices[idx] for idx in indices_by_class[c][start:end]])

    return partitions

def hybrid_partition(dataset, num_partitions, alpha=0.5, stratified_ratio=0.5):
    total_samples = len(dataset)
    stratified_samples = int(total_samples * stratified_ratio)
    dirichlet_samples = total_samples - stratified_samples

    indices = np.arange(total_samples)
    np.random.shuffle(indices)

    stratified_indices = indices[:stratified_samples]
    dirichlet_indices = indices[stratified_samples:]

    stratified_partitions = stratified_partition(Subset(dataset, stratified_indices), num_partitions)
    dirichlet_partitions = dirichlet_partition(Subset(dataset, dirichlet_indices), num_partitions, alpha)

    partitions = []
    for i in range(num_partitions):
        combined_partition = stratified_partitions[i] + dirichlet_partitions[i]
        partitions.append(combined_partition)

    return partitions

# --- Dataset Preparation and Saving ---

base_path = "dataset_nnunet"
output_base = "archive_v4/nnUNet_raw_data_base"
num_datasets = 11
images_per_dataset = 600

alpha = 0.5
stratified_ratio = 0.8

os.makedirs(output_base, exist_ok=True)
os.makedirs("archive_v4/nnUNet_preprocessed", exist_ok=True)
os.makedirs("archive_v4/nnUNet_trained_models", exist_ok=True)


all_images = sorted(os.listdir(os.path.join(base_path, "imagesTr")))
all_masks = sorted(os.listdir(os.path.join(base_path, "labelsTr")))

max_total_images =int(num_datasets * images_per_dataset *(1.1))
if len(all_images) > max_total_images:
    all_images = all_images[:max_total_images]
    all_masks = all_masks[:max_total_images]

# Create dummy dataset object
class DummyDataset:
    def __init__(self, targets):
        self.data = {"Target": targets}
    def __len__(self):
        return len(self.data["Target"])

# Example: simulate 2 classes (0, 1) using mask filenames
# Adjust this to load actual targets from metadata or filenames
# simulated_targets = [1 if "pneumonia" in fname.lower() else 0 for fname in all_masks]
from PIL import Image

simulated_targets = []
for fname in all_masks:
    mask_path = os.path.join(base_path, "labelsTr", fname)
    mask = np.array(Image.open(mask_path))
    simulated_targets.append(1 if np.any(mask == 255) else 0)

dataset = DummyDataset(simulated_targets)

partition_indices = hybrid_partition(dataset, num_datasets, alpha, stratified_ratio)
#print len of each partition
for i, indices in enumerate(partition_indices):
    print(f"Partition {i}: {len(indices)} images")

# def create_nnunet_dataset(dataset_idx, indices):
#     dataset_name = f"Dataset{str(dataset_idx+1).zfill(3)}_Pneumonia"
#     dataset_path = os.path.join(output_base, dataset_name)

#     os.makedirs(os.path.join(dataset_path, "imagesTr"), exist_ok=True)
#     os.makedirs(os.path.join(dataset_path, "labelsTr"), exist_ok=True)
#     os.makedirs(os.path.join(dataset_path, "imagesTs"), exist_ok=True)

#     dataset_images = sorted([all_images[i] for i in indices])
#     dataset_masks = sorted([all_masks[i] for i in indices])

#     masked=0
#     unmasked=0
    
#     for i, (img_name, mask_name) in enumerate(zip(dataset_images, dataset_masks)):
#         mask_path = os.path.join(base_path, "labelsTr", mask_name)
#         mask = np.array(Image.open(mask_path))
#         if np.any(mask > 0):
#             masked += 1
#             mask_path = os.path.join(base_path, "labelsTr", mask_name)
#             new_mask_name = f"chest_{str(i).zfill(5)}.png"
#             shutil.copy(mask_path, os.path.join(dataset_path, "labelsTr", new_mask_name))

#             img_path = os.path.join(base_path, "imagesTr", img_name)
#             new_img_name = f"chest_{str(i).zfill(5)}_0000.png"
#             shutil.copy(img_path, os.path.join(dataset_path, "imagesTr", new_img_name))
#         else:
#             if masked >= unmasked:
#                 unmasked += 1
#                 mask_path = os.path.join(base_path, "labelsTr", mask_name)
#                 new_mask_name = f"chest_{str(i).zfill(5)}.png"
#                 shutil.copy(mask_path, os.path.join(dataset_path, "labelsTr", new_mask_name))

#                 img_path = os.path.join(base_path, "imagesTr", img_name)
#                 new_img_name = f"chest_{str(i).zfill(5)}_0000.png"
#                 shutil.copy(img_path, os.path.join(dataset_path, "imagesTr", new_img_name))


#     dataset_info = {
#         "name": dataset_name,
#         "description": f"RSNA Pneumonia Detection Challenge - Partition {dataset_idx+1}",
#         "reference": "RSNA",
#         "license": "CC BY-NC-SA 4.0",
#         "release": "1.0",
#         "tensorImageSize": "2D",
#         "channel_names": {
#             "0": "CT"
#         },
#         "labels": {
#             "background": 0.0,
#             "pneumonia": 1.0
#         },
#         "file_ending": ".png",
#         "numTraining": len(dataset_images),
#         "numTest": 0,
#         "training": [{"image": f"./imagesTr/chest_{str(i).zfill(5)}_0000.png",
#                       "label": f"./labelsTr/chest_{str(i).zfill(5)}.png"}
#                      for i in range(len(dataset_images))],
#         "test": []
#     }

#     save_json(dataset_info, os.path.join(dataset_path, "dataset.json"))
#     return dataset_path

def create_nnunet_dataset(dataset_idx, indices):
    dataset_name = f"Dataset{(dataset_idx+1):03d}_Pneumonia"
    dataset_path = os.path.join(output_base, dataset_name)

    os.makedirs(os.path.join(dataset_path, "imagesTr"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "labelsTr"), exist_ok=True)
    os.makedirs(os.path.join(dataset_path, "imagesTs"), exist_ok=True)

    dataset_images = sorted([all_images[i] for i in indices])
    dataset_masks = sorted([all_masks[i] for i in indices])

    masked = 0
    unmasked = 0
    used_count = 0
    training_entries = []

    for i, (img_name, mask_name) in enumerate(zip(dataset_images, dataset_masks)):
        mask_path = os.path.join(base_path, "labelsTr", mask_name)
        mask = np.array(Image.open(mask_path))

        include = False
        if np.any(mask > 0):
            if True:
                masked += 1
                include = True
        else:
            if masked >= unmasked:
                unmasked += 1
                include = True

        if include:
            new_mask_name = f"chest_{str(used_count).zfill(5)}.png"
            new_img_name = f"chest_{str(used_count).zfill(5)}_0000.png"

            shutil.copy(mask_path, os.path.join(dataset_path, "labelsTr", new_mask_name))
            img_path = os.path.join(base_path, "imagesTr", img_name)
            shutil.copy(img_path, os.path.join(dataset_path, "imagesTr", new_img_name))

            training_entries.append({
                "image": f"./imagesTr/{new_img_name}",
                "label": f"./labelsTr/{new_mask_name}"
            })

            used_count += 1

    dataset_info = {
        "name": dataset_name,
        "description": f"RSNA Pneumonia Detection Challenge - Partition {dataset_idx+1}",
        "reference": "RSNA",
        "license": "CC BY-NC-SA 4.0",
        "release": "1.0",
        "tensorImageSize": "2D",
        "channel_names": {
            "0": "CT"
        },
        "labels": {
            "background": 0.0,
            "pneumonia": 1.0
        },
        "file_ending": ".png",
        "numTraining": used_count,
        "numTest": 0,
        "training": training_entries,
        "test": []
    }
    print(f"Masked: {masked}, Unmasked: {unmasked}, Used Count: {used_count}")
    save_json(dataset_info, os.path.join(dataset_path, "dataset.json"))
    return dataset_path


# Generate all partitions
for dataset_idx in range(num_datasets):
    print(f"Creating {dataset_idx+1:02d}/10: {len(partition_indices[dataset_idx])} images")
    create_nnunet_dataset(dataset_idx, partition_indices[dataset_idx])

print("Hybrid dataset partitioning complete!")


for dataset_idx in range(num_datasets):
    dataset_name = f"Dataset{str(dataset_idx+1).zfill(3)}_Pneumonia"
    dataset_path = os.path.join(output_base, dataset_name)
    labels_dir = os.path.join(dataset_path, "labelsTr")
    print(f"Processing labels in {labels_dir}")

    for filename in os.listdir(labels_dir):
        filepath = os.path.join(labels_dir, filename)

        #image in grayscale mode
        label = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

        # all pixels value 255 to 1
        label[label == 255] = 1
        cv2.imwrite(filepath, label)

print("Label conversion complete!")

