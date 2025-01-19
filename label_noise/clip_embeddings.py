import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from transformers import CLIPProcessor, CLIPModel
import argparse
import numpy as np

# Parse arguments
parser = argparse.ArgumentParser(description="Compute CLIP embeddings for CIFAR datasets.")
parser.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True, 
                    help="Specify the dataset to use: cifar10 or cifar100.")
args = parser.parse_args()


# Define the device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Transform the dataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to 224x224 as expected by CLIP
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  
                         std=[0.26862954, 0.26130258, 0.27577711])  
])

# Load the dataset based on the argument
if args.dataset == "cifar10":
    dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataset_name = "cifar10"
elif args.dataset == "cifar100":
    dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    dataset_name = "cifar100"

train_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=2)

# Load the CLIP model
model_name = "openai/clip-vit-base-patch32"
model = CLIPModel.from_pretrained(model_name)
model.to(device)
model.eval()

# Compute CLIP embeddings
clip_embeddings = []
labels = []

with torch.no_grad():
    for images, targets in train_loader:
        images = images.to(device)
        outputs = model.get_image_features(pixel_values=images)
        embeddings = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        embeddings = embeddings.cpu().numpy()
        clip_embeddings.append(embeddings)
        labels.extend(targets.numpy())

clip_embeddings = np.concatenate(clip_embeddings, axis=0)
labels = np.array(labels)

print("CLIP Embeddings shape:", clip_embeddings.shape)
print("Labels shape:", labels.shape)

# Save embeddings and labels to a file
output_filename = f"{args.dataset}-clip-embeddings.pkl"
with open(output_filename, "wb") as f:
    pickle.dump(clip_embeddings, f)