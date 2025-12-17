import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import pandas as pd
import os
from skimage.io import imread 
from torch.utils.data import Dataset, DataLoader

# --- USER INPUT REQUIRED (FILL IN THESE 3 PATHS) ---
# 1. Path to your saved model weights
RESNET_AUG_SCHEDULER_PATH = 'resnet18_augmented_final.pth' 

# 2. Path to the folder containing the UNPACKED test images (e.g., test_images/)
TEST_IMAGES_DIR = 'test_images' 

# 3. Output path for the generated test set predictions CSV
OUTPUT_TEST_PREDICTIONS_PATH = 'test_model_predictions.csv' 
# -----------------------------------------------

# --- Configuration (From your training script) ---
N_LABELS = 37
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)

# --- Define the Custom Dataset Class (re-used for inference) ---
class TestGalaxyDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        # --- Dynamically find all image IDs in the directory ---
        self.ids = [
            int(f.replace('.jpg', '')) 
            for f in os.listdir(img_dir) 
            if f.endswith('.jpg')
        ]
        self.ids.sort()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        galaxy_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{galaxy_id}.jpg")
        
        img = imread(img_path)
        img = img.astype("float32") / 255 
        img = torch.tensor(img).permute(2,0,1) 
        
        # We return a dummy label as the test set has no true labels
        dummy_label = torch.zeros(N_LABELS) 
        
        return img, dummy_label, galaxy_id 

# --- Model Customization (Must match your trained model structure) ---
def create_custom_resnet18(num_classes):
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    class GalaxyResNet(nn.Module):
        def __init__(self, resnet_model):
            super().__init__()
            self.resnet = resnet_model
            self.sigmoid = nn.Sigmoid()
        def forward(self, x):
            x = self.resnet(x)
            x = self.sigmoid(x)
            return x
    return GalaxyResNet(model)

# --- Execution Entry Point ---
if __name__ == '__main__':
    # Load Data and Create Test Loader
    test_dataset = TestGalaxyDataset(img_dir=TEST_IMAGES_DIR)

    test_loader = DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0, # Use 0 to avoid multiprocessing errors
        pin_memory=True
    )

    print(f"--- Starting Inference on {len(test_dataset)} Test Images ---")

    # Instantiate and load model
    model = create_custom_resnet18(N_LABELS)
    try:
        # Use the corrected loading command
        checkpoint = torch.load(RESNET_AUG_SCHEDULER_PATH, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        print(f"Model loaded successfully onto {DEVICE}.")
    except FileNotFoundError:
        print(f"FATAL ERROR: Model checkpoint not found at {RESNET_AUG_SCHEDULER_PATH}")
        exit()

    all_predictions = []
    all_galaxy_ids = []

    with torch.no_grad():
        for images, _, galaxy_ids in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            
            predictions_np = outputs.cpu().numpy()
            all_predictions.append(predictions_np)
            all_galaxy_ids.extend(list(galaxy_ids.numpy())) 

    final_predictions = np.concatenate(all_predictions, axis=0)
    final_ids = np.array(all_galaxy_ids)

    # We need the column names to save the prediction CSV properly
    # Assuming standard GZ2 column names
    class_cols = [f'Class{i}.{j}' for i, j in [(1,1), (1,2), (1,3), (2,1), (2,2), (3,1), (3,2), (4,1), (4,2), (5,1), (5,2), (5,3), (5,4), (6,1), (6,2), (7,1), (7,2), (7,3), (8,1), (8,2), (8,3), (8,4), (8,5), (8,6), (8,7), (9,1), (9,2), (9,3), (10,1), (10,2), (10,3), (11,1), (11,2), (11,3), (11,4), (11,5), (11,6)]]
    
    predictions_df = pd.DataFrame(final_predictions, columns=class_cols)
    predictions_df.insert(0, 'GalaxyID', final_ids)

    predictions_df.to_csv(OUTPUT_TEST_PREDICTIONS_PATH, index=False)

    print(f"\nSUCCESS: Test set predictions saved to {OUTPUT_TEST_PREDICTIONS_PATH}")