import os
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import numpy as np
import pandas as pd
from skimage.io import imread
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread, imsave
from skimage.transform import resize
import seaborn as sns
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Task 12
torch.set_num_threads(2)

# Define the number of output labels
NUM_GALAXY_LABELS = 37

# --- 1. Define the Custom Dataset Class (Must be top-level) ---

class GalaxyDataset(Dataset):
    """
    Custom PyTorch Dataset class to load galaxy images and vote fraction labels.
    """
    def __init__(self, df, img_dir, transform=None):
        # Reset index to ensure integer indexing works with .iloc
        self.df = df.reset_index(drop=True) 
        self.img_dir = img_dir
        self.transform = transform
        
        # Pre-extract GalaxyID + label vectors
        self.ids = self.df["GalaxyID"].values
        # Filter for all columns starting with "Class"
        self.label_columns = [col for col in df.columns if col.startswith("Class")]
        self.labels = self.df[self.label_columns].values.astype("float32")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        galaxy_id = self.ids[idx]
        img_path = os.path.join(self.img_dir, f"{galaxy_id}.jpg")
        
        # Load the 64x64 downsized image
        img = imread(img_path) # (64,64,3) uint8
        
        # Normalize to [0,1]
        img = img.astype("float32") / 255 
        
        # Convert to PyTorch tensor and re-order dimensions: (H, W, C) → (C, H, W)
        img = torch.tensor(img).permute(2,0,1) 
        
        # Load the corresponding label vector
        label = torch.tensor(self.labels[idx])

        if self.transform:
            img = self.transform(img)
            
        return img, label

# --- 2. Custom CNN Model for Task 16 ---

class CustomGalaxyCNN(nn.Module):
    """
    Custom CNN based on a simple 4-block architecture.
    It is designed to handle 64x64 input images (adjusting the flattened size).
    """
    def __init__(self, num_labels=NUM_GALAXY_LABELS):
        super(CustomGalaxyCNN, self).__init__()
        
        # Block 1: Input (3 channels, e.g., RGB) -> 16 filters
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2: 16 -> 32 filters
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        
        # Block 3: 32 -> 64 filters
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # Block 4: 64 -> 128 filters
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Calculate the size after all convolutions and poolings (for 64x64 input):
        # 64 -> (conv1, stride 2) 32 -> (pool 2) 16 -> (pool 2) 8 -> (pool 2) 4 -> (pool 2) 2
        # Final feature map size is 128 channels * 2 * 2 pixels = 512 features
        self.flattened_features = 128 * 2 * 2 
        
        # Dense Layer 1
        self.fc1 = nn.Linear(self.flattened_features, 512)
        self.dropout = nn.Dropout(p=0.5) 
        
        # Output Layer
        self.fc_output = nn.Linear(512, num_labels)


    def forward(self, x):
        # x shape: [Batch_Size, 3, 64, 64]
        
        # Block 1: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv1(x)))
        
        # Block 2: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv2(x)))
        
        # Block 3: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv3(x)))
        
        # Block 4: Conv -> ReLU -> Pool
        x = self.pool(F.relu(self.conv4(x)))
        
        # Flatten
        x = torch.flatten(x, 1)

        # Dense layers
        x = self.dropout(F.relu(self.fc1(x)))
        
        # Output layer with Sigmoid
        x = torch.sigmoid(self.fc_output(x))
        
        return x

# --- 3. Load Data and Define Directories ---

filename = "training_classifications.csv" 
new_image_directory = "downsized_images" 

# Load the main data frame
try:
    data = pd.read_csv(filename)
except FileNotFoundError:
    print("ERROR: CSV file not found. Please check the 'filename' path.")
    raise

# --- 4. Split the Data into Training and Validation Sets (Task 12) ---

all_indices = data.index
train_indices, val_indices = train_test_split(
    all_indices,
    test_size=0.2,    # 20% for validation
    random_state=42,    
    shuffle=True        
)

train_df = data.loc[train_indices]
val_df = data.loc[val_indices]

print(f"Data split: Training ({len(train_df)}), Validation ({len(val_df)})")


# --- 5. Create Dataset and DataLoader instances (Task 11) ---

# Create the separate Dataset instances
train_dataset = GalaxyDataset(df=train_df, img_dir=new_image_directory)
val_dataset = GalaxyDataset(df=val_df, img_dir=new_image_directory)

# Create the DataLoader instances
train_loader = DataLoader(
    train_dataset,
    batch_size=128, 
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=128, 
    shuffle=False,
    num_workers=2, 
    pin_memory=True
)


# --- Configuration ---
N_LABELS = 37
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# --- Instantiate the Custom CNN Model (Task 16) ---
model = CustomGalaxyCNN(N_LABELS).to(DEVICE)
print(f"Custom CNN model successfully loaded for {N_LABELS} outputs on {DEVICE}.")

# --- Loss Function and Optimizer ---
# Define the RMSE loss function
def rmse_loss(outputs, targets):
    """Calculates the Root Mean Squared Error (RMSE) loss."""
    # F.mse_loss automatically calculates the mean of squared differences
    mse = F.mse_loss(outputs, targets, reduction='mean') 
    # The L_RMSE is the square root of the final mean
    rmse = torch.sqrt(mse) 
    return rmse

# Since the training loop uses a standard MSELoss criterion, we can redefine it:
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- Training History Tracking ---
history = {
    'train_loss': [], 
    'val_loss': [], 
    'train_images_seen': []
}

# --- Training Loop Setup ---
NUM_EPOCHS = 10 # This is now set to 10 epochs as requested
best_val_loss = float('inf')
global_step_count = 0

print("Starting training of Custom CNN...")

# 

for epoch in range(NUM_EPOCHS):
    # --- TRAINING PHASE ---
    model.train() # Set model to training mode
    running_loss = 0.0
    
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad() # Clear the gradients

        outputs = model(images) # Forward pass
        loss = criterion(outputs, labels) # Calculate the loss (MSE)

        loss.backward() # Backward pass
        optimizer.step() # Update parameters

        # Track metrics
        running_loss += loss.item() * images.size(0)
        global_step_count += images.size(0)

        # Optional: Log batch loss periodically (converted to L_RMSE)
        if (i + 1) % 100 == 0:
            current_loss = np.sqrt(loss.item()) # L_RMSE for this batch
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Batch {i+1}, L_RMSE: {current_loss:.4f}")

    # Calculate Epoch Training L_RMSE
    epoch_train_mse = running_loss / len(train_loader.dataset)
    epoch_train_l_rmse = np.sqrt(epoch_train_mse)
    
    history['train_loss'].append(epoch_train_l_rmse)
    history['train_images_seen'].append(global_step_count)


    # --- VALIDATION PHASE ---
    model.eval() # Set model to evaluation mode
    val_loss = 0.0
    
    with torch.no_grad(): # Disable gradient calculations during validation
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(images)
            loss = criterion(outputs, labels) # Calculate the loss (MSE)
            val_loss += loss.item() * images.size(0)

    # Calculate Epoch Validation L_RMSE
    epoch_val_mse = val_loss / len(val_loader.dataset)
    epoch_val_l_rmse = np.sqrt(epoch_val_mse)
    
    history['val_loss'].append(epoch_val_l_rmse)

    print(f"--- Epoch {epoch+1}: Train L_RMSE: {epoch_train_l_rmse:.4f} | Val L_RMSE: {epoch_val_l_rmse:.4f} ---")

    # --- SAVE NETWORK (Early Stopping Check) ---
    if epoch_val_l_rmse < best_val_loss:
        best_val_loss = epoch_val_l_rmse
        # Save the best model state (renamed to reflect custom CNN)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'history': history
        }, 'custom_cnn_best_model.pth')
        print("Model saved! Validation loss improved.")
    
    # Check for early stopping (if validation loss increases for two consecutive epochs after epoch 2)
    elif epoch > 2 and epoch_val_l_rmse > history['val_loss'][-2] and epoch_val_l_rmse > history['val_loss'][-3]:
        print("Validation loss increasing for two consecutive epochs. Stopping early.")
        break
        
print("Training finished.")
