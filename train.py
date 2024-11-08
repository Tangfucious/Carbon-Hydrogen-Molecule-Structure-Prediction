import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# Function to load CSV files
def load_csv_files(directory):
    elements = []
    x = []
    y = []
    z = []
    bond_types = []
    bond_starts = []
    bond_ends = []

    for filename in os.listdir(directory):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path)

            required_columns = ['Element', 'X', 'Y', 'Z', 'Bond Type', 'Bond Start', 'Bond End']
            for col in required_columns:
                if col not in data.columns:
                    print(f"Warning: Missing column '{col}' in file '{filename}'. Skipping this file.")
                    break
            else:
                element_data = data['Element'].values
                x_data = data['X'].values
                y_data = data['Y'].values
                z_data = data['Z'].values
                bond_type_data = data['Bond Type'].values
                bond_start_data = data['Bond Start'].values
                bond_end_data = data['Bond End'].values

                elements.append(element_data)
                x.append(x_data)
                y.append(y_data)
                z.append(z_data)
                bond_types.append(bond_type_data)
                bond_starts.append(bond_start_data)
                bond_ends.append(bond_end_data)

    return (
        np.concatenate(elements) if elements else np.array([]),
        np.concatenate(x) if x else np.array([]),
        np.concatenate(y) if y else np.array([]),
        np.concatenate(z) if z else np.array([]),
        np.concatenate(bond_types) if bond_types else np.array([]),
        np.concatenate(bond_starts) if bond_starts else np.array([]),
        np.concatenate(bond_ends) if bond_ends else np.array([])
    )

# Custom dataset class
class MolecularDataset(Dataset):
    def __init__(self, elements, x, y, z, bond_types, bond_starts, bond_ends):
        self.elements = elements
        self.x = x
        self.y = y
        self.z = z
        self.bond_types = bond_types
        self.bond_starts = bond_starts
        self.bond_ends = bond_ends

    def __len__(self):
        return len(self.bond_types) // 100  # Each group consists of 100 samples

    def __getitem__(self, idx):
        # Return 100 input features and corresponding target values
        start_idx = idx * 100
        features = torch.tensor(self.elements[start_idx:start_idx + 100], dtype=torch.float32)  # 100 elements

        # Target values are X, Y, Z, Bond Type, Bond Start, Bond End
        target_x = torch.tensor(self.x[start_idx:start_idx + 100], dtype=torch.float32)
        target_y = torch.tensor(self.y[start_idx:start_idx + 100], dtype=torch.float32)
        target_z = torch.tensor(self.z[start_idx:start_idx + 100], dtype=torch.float32)
        target_bond_type = torch.tensor(self.bond_types[start_idx:start_idx + 100], dtype=torch.float32)
        target_bond_start = torch.tensor(self.bond_starts[start_idx:start_idx + 100], dtype=torch.float32)
        target_bond_end = torch.tensor(self.bond_ends[start_idx:start_idx + 100], dtype=torch.float32)

        return features, (target_x, target_y, target_z, target_bond_type, target_bond_start, target_bond_end)

# Define the neural network
class RelationNetwork(nn.Module):
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(100, 64)  # 100 input features
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 100)  # Output 100 elements (can be adjusted as needed)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Main execution section
if __name__ == '__main__':
    folder_path = r'C:\Users\TJJ\Desktop\Protein_Prediction\train_sets'  # Folder path

    # Load data
    elements, x, y, z, bond_types, bond_starts, bond_ends = load_csv_files(folder_path)

    # Create dataset and data loader
    dataset = MolecularDataset(elements, x, y, z, bond_types, bond_starts, bond_ends)
    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)  # Load one group of 100 samples at a time

    # Initialize six networks
    networks = [RelationNetwork() for _ in range(6)]  # Create 6 networks

    # Set loss function and optimizers
    criterion = nn.MSELoss()  # Choose an appropriate loss function based on the task
    optimizers = [optim.Adam(net.parameters(), lr=0.001) for net in networks]

    # Training loop
    num_epochs = 5000
    for epoch in range(num_epochs):
        for features, targets in trainloader:
            losses = []  # Initialize losses list for each batch

            # Unpack targets
            target_x, target_y, target_z, target_bond_type, target_bond_start, target_bond_end = targets

            for i, net in enumerate(networks):
                optimizers[i].zero_grad()  # Clear gradients
                outputs = net(features[0])  # Forward pass

                # Calculate loss based on network index
                if i == 0:
                    loss = criterion(outputs, target_x)
                elif i == 1:
                    loss = criterion(outputs, target_y)
                elif i == 2:
                    loss = criterion(outputs, target_z)
                elif i == 3:
                    loss = criterion(outputs, target_bond_type)
                elif i == 4:
                    loss = criterion(outputs, target_bond_start)
                elif i == 5:
                    loss = criterion(outputs, target_bond_end)

                loss.backward()  # Backpropagation
                optimizers[i].step()  # Update weights
                losses.append(loss.item())  # Append the loss to the list

        print(f'Epoch [{epoch + 1}/{num_epochs}], Losses: {losses}')

    # Save models
    model_dir = 'C:\\Users\\TJJ\\Desktop\\Protein_Prediction\\model'
    os.makedirs(model_dir, exist_ok=True)  # Create the directory to save models

    for i, net in enumerate(networks):
        model_path = os.path.join(model_dir, f'relation_network_{i + 1}.pth')
        torch.save(net.state_dict(), model_path)  # Save model's state dict
        print(f'Model {i + 1} saved to {model_path}')

# import os
# import pandas as pd
# import numpy as np
#
# # Function to load CSV files from a directory
# def load_csv_files(directory):
#     elements = []
#     x = []
#     y = []
#     z = []
#     bs = []  # Bond Start
#     be = []  # Bond End
#     bt = []  # Bond Type
#
#     for filename in os.listdir(directory):
#         if filename.endswith('.csv'):
#             file_path = os.path.join(directory, filename)
#             data = pd.read_csv(file_path)
#
#             # Check that necessary columns exist
#             required_columns = ['Element', 'X', 'Y', 'Z', 'Bond Start', 'Bond End', 'Bond Type']
#             for col in required_columns:
#                 if col not in data.columns:
#                     print(f"Warning: Missing column '{col}' in file '{filename}'. Skipping this file.")
#                     break
#             else:
#                 element_data = data['Element'].values
#                 x_data = data['X'].values
#                 y_data = data['Y'].values
#                 z_data = data['Z'].values
#                 bs_data = data['Bond Start'].values
#                 be_data = data['Bond End'].values
#                 bt_data = data['Bond Type'].values
#
#                 elements.append(element_data)
#                 x.append(x_data)
#                 y.append(y_data)
#                 z.append(z_data)
#                 bs.append(bs_data)
#                 be.append(be_data)
#                 bt.append(bt_data)
#
#     # Concatenate results only if lists are not empty
#     return (
#         np.concatenate(elements) if elements else np.array([]),
#         np.concatenate(x) if x else np.array([]),
#         np.concatenate(y) if y else np.array([]),
#         np.concatenate(z) if z else np.array([]),
#         np.concatenate(bs) if bs else np.array([]),
#         np.concatenate(be) if be else np.array([]),
#         np.concatenate(bt) if bt else np.array([])
#     )
#
# # Function to split data into groups of a specified size
# def split_data(data, group_size):
#     return [data[i:i + group_size] for i in range(0, len(data), group_size)]
#
# if __name__ == '__main__':
#     # Main execution
#     folder_path = r'C:\Users\TJJ\Desktop\Protein_Prediction\train_sets'  # Your folder path
#
#     # Load the data
#     elements, x, y, z, bs, be, bt = load_csv_files(folder_path)
#
#     # Print the loaded data shapes for verification
#     print(f"Loaded data shapes:\n"
#           f"Elements: {elements.shape}\n"
#           f"X: {x.shape}\n"
#           f"Y: {y.shape}\n"
#           f"Z: {z.shape}\n"
#           f"Bond Start: {bs.shape}\n"
#           f"Bond End: {be.shape}\n"
#           f"Bond Type: {bt.shape}\n")
#
#     # Split the data into groups of 100
#     group_size = 100
#     elements_groups = split_data(elements, group_size)
#     x_groups = split_data(x, group_size)
#     y_groups = split_data(y, group_size)
#     z_groups = split_data(z, group_size)
#     bs_groups = split_data(bs, group_size)
#     be_groups = split_data(be, group_size)
#     bt_groups = split_data(bt, group_size)
#
#     # Print all groups for each data type
#     for idx, group in enumerate(elements_groups):
#         print(f"\nGroup {idx + 1} Elements:")
#         print(group)
#
#     for idx, group in enumerate(x_groups):
#         print(f"\nGroup {idx + 1} X coordinates:")
#         print(group)
#
#     for idx, group in enumerate(y_groups):
#         print(f"\nGroup {idx + 1} Y coordinates:")
#         print(group)
#
#     for idx, group in enumerate(z_groups):
#         print(f"\nGroup {idx + 1} Z coordinates:")
#         print(group)
#
#     for idx, group in enumerate(bs_groups):
#         print(f"\nGroup {idx + 1} Bond Start:")
#         print(group)
#
#     for idx, group in enumerate(be_groups):
#         print(f"\nGroup {idx + 1} Bond End:")
#         print(group)
#
#     for idx, group in enumerate(bt_groups):
#         print(f"\nGroup {idx + 1} Bond Type:")
#         print(group)