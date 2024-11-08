import torch
import numpy as np
import pandas as pd
import os
import torch.nn as nn

# Define the neural network structure
class RelationNetwork(nn.Module):
    def __init__(self):
        super(RelationNetwork, self).__init__()
        self.fc1 = nn.Linear(100, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 100)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Function to load the models
def load_models(model_dir, num_models=6):
    models = []
    for i in range(num_models):
        model = RelationNetwork()
        model_path = os.path.join(model_dir, f'relation_network_{i + 1}.pth')
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
        models.append(model)
    return models

# Function to predict from a string of elements
def predict_from_string(element_string, models):
    # Map elements to numeric values (example mapping, adjust as needed)
    element_map = {'C': 1.0, 'H': 0.5}  # Add mappings for other elements if needed
    elements = np.array([element_map[element] for element in element_string if element in element_map])

    if len(elements) < 100:
        # Pad with zeros if less than 100
        elements = np.pad(elements, (0, 100 - len(elements)), 'constant')
    elif len(elements) > 100:
        # Trim to 100 if more than 100
        elements = elements[:100]

    features = torch.tensor(elements, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    # Collect predictions from each model
    predictions = {}
    with torch.no_grad():
        for i, model in enumerate(models):
            output = model(features)
            predictions[i] = output.numpy().flatten()  # Flatten output to 1D

    return predictions

# Main execution section
if __name__ == '__main__':
    model_dir = 'C:\\Users\\TJJ\\Desktop\\Protein_Prediction\\model'
    models = load_models(model_dir)

    # Example string of elements
    element_string = "CCCHH"  # Adjust this string as needed

    # Make predictions
    predictions = predict_from_string(element_string, models)

    # Create a DataFrame to display the predictions in a column format
    prediction_df = pd.DataFrame({
        'x': predictions[0],
        'y': predictions[1],
        'z': predictions[2],
        'bond_type': predictions[3],
        'bond_start': predictions[4],
        'bond_end': predictions[5]
    })

    # Set pandas options to display all columns and rows without truncation
    pd.set_option('display.max_columns', None)  # Show all columns
    pd.set_option('display.max_rows', None)     # Show all rows
    pd.set_option('display.max_colwidth', None) # Show full width of each column

    # Output the predictions in a tabular format
    print(prediction_df)