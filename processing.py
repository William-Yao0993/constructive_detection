import json 
from sklearn.model_selection import train_test_split
import os

with open ('datasets/data.json','r') as file:
    data = json.load(file)


annotations = data['shapes']

# Split the list into training, validation, and test sets
train, test = train_test_split(annotations, test_size=0.3)
val, test = train_test_split(test, test_size=0.5)

# Define the directory where you want to save the split datasets
base_dir = 'datasets\labelme_json_dir'

# Create the directories if they do not exist
os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)

# Write the subsets to their respective JSON files
def write_annotations(annotations, filename):
    # Create a new dictionary to store the data
    new_data = {
        "version": data['version'],
        "flags": data['flags'],
        "shapes": annotations,
        # Add any other keys that need to be preserved from the original data
    }
    with open(filename, 'w') as file:
        json.dump(new_data, file, indent=4)

write_annotations(train, os.path.join(base_dir, 'train', 'train_data.json'))
write_annotations(val, os.path.join(base_dir, 'val', 'val_data.json'))
write_annotations(test, os.path.join(base_dir, 'test', 'test_data.json'))