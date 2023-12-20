import json 
from sklearn.model_selection import train_test_split
import os
# Open the Json format Dataset
with open ('datasets/data.json','r') as file:
    data = json.load(file)


annotations = data['shapes']

# Classes
labels = ['column_1','column_2','column_3','column_4', 
         'negative rebar',
         'hollow_1', 'hollow_2', 
         'lift', 'stair', 'air_conditionor',
         'column_vertical', 'column_horizontal']

annotation_dict = {label: [] for label in labels}


# Gather instances by label
for item in annotations:
    label = item['label']
    if (label != 'rebar'):
        # Hold out rebar
        annotation_dict[label].append(item)

for label, items in annotation_dict.items():
    print(f"Class {label} has {len(items)} annotations.")

train_set , valid_set, test_set= [], [], []
for label, instance in annotation_dict.items():
    if (len(instance) < 3):
        # Add the instance into three set if there is only one
        # to make sure the completeness of .yaml file later 
        train_set.extend(instance)
        test_set.extend(instance)
        valid_set.extend(instance)
    else:
        # 
        train, test = train_test_split(instance, test_size= 0.2, random_state=42)
        valid,test = train_test_split(test, test_size= 0.5, random_state=42)
        
        # Add Splitting results
        train_set.extend(train)
        valid_set.extend(valid)
        test_set.extend(test)

# Define the directory where you want to save the split datasets
base_dir = 'datasets\labelme_json_dir'

# Create the directories if they do not exist
os.makedirs(os.path.join(base_dir, 'train'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'val'), exist_ok=True)
os.makedirs(os.path.join(base_dir, 'test'), exist_ok=True)


# Write each partition back
data['shape'] = train_set
with open (os.path.join(base_dir, 'train', 'train_data.json'), 'w') as file:
    json.dump(data, file, indent= 4)

data['shape'] = valid_set
with open (os.path.join(base_dir, 'val', 'val_data.json'), 'w') as file:
    json.dump(data, file, indent= 4)

data['shape'] = test_set
with open (os.path.join(base_dir, 'test', 'test_data.json'), 'w') as file:
    json.dump(data, file, indent= 4)
