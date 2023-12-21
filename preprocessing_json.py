import json 
from sklearn.model_selection import train_test_split
import os
from collections import OrderedDict
# Open the Json format Dataset
with open ('datasets/data.json','r') as file:
    data = json.load(file)



# Hold out rebar
data_without_rebar = [i for i in data['shapes'] if i['label'] != 'rebar']

data['shapes'] = data_without_rebar

# Get the Class_id order
ordered_label = OrderedDict()
for shape in data['shapes']:
    label = shape['label']
    if (label not in ordered_label):
        ordered_label[label] = None
ordered_label_list = list(ordered_label.keys())
print(ordered_label_list)



# os.mkdir
# with open('datasets/data_without_rebar.json', 'w') as file:
#     json.dump(data, file, indent=4)

