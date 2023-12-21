import os 
from sklearn.model_selection import train_test_split

dataset = 'datasets\data_without_rebar.txt'

with open (dataset, 'r') as file:
    annotations = file.readlines()

annotations_dic = {}

for annotation in annotations:
    class_id = annotation.split()[0]
    if (class_id not in annotations_dic):
        annotations_dic[class_id] = []
    annotations_dic[class_id].append(annotation)


train_file = open('train.txt', 'w')
valid_file = open ('valid.txt', 'w')
test_file = open ('test.txt', 'w')

for class_id, items in annotations_dic.items():
    if (len(items) < 3 ):
        train_file.writelines(items)
        valid_file.writelines(items)
        test_file.writelines(items)
    else:
        train, test = train_test_split(items, test_size= 0.3)
        valid, test = train_test_split(test, test_size=0.5)
        
        train_file.writelines(train)
        valid_file.writelines(valid)
        test_file.writelines(test)

train_file.close() 
valid_file.close()
test_file.close()