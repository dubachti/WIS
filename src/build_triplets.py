import os

# get file names for all data instances
file_names = []
for root, _, files in os.walk('small_data_transformed'):
    for file in files:
        if file.startswith('.'): continue # ignore .DS_store files
        with open(os.path.join(root, file), "r") as auto:
            file_names.append(auto.name)

dic = {}
for file_name in file_names:
    file = file_name.split('/')[-1]
    speaker = file.split('-')[0]
    dic.setdefault(speaker, []).append(file)


print(dic.keys())


