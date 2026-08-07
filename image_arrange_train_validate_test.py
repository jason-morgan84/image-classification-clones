import os
import math
import random
import shutil

#TODO: Include ability to switch between windows and linux file locations

def separate(file_list, list_name, output_path):
    print("Copying", list_name,"files")
    output_folder = os.path.join(output_path, list_name)
    output_csv = os.path.join(output_folder, list_name + ".csv")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(output_csv, "w") as output_file:
        output_file.writelines("genotype,date,name,clone,input_location,image_id\n")

    # check output folder exists


    # copy given file list to folder    
    
    for n, file in enumerate(file_list):
        print(n,"/",len(file_list),end='\r')
        genotype, date, name, input_location, clone, full_path = tuple(file.rstrip().split(","))
        file_location, file_name = os.path.split(full_path)
        image_id, _ = os.path.splitext(file_name)
        shutil.copyfile(full_path, os.path.join(file_location, list_name, file_name))
        # save file list to csv
        with open(output_csv, "a") as output_file:
            output_file.writelines(",".join([genotype,date,name,clone,input_location,image_id + "\n"]))

    
def randomise_files(file_list, start_proportion, end_proportion):
    output_list = []
    list_length = len(image_files)

    for i in range(math.ceil(start_proportion * list_length), math.floor(end_proportion * list_length),1):
        output_list.append(file_list[i])

    return output_list

input_file = "output.csv"
input_path = "D:\\Lab\\Classification\\All images\\Processed 260805\\"
output_path = "D:\\Lab\\Classification\\All images\\Processed 260805\\"
image_files = []

with open(os.path.join(input_path,input_file)) as input_file:
    _ = input_file.readline()
    for line in input_file:
        image_files.append(line)

train_proportion = 0.7
validate_proportion = 0.2
test_proportion = 0.1

random.shuffle(image_files)

train_files = randomise_files(image_files, 0, train_proportion )
validate_files = randomise_files(image_files, train_proportion, train_proportion + validate_proportion)
test_files = randomise_files(image_files, train_proportion + validate_proportion, train_proportion + validate_proportion + test_proportion)

separate(train_files,"training", output_path)
separate(validate_files, "validation", output_path)
separate(test_files, "testing", output_path)



