import shutil
import os


root_dir = "tau-ethiopic-digit-recognition/"

if not os.path.isdir(root_dir + "val"): 
    os.makedirs(root_dir + "val", exist_ok=True)

    folders = os.listdir(root_dir+ "train/")

    for folder in folders:
        new_val_folder = root_dir + "val/" + folder  
        os.makedirs(new_val_folder, exist_ok=True)   
        images_list = os.listdir(root_dir + "train/" + folder) 
        images_list = images_list[-1000:]
       
        for img in images_list:
            src = root_dir + "train/" + folder + "/" + img 
            dest = root_dir + "val/" + folder + "/" + img  
            shutil.move(src, dest)

