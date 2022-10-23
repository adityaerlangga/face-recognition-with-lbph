
import os
 
folder_path = r"C:\Users\hmant\Desktop\Dataset\race\Asian"
file_number = 1

for file in os.listdir(folder_path):
    os.rename(os.path.join(folder_path, file), os.path.join(folder_path, "Asian.1." + str(file_number) + ".png"))
    file_number += 1