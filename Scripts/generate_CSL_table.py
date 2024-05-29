"""
Extract all CSL results into a simple latex table.
"""

import glob 
import os
import yaml

import pandas as pd

from Misc.config import config

max_nr_layers = 2
max_path_length = 6

def main():    
    spd_ls, marked_ls, unmarked_ls = [["" for _ in range(max_path_length)] for _ in range(max_nr_layers)], [["" for _ in range(max_path_length)] for _ in range(max_nr_layers)], [["" for _ in range(max_path_length)] for _ in range(max_nr_layers)]
    
    for results_dir in glob.glob(os.path.join(config.RESULTS_PATH, "CSL_*")):
        dir_name = os.path.split(results_dir)[-1]
        name_split = dir_name.split('_')

        path_length = int(name_split[3][2:])
        mark_neighbors = int(name_split[4][2])
        num_layers = int(name_split[2][2:])
        
        results_file_path = os.path.join(results_dir, "final.json")
        
        if not os.path.exists(results_file_path):
            continue
        
        with open(results_file_path) as file:
            results_dict = yaml.safe_load(file)
            
        if mark_neighbors == 1:
            marked_ls[num_layers-1][path_length-1] = "$" + str(round(results_dict["test-avg"]*100)) + r" \pm " + str(round(results_dict["test-std"]*100)) + "$"
        elif mark_neighbors == 2:
            spd_ls[num_layers-1][path_length-1] = "$" + str(round(results_dict["test-avg"]*100)) + r" \pm " + str(round(results_dict["test-std"]*100)) + "$"
        else:
            unmarked_ls[num_layers-1][path_length-1] = "$" + str(round(results_dict["test-avg"]*100)) + r" \pm " + str(round(results_dict["test-std"]*100)) + "$"
            
    print(marked_ls)
    print(unmarked_ls)
    
    df = pd.DataFrame({
        "Path Length": range(1, max_path_length+1),
        "1 L": unmarked_ls[0],
        "1 L+M": marked_ls[0],
        "1 L+M+S": spd_ls[0],
        "2 L": unmarked_ls[1],
        "2 L+M": marked_ls[1], 
        "2 L+M+S": spd_ls[1], 
    })
    print(df)
    print("\n\n\n")
    print(df.style.to_latex())

if __name__ == "__main__":
    main()