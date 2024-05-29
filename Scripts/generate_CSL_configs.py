"""
Automatically generate the config file for all CSL experiments in Config/CSL.
Generates a script to run all CSL experiments in Scripts/run_csl_exp.sh
"""

import os
from Misc.config import config

config_template = \
"""
epochs: [500]
model: ["pathGNN-S"]
batch_size: [8]
emb_dim: [16]
drop_out: [0.0]
lr:  [1e-4]
tracking: [1]
lr_scheduler: ["None"]
lstm_depth: [1]
share_lstm: [0]
pooling: ["mean"]
"""

script_template = "python Exp/run_experiment.py -dataset CSL --folds 5 --repeats 1"
storage_dir = os.path.join(config.CONFIG_PATH, "CSL")
script_dir = config.SCRIPT_PATH

def main():
    script = ""
    
    if not os.path.isdir(storage_dir):
        os.mkdir(storage_dir)
    
    
    for path_length in [2,3,4,5,6]:
        for num_layers in [2,1]:
            for mark_neighbors in [2,1,0]:
                config = config_template
                config += f"num_layers: [{num_layers}]\n"  
                config += f"path_length: [{path_length}]\n"
                config += f"mark_neighbors: [{mark_neighbors}]\n"       
                
                config_name = f"csl_nl{num_layers}_pl{path_length}_mn{mark_neighbors}.yaml"
                config_path = os.path.join(storage_dir, config_name)
                
                with open(config_path, 'w') as file:
                    file.write(config)
                
                script += script_template + f" -grid {config_path};\n"
              
    script_path =  os.path.join(script_dir, "run_csl_exp.sh")
    with open(script_path, 'w') as file:
        file.write(script)
        
    print(f"Script can be found at {script_path}")
    
if __name__ == "__main__":
    main()