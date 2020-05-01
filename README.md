# REMI

0. Install miniconda if not already installed: https://docs.conda.io/en/latest/miniconda.html
1. Create conda environment: `conda env create -f environment.yml`
2. Activate conda environment: `conda activate remi` 
3. Download REMI-tempo-checkpoint pre-trained model and move it to this folder (https://drive.google.com/open?id=1gxuTSkF51NP04JZgTE46Pg4KQsbHQKGo).
4. Unzip pre-trained model: `unzip REMI-tempo-checkpoint.zip`
5. To reproduce Transformer results in report: `nice python3 final_proj_main.py &> out.txt &`

* The code that I wrote for this project is located in final_proj_main.py and final_proj_model.py. Parts of this code are based on code in model.py, utils.py, and main.py.
