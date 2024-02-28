from PIL import Image
import wandb 
from time import time 

import argparse


parser = argparse.ArgumentParser(description='aipi590')
parser.add_argument('--model_path', type=str, default= "/home/featurize/work/Transformer-Patcher-main/model_cache/llama2-7b")
parser.add_argument('--max_seq_len', type=int, default=1024)
parser.add_argument('--chunk_size', type=int, default=300)
parser.add_argument('--step_size', type=int, default=200)
parser.add_argument('--top_k', type=int, default=3)
parser.add_argument('--index_name', type=str, default="aipi590")
parser.add_argument('--API_key', type=str, default=None)
args = parser.parse_args()
    
    
run = wandb.init(project = "aipi549", name = "wandb_log_image_table")



artifact = wandb.Artifact(name="image_table", type="dataset", metadata = )


table = wandb.Table(columns = ["image", "text"])

text = "*" * 20


data_dir = './Dataset/1.jpg'

for i in range(n)
    image_array = wandb.Image(Image.open(data_dir))
    table.add_data(
        image_array,
        text,
    )
artifact.add(table, "image lable")
run.log_artifact(artifact)


run.log({"iris": iris_table})

run.finish()