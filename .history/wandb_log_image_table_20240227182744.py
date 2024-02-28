from PIL import Image
import wandb 
from time import time 

import argparse


parser = argparse.ArgumentParser(description='aipi549')
parser.add_argument('--top_k', type=int, default=3)

args = parser.parse_args()
    
    
run = wandb.init(project = "aipi549", name = "wandb_log_image_table")



artifact = wandb.Artifact(name="image_table", type="dataset", metadata = )


table = wandb.Table(columns = ["image", "text"])

text = "*" * 20

start_time = time.time()


data_dir = './Dataset/1.jpg'
for i in range(n):
    image_array = wandb.Image(Image.open(data_dir))
    table.add_data(
        image_array,
        text,
    )
artifact.add(table, "image lable")
run.log_artifact(artifact)


end_time = time.time()

run.log({"logging time ": end_time - start_time})

run.finish()