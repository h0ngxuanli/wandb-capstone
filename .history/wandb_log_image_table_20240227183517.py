from PIL import Image
import wandb 
from time import time 
import os
import argparse

def get_image_memory_size(image_path):
    size_bytes = os.path.getsize(image_path)
    # Convert bytes to gigabytes
    size_gb = size_bytes / (1024 * 1024 * 1024)
    return size_gb


parser = argparse.ArgumentParser(description='aipi549')
parser.add_argument('--n_img', type=int, default=10)
args = parser.parse_args()
    
run = wandb.init(project = "aipi549", name = "wandb_log_image_table")
artifact = wandb.Artifact(name="image_table", type="dataset", metadata = vars(args))
table = wandb.Table(columns = ["image"])

# text = "*" * 20

start_time = time.time()
data_dir = './Dataset/1.jpg'
for i in range(n_img):
    image_array = wandb.Image(Image.open(data_dir))
    table.add_data(
        image_array,
        text,
    )
artifact.add(table, "image lable")
run.log_artifact(artifact)

end_time = time.time()

run.log({"logging time ": end_time - start_time})
run.log({"table size ": get_image_memory_size(data_dir)*n_img})


run.finish()