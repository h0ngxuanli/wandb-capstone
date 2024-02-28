from PIL import Image
import wandb 
import time
import os
import argparse
import psutil

def get_image_memory_size(image_path):
    size_bytes = os.path.getsize(image_path)
    # Convert bytes to gigabytes
    size_gb = size_bytes / (1024 * 1024 * 1024)
    return size_gb

def get_hardware_information():
    # CPU information
    cpu_info = {
        "CPU Count": psutil.cpu_count(logical=False),
        "Logical CPU Count": psutil.cpu_count(logical=True),
        "CPU Frequency": psutil.cpu_freq(),
        "CPU Usage": psutil.cpu_percent(interval=1, percpu=True)
    }

    # Memory information
    mem_info = {
        "Total Memory": psutil.virtual_memory().total,
        "Available Memory": psutil.virtual_memory().available,
        "Used Memory": psutil.virtual_memory().used,
        "Memory Usage Percentage": psutil.virtual_memory().percent
    }

    # Disk information
    disk_info = {
        "Disk Usage": psutil.disk_usage('/'),
        "Disk Partitions": psutil.disk_partitions()
    }

    # Network information
    net_info = {
        "Network Interfaces": psutil.net_if_addrs()
    }

    return {
        "CPU": cpu_info,
        "Memory": mem_info,
        "Disk": disk_info,
        "Network": net_info
    }

parser = argparse.ArgumentParser(description='aipi549')
parser.add_argument('--n_img', type=int, default=10)
parser.add_argument('--core', type=str, default="new")
args = parser.parse_args()
hardware_info = get_hardware_information()    
    
run = wandb.init(project = "aipi549", name = "wandb_log_image_table"+ "_" + args.core)

for i in range(1, )
artifact = wandb.Artifact(name="image_table", type="dataset", metadata = {**vars(args), **hardware_info})
table = wandb.Table(columns = ["image"])

start_time = time.time()
data_dir = './Dataset/1.jpg'
for i in range(args.n_img):
    image_array = wandb.Image(Image.open(data_dir))
    table.add_data(
        image_array,
    )
    
artifact.add(table, "image lable")
run.log_artifact(artifact)
end_time = time.time()
run.log({"logging time ": end_time - start_time})
run.log({"table size ": get_image_memory_size(data_dir)*args.n_img})


run.finish()