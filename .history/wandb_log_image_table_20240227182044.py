from PIL import Image
import wandb 


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
    

artifact = wandb.Artifact(name="chatgpt4_labeling", type="dataset")
table = wandb.Table(columns = ["image", "text"])

text = "*" * 20


data_dir = './Dataset/'

for image_local in tqdm(os.listdir(data_dir)):
    image_array = wandb.Image(Image.open(data_dir + image_local))
    table.add_data(
        image_array,
        text,
    )
    
artifact.add(table, "text lable extracted by chatgpt4")
run.log_artifact(artifact)


run.finish()