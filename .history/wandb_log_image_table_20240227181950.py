from PIL import Image
import wandb 
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