import os
from tqdm.auto import tqdm
import pandas as pd
from peft import PeftModel, PeftConfig
from utils import clean_html_string, write_html_to_file
from transformers import AutoModelForCausalLM, AutoTokenizer


# --------------------->>>> Load Model <<<<------------------------ #
# Load base model and tokenizer
model_name = "meta-llama/CodeLlama-7b-hf"  # example base model
base_model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load the LoRA configuration and model
lora_path = "./codellama-a11y"  # path to your LoRA weights
config = PeftConfig.from_pretrained(lora_path)
model = PeftModel.from_pretrained(base_model, lora_path)
model.to('cuda')
model.eval() # Set model to evaluation mode


# ------------------ >>>> Load Dataset <<<< -------------------------#

dev_dataset = pd.read_csv('./developer_prompt_dataset_v2.csv', encoding='latin-1')
dev_dataset_dict = dev_dataset.to_dict('records')

# missing_ds = [item for item in dev_dataset_dict if item['Developer'] == 'D50'] # in ['D32', 'D11', 'D26', 'D41', 'D22']]

# --------------- >>>> Generate Responses <<<< -----------------------#

for item in tqdm(dev_dataset_dict, desc="Generating Developer Responses"):
    # Tokenize input
    inputs = tokenizer(item['Prompt'], return_tensors="pt")

    # Move inputs to the same device as the model (CUDA)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}

    # Generate text
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=4096,
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    # Decode the generated text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(generated_text)

    # clean the generated dataset
    cleaned_generated_text = clean_html_string(generated_text)

    # Save Dataset
    os.makedirs(f"./html_files_old/{item['Developer']}", exist_ok=True)
    html_content =  cleaned_generated_text 
    output_path = f"./html_files_old/{item['Developer']}/index.html"
    write_html_to_file(html_content, output_path)