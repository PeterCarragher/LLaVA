import json
import numpy as np
import pandas as pd
from llava.mm_utils import get_model_name_from_path
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from tqdm import tqdm
from eval_utils import *

qids = pd.read_csv('results/counterfactual_qa_check.csv', header=None)[0].tolist()

# single image = perturbation, 2 image = conflicting
eval_data = json.load(open("WebQA_train_val_color_gpt_matched.json", "r"))
# eval_data = {k: v for k, v in eval_data.items() if k in qids}

model_paths = [
    "liuhaotian/llava-v1.6-vicuna-7b", 
    "liuhaotian/llava-v1.6-vicuna-13b", 
    "liuhaotian/llava-v1.6-mistral-7b", 
    "liuhaotian/llava-v1.6-34b",
    # "liuhaotian/llava-v1.5-7b", 
    # "liuhaotian/llava-v1.5-13b"
]
blank_image_file ='/home/pcarragh/dev/webqa/LLaVA/playground/counterfactual_exp/BLANK.jpg'
perturbation_path = "/home/pcarragh/dev/webqa/segment/Inpaint-Anything/results/webqa"

results = {}
prompt_addition = "Answer the following question based only on the provided images.\n"

for model_path in model_paths:
    print(f"Running evaluation for model: {model_path}")
    
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name
    )

    llava_results_baseline_original_label = {}
    llava_results_baseline_perturbed_label = {}
    llava_results_blank_original_label = {}
    llava_results_blank_perturbed_label = {}
    llava_results_perturbed_original_label = {}
    llava_results_perturbed_perturbed_label = {}

    for k in tqdm(list(eval_data.keys())[:10]):
        example = eval_data[k]
        question = prompt_addition + get_prompt(example)
        original_image_files = ','.join([str(img_data['image_id']) for img_data in example['img_posFacts']])
        blank_image_files = ','.join([blank_image_file for _ in example['img_posFacts']])
        try:
            baseline_answer = llava_eval_on_webqa_sample(question, original_image_files, model_path, model_name, model, image_processor, tokenizer)
            blank_answer = llava_eval_on_webqa_sample(question, blank_image_files, model_path, model_name, model, image_processor, tokenizer)
        except Exception as e:
            print(f"Error: {e}")
            continue
        
        llava_results_baseline_original_label[k] = webqa_accuracy(baseline_answer, example['A'], example['Qcate'].lower())
        llava_results_blank_original_label[k] = webqa_accuracy(blank_answer, example['A'], example['Qcate'].lower())
        llava_results_perturbed_original_label[k] = webqa_accuracy(perturbed_answer, example['A'], example['Qcate'].lower())
        llava_results_baseline_original_label[k] = webqa_accuracy(baseline_answer, [label], example['Qcate'].lower())

        for idx, label in eval_data[k]['A_perturbed'].items():
            generated_image_files = ""
            for img in example['img_posFacts']:
                generated_file = f"{perturbation_path}/{str(img['image_id'])}_{k}_{idx}.jpeg"
                if os.path.exists(generated_file):
                    generated_image_files += "," + generated_file
                else:
                    generated_image_files += "," + str(img['image_id'])
            try:
                perturbed_answer = llava_eval_on_webqa_sample(question, generated_image_files[1:], model_path, model_name, model, image_processor, tokenizer)
            except Exception as e:
                print(f"Error: {e}")
                continue
            
            llava_results_blank_original_label[k][idx] = webqa_accuracy(blank_answer, [label], example['Qcate'].lower())
            llava_results_perturbed_original_label[k][idx] = webqa_accuracy(perturbed_answer, [label], example['Qcate'].lower())
               
    results[model_path] = {
        "baseline_original_label": accuracy_agg_results(llava_results_baseline_original_label),
        "baseline_perturbed_label": accuracy_agg_results(llava_results_baseline_perturbed_label),
        "blank_original_label": accuracy_agg_results(llava_results_blank_original_label),
        "blank_perturbed_label": accuracy_agg_results(llava_results_blank_perturbed_label),
        "perturbed_original_label": accuracy_agg_generated_results(llava_results_perturbed_original_label),
        "perturbed_perturbed_label": accuracy_agg_generated_results(llava_results_perturbed_perturbed_label)
    }
    print(results[model_path])


results_df = pd.DataFrame(results).T
exp_name = __file__.split('/')[-1].split('.')[0]
results_df.to_csv(f"results/{exp_name}.csv")