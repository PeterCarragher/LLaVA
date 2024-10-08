from eval_1022 import *
import json
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
import json
# from llava.model.builder import load_pretrained_model


train_val_dataset = json.load(open("/home/pcarragh/dev/webqa/MultiModalQA/data/WebQA_train_val_obj_v2.json.pert.lama.v3", "r"))
eval_data = {k:v for k, v in train_val_dataset.items() if v['split'] == 'val' and v['Qcate'].lower() in ['color']}#, 'shape', 'yesno', 'number']}
len(eval_data)

model_path = "liuhaotian/llava-v1.5-7b"
qa_results = {}
for k in eval_data.keys():
    question = eval_data[k]['Q']
    image_files = ','.join([str(img_data['image_id']) for img_data in eval_data[k]['img_posFacts']])
    print(image_files)

    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": get_model_name_from_path(model_path),
        "query": question,
        "conv_mode": None,
        "image_file": image_files,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
        # "webqa": True, TODO
    })()

    answer = eval_model(args)

    # except Exception as e: 
    #     # print(k)
    #     answer = ['']
    label = eval_data[k]['A']
    eval_data[k]['A_llava'] = answer
    print(f"Q: {question}, A: {answer}, L: {label}")

    # Qcate = eval_data[k]['Qcate'].lower()
    # if Qcate == 'color': 
    #     F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(answer, label[0], "", color_set)
    # elif Qcate == 'shape': 
    #     F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(answer, label[0], "", shape_set)
    # elif Qcate == 'yesno': 
    #     F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(answer, label[0], "", yesno_set)
    # elif Qcate == 'number': 
    #     F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics(answer, label[0], "", {"NUMBER"})
    # else:
    #     continue
    
    # if not Qcate in qa_results:
    #     qa_results[Qcate] = []
    # qa_results[Qcate].append(PR_avg)
