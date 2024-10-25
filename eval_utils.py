from eval_1022 import *
from llava.eval.run_llava import eval_model, eval

def llava_eval_on_webqa_sample(question, image_files, model_path, model_name, model, image_processor, tokenizer):
    args = type('Args', (), {
        "model_path": model_path,
        "model_base": None,
        "model_name": model_name,
        "query": question,
        "conv_mode": None,
        "image_file": image_files,
        "sep": ",",
        "temperature": 0,
        "top_p": None,
        "num_beams": 1,
        "max_new_tokens": 512,
    })()

    return eval(tokenizer, model, image_processor,  args)

def webqa_accuracy(answer, label, Qcate):
    if Qcate == 'color':
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics([answer], label[0], "", color_set)
    elif Qcate == 'shape': 
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics([answer], label[0], "", shape_set)
    elif Qcate == 'yesno': 
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics([answer], label[0], "", yesno_set)
    elif Qcate == 'number': 
        F1_avg, F1_max, EM, RE_avg, PR_avg = compute_vqa_metrics([answer], label[0], "", {"NUMBER"})
    else:
        return None
    return (F1_avg, F1_max, EM, RE_avg, PR_avg)

def ans_contains_any_label(ans, labels = ['yes', 'no']):
        return any([label in ans.lower() for label in labels])
    
def ans_contains_correct_label(ans, correct_ans, qcate):
    _,_,_,_,pr = webqa_accuracy(ans, correct_ans, qcate)
    return pr

def accuracy_agg_results(qa_results, eval_data):
    single_image_keys = [k for k in qa_results.keys() if len(eval_data[k]['img_posFacts']) == 1]
    two_image_keys = [k for k in qa_results.keys() if len(eval_data[k]['img_posFacts']) == 2]

    single_acc = np.mean([PR_avg for key, (F1_avg, F1_max, EM, RE_avg, PR_avg) in qa_results.items() if key in single_image_keys])
    two_image_acc = np.mean([PR_avg for key, (F1_avg, F1_max, EM, RE_avg, PR_avg) in qa_results.items() if key in two_image_keys])
    avr_acc = np.mean([PR_avg for key, (F1_avg, F1_max, EM, RE_avg, PR_avg) in qa_results.items()])
    return (single_acc, two_image_acc, avr_acc)

def accuracy_agg_generated_results(qa_results, eval_data):
    single_image_keys = [k for k in qa_results.keys() if len(eval_data[k]['img_posFacts']) == 1]
    two_image_keys = [k for k in qa_results.keys() if len(eval_data[k]['img_posFacts']) == 2]

    single_acc = np.mean([PR_avg for key, dict in qa_results.items() if key in single_image_keys for idx, (_,_,_,_,PR_avg) in dict.items()])
    two_image_acc = np.mean([PR_avg for key, dict in qa_results.items() if key in two_image_keys for idx, (_,_,_,_,PR_avg) in dict.items()])
    avr_acc = np.mean([PR_avg for key, dict in qa_results.items() for idx, (_,_,_,_,PR_avg) in dict.items()])
    
    return (single_acc, two_image_acc, avr_acc)

def get_prompt(data, reverse_images = False):
    imgs = data['img_posFacts']
    if len(imgs) == 1:
        return f"<image-placeholder> \n Caption: {imgs[0]['title']} \n Question: {data['Q']}"
    assert(len(imgs) == 2)
    if reverse_images:
        return f"<image-placeholder> \n Caption: {imgs[1]['title']} \n <image-placeholder> \n Caption: {imgs[0]['title']} \n Question: {data['Q']}"
    return f"<image-placeholder> \n Caption: {imgs[0]['title']} \n <image-placeholder> \n Caption: {imgs[1]['title']} \n Question: {data['Q']}"
