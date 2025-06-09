#### 
import json
import random

vizwiz_prompt = """First carefully understand the given examples. 
Then use the given image and answer the question in the same way as the examples. 
If the question can not be answered, respond unanswerable. """
####

def open_data(dataset_name, path, dialogue_act=None):
    jsonl_format_dataset = ["vizwiz", "okvqa"]
    list_format_dataset = ["flower", "cub", "dtd", "swda"]

    with open(path, 'r') as json_file:
        if dataset_name in jsonl_format_dataset:
            dataset = list(json_file)
        elif dataset_name in list_format_dataset:
            dataset = json.load(json_file)
    
    # If dialogue_act is specified and dataset is swda, filter for that act
    if dialogue_act is not None and dataset_name == "swda":
        filtered_dataset = []
        for item in dataset:
            # item is already a dict for .json files
            if item.get('dialog_act') == dialogue_act:
                filtered_dataset.append(item)
        dataset = filtered_dataset
    return dataset


### Each format function should return (full_text, image_list, answer, question_id)
def get_format_func(cur_dataset, zero_shot=False):

    if cur_dataset == "vizwiz":
        return format_vizwiz
    if cur_dataset == "okvqa":
        return format_okvqa
    if cur_dataset == "flower":
        return format_flower
    if cur_dataset == "cub":
        return format_cub
    if cur_dataset == "dtd":
        return format_dtd
    if cur_dataset == "swda" or cur_dataset == "swda_nextutt":
        return format_swda_next_utt


####All return format will be in the form (Text, list of images, Answer, Question_id)
def format_vizwiz(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    prompt = '<image>{} Answer:'

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question, answer, question_id = data['image'], data['question'], data['answer'], data['question_id']

    few_shot_prompt = ''
    if num_shot > 0:

        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(sample['question']) + f" {sample['answer']}"
            image_list.append("../" + sample["image"])


    full_text = vizwiz_prompt + few_shot_prompt + prompt.format(question)
    image_list.append("../" + image)

    return full_text, image_list, answer, question_id


def format_okvqa(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    prompt = '<image>{} Answer:'

    image_list = []

    if cur_item is None:
        data = json.loads(random.sample(all_data, 1)[0])
    else:
        data = json.loads(cur_item)

    image, question, answer, question_id = data['image'], data['question'], data['answer'], data['question_id']

    few_shot_prompt = ''
    if num_shot > 0:
        sampled_data = random.sample(all_data, num_shot)
        for sample in sampled_data:
            sample = json.loads(sample.strip())
            few_shot_prompt += prompt.format(sample['question']) + f"{sample['answer']}"
            image_list.append(sample["image"])

    
    full_text = few_shot_prompt + prompt.format(question)
    image_list.append(image)

    return full_text, image_list, answer, question_id


def format_flower(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1

#actual zero
def format_flower_for_eval(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        cur_query = f"<image>What is the type of flower in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"
    else:
        cur_query = f"<image>What is the type of flower in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

    return cur_query, [pos, neg, query], query_label, -1


def format_cub(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of bird in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of bird in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1
    

def format_dtd(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):

    if cur_item is None:
        cur_item = random.sample(all_data, 1)[0]

    pos = cur_item["pos"]
    neg = cur_item["neg"]
    pos_label = cur_item["pos_label"]
    neg_label = cur_item["neg_label"]
    query = cur_item["query"]
    rand_num = random.randint(0,1)
    if rand_num == 0:
        pos_example = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        neg_example = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        cur_query = f"<image>What is the type of texture in the image? A.{pos_label} B.{neg_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "A"

        return pos_example + neg_example + cur_query, [pos, neg, query], query_label, -1
    else:
        pos_example = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: B\n"
        neg_example = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer: A\n"
        cur_query = f"<image>What is the type of texture in the image? A.{neg_label} B.{pos_label}\nAnswer with the option's letter from the given choice directly. Answer:"
        query_label = "B"

        return neg_example + pos_example + cur_query, [neg, pos, query], query_label, -1

#NEW

def format_swda_next_utt(all_data, cur_item=None, num_shot=0, model_helper=None, split="train"):
    """Format function for next utterance generation in SWDA with N-shot prompting."""
    if cur_item is None:
        data = random.sample(all_data, 1)[0]
    else:
        data = cur_item

    # Get N random examples from the same dialog act class if num_shot > 0
    prompt = ""
    if num_shot > 0 and split == "train":
        # Get examples with the same dialog act
        same_act_examples = [ex for ex in all_data if ex.get('dialog_act') == data.get('dialog_act') and ex != data]
        if same_act_examples:
            # Sample N examples
            shot_examples = random.sample(same_act_examples, min(num_shot, len(same_act_examples)))
            
            # Format each example
            for i, ex in enumerate(shot_examples):
                prompt += f"Chat {i+1}\n"
                if ex.get("text", "").strip() != "":
                    prompt += ex["text"] + "\n"
                prompt += f"Final Response {ex['caller']}: {ex['response']}\n"
                prompt += "______\n"
            
            prompt += "\n"  # Add spacing between examples and query

    # Add the current query
    if data.get("text", "").strip() != "":
        prompt += data["text"] + "\n"
    prompt += f"Final Response {data['caller']}:"

    target = data["response"]
    utt_id = data.get("utterance_id", -1)
    
    return prompt, [], target, utt_id