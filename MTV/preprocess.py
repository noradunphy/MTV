#### 
import json
import random

vizwiz_prompt = """First carefully understand the given examples. 
Then use the given image and answer the question in the same way as the examples. 
If the question can not be answered, respond unanswerable. """
####

def open_data(dataset_name, path, dialogue_act=None):
    jsonl_format_dataset = ["vizwiz", "okvqa"]
    list_format_dataset = ["flower", "cub", "dtd", "swda"]  # saved as full JSON arrays

    # DailyDialog is stored as JSON Lines (one object per line)
    if dataset_name == "dailydialog":
        with open(path, 'r', encoding='utf-8') as json_file:
            dataset = [json.loads(l) for l in json_file]
    else:
        with open(path, 'r') as json_file:
            if dataset_name in jsonl_format_dataset:
                dataset = list(json_file)
            elif dataset_name in list_format_dataset:
                dataset = json.load(json_file)
    
    # If dialogue_act is specified and dataset is swda or dailydialog, filter for that act
    if dialogue_act is not None and dataset_name in ("swda", "dailydialog"):
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
    if cur_dataset in ("swda", "swda_nextutt", "dailydialog"):
        # Re-use the same multiple-choice formatting for DailyDialog
        return format_swda_multiple_choice


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
    #check model outputs multiple tokens or not
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
    if num_shot > 0:
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

def format_swda_multiple_choice(filtered_dataset, full_dataset, cur_item=None, num_shot=0, model_helper=None, split="train", mcq_options=None):
    """
    Format function for SWDA multiple choice with improved prompt engineering for Llama-3.1-8B-Instruct.
    
    Args:
        filtered_dataset: Dataset filtered to target dialogue act (for correct answers)
        full_dataset: Complete dataset with all acts (for distractors)
        cur_item: Current example to format
        num_shot: Number of few-shot examples
        model_helper: Model helper instance
        split: Dataset split
        mcq_options: Pre-defined multiple choice options (optional)
    """
    if cur_item is None:
        data = random.sample(filtered_dataset, 1)[0]
    else:
        data = cur_item

    # Group ALL examples by dialogue act once (using full dataset)
    act_to_examples = {}
    for ex in full_dataset:  # Use full_dataset here
        act = ex.get('dialog_act', 'o')
        if act not in act_to_examples:
            act_to_examples[act] = []
        act_to_examples[act].append(ex)

    prompt = """You are given a short conversation and four candidate responses. Pick the SINGLE best response. Output ONLY the option number (1-4). Do not output anything else.\n\n"""

    if num_shot > 0:
        # Get examples with the same act as current example
        target_act = data.get('dialog_act', 'o')
        
        # For each few-shot example
        for i in range(num_shot):
            # Get examples from target act, excluding current example and identical responses
            same_act_examples = [e for e in filtered_dataset 
                               if e != data 
                               and e.get('response', '').strip() != data.get('response', '').strip()]
            
            if same_act_examples:
                # Sample one correct answer from target act
                correct_example = random.choice(same_act_examples)
                correct_utterance = correct_example.get('response', '')
            else:
                correct_utterance = "I understand."

            # Get other acts to sample distractors from
            other_acts = [act for act in act_to_examples.keys() if act != target_act]
            if other_acts:
                # Sample 3 different acts and get one utterance from each
                sampled_acts = random.sample(other_acts, min(3, len(other_acts)))
                distractor_utterances = []
                for act in sampled_acts:
                    if act_to_examples[act]:
                        distractor = random.choice(act_to_examples[act])
                        distractor_utterances.append(distractor.get('response', ''))
            else:
                distractor_utterances = ["I see.", "That's interesting.", "Go on."]

            # Create and shuffle options
            options = [correct_utterance] + distractor_utterances[:3]
            random.shuffle(options)

            # Ensure exactly 4 options
            while len(options) < 4:
                options.append("I see.")
            options = options[:4]

            # Find correct number after shuffling
            correct_number = options.index(correct_utterance) + 1

            # Format the example with clear structure
            context = correct_example.get("text", "").strip()
            if context:
                context_lines = context.split('\n')
                formatted_context = '\n'.join([line.strip() for line in context_lines if line.strip()])
                prompt += f"EXAMPLE:\n{formatted_context}\n"
                prompt += f"{correct_example['caller']}:\n"
            else:
                prompt += "EXAMPLE:\n[Brief conversation]\n"

            prompt += f"QUESTION: Given the conversation context above, which response is most appropriate?\n"
            prompt += f"OPTIONS: 1) {options[0]} 2) {options[1]} 3) {options[2]} 4) {options[3]}\n"
            prompt += f"ANSWER: {correct_number}\n"
            prompt += "______\n\n"

    # Add the current query with improved formatting
    context = data.get("text", "").strip()
    if context:
        context_lines = context.split('\n')
        formatted_context = '\n'.join([line.strip() for line in context_lines if line.strip()])
        prompt += f"TEST EXAMPLE:\n{formatted_context}\n"
        prompt += f"{data['caller']}:\n"
    else:
        prompt += "TEST EXAMPLE:\n[Brief conversation]\n"

    # Handle current example options
    target_act = data.get('dialog_act', 'o')
    
    if filtered_dataset is None:
        filtered_dataset = [ex for ex in full_dataset if ex.get('dialog_act') == target_act]

    # Get pool of target act examples (excluding current)
    same_act_examples = [e for e in filtered_dataset
                        if e != data 
                        and e.get('response', '').strip() != data.get('response', '').strip()]
    
    if same_act_examples:
        correct_example = random.choice(same_act_examples)
        correct_utterance = correct_example.get('response', '')
    else:
        correct_utterance = "I understand."

    # Get distractors from other acts
    other_acts = [act for act in act_to_examples.keys() if act != target_act]
    if other_acts:
        sampled_acts = random.sample(other_acts, min(3, len(other_acts)))
        distractor_utterances = []
        for act in sampled_acts:
            if act_to_examples[act]:
                distractor = random.choice(act_to_examples[act])
                distractor_utterances.append(distractor.get('response', ''))
    else:
        distractor_utterances = ["I see.", "That's interesting.", "Go on."]

    # Create and shuffle options
    options = [correct_utterance] + distractor_utterances[:3]
    random.shuffle(options)

    # Ensure exactly 4 options
    while len(options) < 4:
        options.append("I see.")
    options = options[:4]

    # Find correct number after shuffling
    target_number = options.index(correct_utterance) + 1

    # Format final prompt with clear structure
    prompt += f"QUESTION: Given the conversation context above, which response is most appropriate?\n"
    prompt += f"OPTIONS: 1) {options[0]} 2) {options[1]} 3) {options[2]} 4) {options[3]}\n"
    prompt += "ANSWER:"

    return prompt, [], str(target_number), data.get("utterance_id", -1)