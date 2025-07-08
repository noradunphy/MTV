from baukit import TraceDict, get_module
from models import *
from preprocess import *
import sys
import torch
import torch.nn.functional as F
import numpy as np
import json
import random
from tqdm import tqdm
import pdb
import re  # Added for regex operations in extract_first_turn

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForVision2Seq, logging
import sys
from torchvision.ops.boxes import box_area

logging.set_verbosity_warning()
import warnings

torch.autograd.set_detect_anomaly(True)

#sys.path.append('../eval_mm')
sys.path.append('/home/noradunphy/projects/mtv/eval_mm')
from vqa import VQA
from vqa_eval import VQAEval


def load_model(model_name, cur_dataset, zero_shot=False):

    """
    A function that loads the model and a corresponding model_helper. Refer to model.py for more detail.

    Parameters:
    model_name: The name of the model you are attempting to load
    cur_dataset: The name of dataset you are attempting to load
    zero_shot: Whether the model is being loaded for zero-shot evaluation

    Returns: 
    model_helper: A helper class that contains the model as well as other functionality.
    """

    if model_name == "swda":
        # Use the same logic as "text"
        model_name = "text"

    if model_name == "Qwen-VL":
        
        model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL", device_map="auto", trust_remote_code=True, fp16=True).eval()

        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL", trust_remote_code=True)
        tokenizer.padding_side = 'left'
        tokenizer.pad_token_id = tokenizer.eod_id

        model_helper = QwenHelper(model, tokenizer, cur_dataset)

    if model_name == "ViLA":
        from peft import PeftModel, PeftConfig

        from llava.mm_utils import get_model_name_from_path
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        disable_torch_init()
        model_name = get_model_name_from_path("Efficient-Large-Model/Llama-3-VILA1.5-8b")
        tokenizer, model, image_processor, context_len = load_pretrained_model("Efficient-Large-Model/Llama-3-VILA1.5-8b", model_name, None)
        model_helper = ViLAHelper(model, tokenizer, image_processor, cur_dataset)

    if model_name == "idefics2":
        
        processor = AutoProcessor.from_pretrained("HuggingFaceM4/idefics2-8b")
        processor.image_processor.do_image_splitting = False
        model = AutoModelForVision2Seq.from_pretrained(
            "HuggingFaceM4/idefics2-8b",
            torch_dtype=torch.float16,
            _attn_implementation="flash_attention_2",
            device_map="auto"
        )

        model_helper = Idefics2Helper(model, processor, cur_dataset)

    if model_name == "text":
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            torch_dtype=torch.bfloat16,
            device_map="auto"
        ).eval()
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.1-8B-Instruct",
            padding_side="right",
            use_fast=False
        )
        # Set pad token to eos token for Llama models (this is the standard practice)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model_helper = TextModelHelper(model, tokenizer, cur_dataset, zero_shot)
        print("this is in mtv_utils.py")
    
    if model_name == "llava":
        from llava.model.builder import load_pretrained_model
        from llava.mm_utils import get_model_name_from_path
        from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
        from llava.constants import IMAGE_TOKEN_INDEX
        from llava.mm_utils import process_images

        model_path = "liuhaotian/llava-v1.5-7b"
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, processor, _ = load_pretrained_model(model_path, None, model_name, load_8bit=False, load_4bit=False)
        model = model.to("cuda")
        return llavaOAHelper(model, tokenizer, processor, cur_dataset)

    return model_helper


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/308e9d174cf0a1cf910b891d340f0dfd14168668/src/utils/extract_utils.py#L15
def gather_last_attn_activations(inputs, model_helper):

    """
    A function that performs a forward pass and extract the activation at certain location of the layer.

    Parameters:
    inputs: input to the model. Created with model_helper
    model_helper

    Returns: 
    td: The attention activations.
    result: The output logits from forward method.
    """

    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], retain_input=True, retain_output=True) as td:                
        result = model_helper.forward(inputs)
    return td, result


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/308e9d174cf0a1cf910b891d340f0dfd14168668/src/utils/extract_utils.py#L65
def split_activations_by_head(activations, model_config):

    """
    The model concatenate the output of multi-headed attention to a single vector. This function splits this vector back to different heads.
    From

    Parameters:
    activations: From gather_last_attn_activations
    model_config: Refer to model.py

    Returns: 
    the activation partitioned by attention heads
    """

    new_shape = activations.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
    activations = activations.view(*new_shape)  # (batch_size, n_tokens, n_heads, head_hidden_dim)
    return activations.to("cuda")


def extract_first_turn(text):
    """Extract the model-generated first turn/answer.

    Behaviour:
    1. Finds the first non-empty line of *text* that is not clearly part of the
       dialogue history (lines that contain a speaker prefix like "A:" or
       "Speaker:").
    2. If that line begins with one or more digits (e.g. "1", "2) …", "3.",
       etc.) we return *only* the leading digit sequence.  This is crucial for
       SWDA multiple-choice where the answer is exactly the number.
    3. Otherwise we return the cleaned line itself.

    The function always returns a **string** so downstream logic that slices
    (e.g. `s[0]`) continues to work unchanged.
    """

    if text is None:
        return ""

    # Split text into non-empty stripped lines
    lines = [ln.strip() for ln in str(text).split("\n") if ln.strip()]
    if not lines:
        return ""

    # Helper to post-process a candidate line
    def _clean_line(ln: str) -> str:
        # Remove speaker labels like "A:" or "Speaker:" if present
        if ':' in ln and not ln.split(':', 1)[0].strip().isdigit():
            ln = ln.split(':', 1)[1].strip()
        return ln

    # Iterate to find the first contentful line
    for ln in lines:
        candidate = _clean_line(ln)
        if candidate:
            # If the candidate starts with digits, return just those digits
            m = re.match(r"^(\d+)", candidate)
            return m.group(1) if m else candidate

    # Fallback to very first line if none matched above
    first = _clean_line(lines[0])
    m = re.match(r"^(\d+)", first)
    return m.group(1) if m else first


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/308e9d174cf0a1cf910b891d340f0dfd14168668/src/utils/extract_utils.py#L46
def get_last_mean_head_activations(dataset, model_helper, N_TRIALS = 50, shot=4, no_mean=False, save_path=None, load_path=None, full_dataset=None):

    """
    This function extracts the activation of the last input token.

    Parameters:
    dataset: a iterable item suitable for model_helper.format_func. Essentially a dataloader.
    model_helper:
    N_TRIALS: How many example to average the activation over
    shot: Number of shots per example
    no_mean: Whether you want to take the mean of the examples or save it for other preprocess
    save_path: Optional path to save activations
    load_path: Optional path to load pre-computed activations

    Returns: 
    mean_activations: It has the dimension of (layer, head, Token_len, residual_dim) or (N_TRIALS, layer, head, Token_len, residual_dim). Token_len is set to 1 in this case.
    """
    if load_path is not None:
        print(f"Loading pre-computed activations from {load_path}")
        return torch.load(load_path)

    activation_storage = None

    for n in tqdm(range(N_TRIALS)):
        # Clear CUDA cache before each trial
        torch.cuda.empty_cache()
        
        if model_helper.cur_dataset == "swda":
            text, image_list, _, _ = model_helper.format_func(dataset, full_dataset, None, num_shot=shot, model_helper=model_helper)
        else:
            text, image_list, _, _ = model_helper.format_func(dataset, full_dataset, None, num_shot=shot, model_helper=model_helper)
        if n % 10 == 0:
            print(text)
        # print(text)
        # pdb.set_trace()
        #is this actually getting the last token?
        inputs = model_helper.insert_image(text, image_list)
        activations_td, result = gather_last_attn_activations(inputs, model_helper)

        # Process activations in smaller chunks if needed
        stack_initial = torch.vstack([split_activations_by_head(activations_td[layer].input, model_helper.model_config) for layer in model_helper.model_config['attn_hook_names']]).permute(0,2,1,3) 
        cur_activation = stack_initial[:, :, -1, :].unsqueeze(dim=2).unsqueeze(dim=0)
        
        # Move to CPU if memory is tight
        cur_activation = cur_activation.cpu()
        
        if activation_storage is None:
            activation_storage = cur_activation
        else:
            activation_storage = torch.vstack((activation_storage, cur_activation))
        
        # Clear intermediate tensors
        del stack_initial
        del activations_td
        del result
        torch.cuda.empty_cache()

    if no_mean:
        if save_path is not None:
            torch.save(activation_storage, save_path)
        return activation_storage
    
    mean_activations = activation_storage.mean(dim=0)
    
    if save_path is not None:
        torch.save(mean_activations, save_path)
    
    return mean_activations


def reinforce(mean_activations, model_helper, reinforce_data, eval_data, full_dataset):

    """
    This function performs Reinforce to select the attentions that encodes ICL examples.

    Parameters:
    mean_activations: From get_last_mean_head_activations
    model_helper:
    reinforce_data: Dataset used during reinforce optimization
    eval_data: Dataset used for Validation

    Returns: 
    bernoullis: A tensor of bernoullis variable. One variable for each attention heads. Each denote the probability of selecting this attention head.
    """

    num_layer = model_helper.model_config["n_layers"]
    num_heads = model_helper.model_config["n_heads"]
    lr = 0.1
    eps = 1e-3
    epoch = 600

    #(num_layer, num_head)
    bernoullis = [torch.neg(torch.ones(num_heads)).requires_grad_() for _ in range(num_layer)]
    optim = torch.optim.Adam(bernoullis, lr=lr)
    with torch.set_grad_enabled(True):

        for epoch in tqdm(range(epoch)):
            
            loss_list = []
            saved_log_probs = []

            text, image_list, target_out, _ = model_helper.format_func(reinforce_data, full_dataset, None, num_shot=0, model_helper=model_helper)
            new_input = model_helper.insert_image(text, image_list)

            if type(target_out)==list:
                target_out = target_out[0]

            if model_helper.space:
                target_out = " " + target_out

            # Extract first turn from target output
            target_out = extract_first_turn(target_out)

            target_token = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0][model_helper.nonspecial_idx].unsqueeze(dim=0).to("cuda")
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)

            ###Sampling the distribution many times to reduce variance.
            for _ in range(32):

                ##Current sample
                sampled = prob_dist.sample()
                saved_log_probs.append(prob_dist.log_prob(sampled))

                with torch.no_grad():
                    out_logit = reinforce_activation_replacement(new_input, mean_activations, model_helper, sampled, last_token_only=True)
                    task_loss = torch.nn.functional.cross_entropy(out_logit, target_token)
                    loss_list.append(task_loss)

            policy_loss = []
            loss_list = torch.tensor(loss_list)
            loss_list = (loss_list - loss_list.mean())/(loss_list.std() + eps)

            for log_prob, R in zip(saved_log_probs, loss_list):
                policy_loss.append(log_prob * R)

            optim.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optim.step()
            torch.cuda.empty_cache()
            if epoch % 50 == 0:
                validate_reinforce(model_helper, bernoullis, eps, mean_activations, eval_data, epoch, full_dataset=full_dataset)
    return bernoullis


def validate_reinforce(model_helper, bernoullis, eps, mean_activations, eval_data, epoch, sampled=None, full_dataset=None):

    with torch.no_grad():
        if sampled is None:
            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)
            sampled = prob_dist.sample()

        loss_list = []
        for item in eval_data:
            text, image_list, target_out, _ = model_helper.format_func(filtered_dataset=None, full_dataset=full_dataset, cur_item=item, num_shot=0, split="test", model_helper=model_helper)
            new_input = model_helper.insert_image(text, image_list)

            if model_helper.space:
                target_out = " " + target_out

            # Extract first turn from target output
            target_out = extract_first_turn(target_out)

            target_token = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0][model_helper.nonspecial_idx].unsqueeze(dim=0).to("cuda")

            out_logit = reinforce_activation_replacement(new_input, mean_activations, model_helper, sampled, last_token_only=True)
            task_loss = torch.nn.functional.cross_entropy(out_logit, target_token)

            loss_list.append(task_loss)


        print(f"validation loss at {epoch} epoch:", torch.tensor(loss_list).mean())
    return torch.tensor(loss_list).mean().item()


def avg_reinforce(mean_activations, model_helper, reinforce_data, eval_data, full_dataset):

    """
    This function performs Reinforce to select the attentions that encodes ICL examples.
    It computes the average loss over all target tokens instead of the first generated token.

    Parameters:
    mean_activations: From get_last_mean_head_activations
    model_helper:
    reinforce_data: Dataset used during reinforce optimization
    eval_data: Dataset used for Validation

    Returns: 
    bernoullis: A tensor of bernoullis variable. One variable for each attention heads. Each denote the probability of selecting this attention head.
    """

    num_layer = model_helper.model_config["n_layers"]
    num_heads = model_helper.model_config["n_heads"]
    lr = 0.1
    eps = 1e-3
    epoch = 600

    #(num_layer, num_head)
    bernoullis = [torch.neg(torch.ones(num_heads)).requires_grad_() for _ in range(num_layer)]
    optim = torch.optim.Adam(bernoullis, lr=lr)
    with torch.set_grad_enabled(True):

        for epoch in tqdm(range(epoch)):
            
            loss_list = []
            saved_log_probs = []

            text, image_list, target_out, _ = model_helper.format_func(reinforce_data, full_dataset, None, num_shot=0, model_helper=model_helper)

            if type(target_out)==list:
                target_out = target_out[0]


            ###Constructing a label for taking average loss
            if model_helper.space:
                            target_out = " " + target_out
            # target_token = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0][model_helper.nonspecial_idx].unsqueeze(dim=0).to("cuda")
            input_full = model_helper.insert_image(text, image_list, gt=target_out)
            labels = input_full[0].clone()
            target_len = model_helper.tokenizer(target_out, return_tensors='pt')["input_ids"][0].shape[0]
            labels[:, :-target_len] = -100


            sigmoid_tensor = torch.stack([torch.sigmoid(bernoulli).clamp(min=eps, max=1-eps) for bernoulli in bernoullis])
            prob_dist = torch.distributions.Bernoulli(sigmoid_tensor)

            ###Sampling the distribution many times to reduce variance. Each 
            for _ in range(8):

                ##Current sample
                sampled = prob_dist.sample()
                saved_log_probs.append(prob_dist.log_prob(sampled))

                with torch.no_grad():
                    out= reinforce_activation_replacement(input_full, mean_activations, model_helper, sampled, last_token_only=True, gt=labels, intervention_token=-target_len-1)
                    loss_list.append(out)

            policy_loss = []
            loss_list = torch.tensor(loss_list)
            loss_list = (loss_list - loss_list.mean())/(loss_list.std() + eps)

            for log_prob, R in zip(saved_log_probs, loss_list):
                policy_loss.append(log_prob * R)

            optim.zero_grad()
            policy_loss = torch.cat(policy_loss).sum()
            policy_loss.backward()
            optim.step()
            torch.cuda.empty_cache()
            if epoch % 50 == 0:
                print(policy_loss.item())
                validate_reinforce(model_helper, bernoullis, eps, mean_activations, eval_data, epoch)
    return bernoullis


def reinforce_activation_replacement(model_input, avg_activations, model_helper, sampled, last_token_only=True, gt=None, intervention_token=None):

    """
    This function performs Reinforce to select the attentions that encodes ICL examples.

    Parameters:
    model_input: Input to the forward function. Refer to model.py
    avg_activations:get_last_mean_head_activations
    model_helper:
    sampeld:
    last_token_only:

    Returns: 
    output: The logit of the first output token
    """

    ###This function returns a list of locations to perform intervention on based on sampled. List((layer, head, token_idx)). Token_idx is default to -1, meaning we always perform intervention on the generated token
    intervention_locations = reinforce_intervention_location(sampled)


    intervention_fn = last_replace_activation_w_avg(layer_head_token_pairs=intervention_locations, avg_activations=avg_activations, 
                                                model=model_helper.model, model_config=model_helper.model_config,
                                                batched_input=False, last_token_only=last_token_only, split_idx=model_helper.split_idx, intervention_token=intervention_token)

    with TraceDict(model_helper.model, layers=model_helper.model_config['attn_hook_names'], edit_output=intervention_fn, retain_grad=True) as td: 
        if gt is None:               
            output = model_helper.forward(model_input, labels=gt).logits[:,-1,:] # batch_size x n_tokens x vocab_size, only want last token prediction
        else:
            output = model_helper.forward(model_input, labels=gt).loss

    return output


def reinforce_intervention_location(sampled, categorical=None, token_idx = -1):
    intervention_locations = []
    #(layer, head)

    patch_idx = torch.nonzero(sampled)
    count = 0
    for _ in patch_idx:
        cur_layer = _[0]
        cur_head = _[1]
        intervention_locations.append((cur_layer, cur_head, -1))

    return intervention_locations


###Based on Function Vector: https://github.com/ericwtodd/function_vectors/blob/874d6e93c099d71fe4a2d76551fab233e60062c2/src/utils/intervention_utils.py#L16
def last_replace_activation_w_avg(layer_head_token_pairs, avg_activations, model, model_config, batched_input=False, last_token_only=False, patching=False, replace_layer = 0, split_idx=2, intervention_token=None):

    """
    This function performs intervention on during generation.

    This function defaults to perform intervention during the full generation. To perform intervention on certain token/generation step, modify the function accordingly.
    """


    if patching:
        edit_layers = [replace_layer]
    else:
        edit_layers = [x[0] for x in layer_head_token_pairs]


    def rep_act(output, layer_name, inputs):
        current_layer = int(layer_name.split('.')[split_idx])

        token_len = inputs[0].shape[1]
        if current_layer in edit_layers:
            if isinstance(inputs, tuple):
                inputs = inputs[0]

            
            # Determine shapes for intervention
            original_shape = inputs.shape
            new_shape = inputs.size()[:-1] + (model_config['n_heads'], model_config['resid_dim']//model_config['n_heads']) # split by head: + (n_attn_heads, hidden_size/n_attn_heads)
            inputs = inputs.view(*new_shape) # inputs shape: (batch_size , tokens (n), heads, hidden_dim)

            # Patch activations only at the last token for interventions like


            # cloned_inputs = inputs.clone()

            if last_token_only:

                for (layer,head_n, token_n) in layer_head_token_pairs:

                    if layer == current_layer:
   
                        #cloned_inputs[-1,-1,head_n] = avg_activations[layer,head_n,0]
                        inputs[-1,-1,head_n] = avg_activations[layer,head_n,0]

            elif intervention_token is not None:
                for (layer,head_n, token_n) in layer_head_token_pairs:

                    if layer == current_layer:
   
                        #cloned_inputs[-1,intervention_token,head_n] = avg_activations[layer,head_n,0]
                        inputs[-1,intervention_token,head_n] = avg_activations[layer,head_n,0]

            #cloned_inputs = cloned_inputs.view(*original_shape)
            inputs = inputs.view(*original_shape)

            proj_module = get_module(model, layer_name)

            out_proj = proj_module.weight

            #new_output = torch.matmul(cloned_inputs, out_proj.T)
            new_output = torch.matmul(inputs, out_proj.T)

            return new_output
        else:
            return output
    return rep_act


def compute_avg_perplexities(
    model_input,
    model_helper,
    ref_utterances,
    intervention_fn=None,
    skip_generation=False
):
    """
    Compute average perplexities for each dialogue act using reference utterances.
    
    Args:
        model_input: Tokenized input to model
        model_helper: ModelHelper instance
        ref_utterances: Dict mapping dialogue acts to lists of reference utterances
        intervention_fn: Optional intervention function
        skip_generation: Whether to skip generation and only compute perplexities
        
    Returns:
        Dict mapping dialogue acts to average perplexities
    """
    avg_perplexities = {}
    
    for act, references in ref_utterances.items():
        if not references:  # Skip empty reference sets
            continue
            
        # Compute perplexity for each reference
        perplexities = []
        for ref in references:
            _, ppl = model_helper.generate_and_score(
                prefix_input={k: v.clone() if hasattr(v, 'clone') else v for k, v in model_input.items()},
                target_text=ref,
                max_new_tokens=len(ref.split()) + 5,  # Add small buffer
                intervention_fn=intervention_fn,
                skip_generation=skip_generation
            )
            if ppl is not None:
                perplexities.append(ppl)
                
        # Compute average if we have valid perplexities
        if perplexities:
            avg_perplexities[act] = sum(perplexities) / len(perplexities)
            
    return avg_perplexities


def fv_intervention_natural_text(
    model_input,
    model_helper,
    max_new_tokens,
    return_item="both",
    intervention_locations=None,
    avg_activations=None,
    target_output=None,
    ref_utterances=None,
    f=None,
    skip_generation=False
):
    """
    Run clean and intervention passes on model input, returning generated text and perplexities.
    
    Args:
        model_input: Tokenized input to model
        model_helper: ModelHelper instance
        max_new_tokens: Max tokens to generate
        return_item: What to return - "clean", "interv", or "both"
        intervention_locations: Locations to intervene on
        avg_activations: Average activations to use for intervention
        target_output: Target text for perplexity calculation
        ref_utterances: Optional dict of reference utterances per dialogue act
        f: Optional file handle for debug logging
        skip_generation: Whether to skip generation and only compute perplexities
        
    Returns:
        clean_text: Text from clean pass (empty string if skip_generation=True)
        interv_text: Text from intervention pass (empty string if skip_generation=True)
        clean_ppl: Perplexity from clean pass (or dict of avg perplexities if using ref_utterances)
        interv_ppl: Perplexity from intervention pass (or dict of avg perplexities if using ref_utterances)
    """
    # Clean pass
    if ref_utterances is not None:
        # Use reference sets
        clean_text, _ = model_helper.generate_and_score(
            prefix_input=model_input,
            target_text=target_output,  # Still need this for generation
            max_new_tokens=max_new_tokens,
            intervention_fn=None,
            skip_generation=skip_generation
        )
        clean_ppl = compute_avg_perplexities(
            model_input,
            model_helper,
            ref_utterances,
            skip_generation=skip_generation
        )
    else:
        # Original single-target behavior
        clean_text, clean_ppl = model_helper.generate_and_score(
            prefix_input=model_input,
            target_text=target_output,
            max_new_tokens=max_new_tokens,
            intervention_fn=None,
            skip_generation=skip_generation
        )

    # Intervention pass
    if return_item in ("interv", "both"):
        interv_fn = last_replace_activation_w_avg(
            layer_head_token_pairs=intervention_locations,
            avg_activations=avg_activations,
            model=model_helper.model,
            model_config=model_helper.model_config,
            batched_input=False,
            last_token_only=True,
            split_idx=model_helper.split_idx
        )
        
        if ref_utterances is not None:
            # Use reference sets
            interv_text, interv_ppl = model_helper.generate_and_score(
                prefix_input=model_input,
                target_text=target_output,  # Still need this for generation
                max_new_tokens=max_new_tokens,
                intervention_fn=interv_fn,
                skip_generation=skip_generation
            )
            interv_ppl = compute_avg_perplexities(
                model_input,
                model_helper,
                ref_utterances,
                intervention_fn=interv_fn,
                skip_generation=skip_generation
            )
        else:
            # Original single-target behavior
            interv_text, interv_ppl = model_helper.generate_and_score(
                prefix_input=model_input,
                target_text=target_output,
                max_new_tokens=max_new_tokens,
                intervention_fn=interv_fn,
                skip_generation=skip_generation
            )
    else:
        interv_text, interv_ppl = None, None

    return clean_text, interv_text, clean_ppl, interv_ppl


def eval_vqa(cur_dataset, results_path, answers):
    ds_collections = {
        'vizwiz_val': {
        'train': '../data/vizwiz/vizwiz_train.jsonl',
        'test': '../data/vizwiz/vizwiz_val.jsonl',
        'question': '../data/vizwiz/vizwiz_val_questions.json',
        'annotation': '../data/vizwiz/vizwiz_val_annotations.json',
        'metric': 'vqa_score',
        'max_new_tokens': 10,
    },
        'okvqa_val': {
            'train': '../data/okvqa/okvqa_train.jsonl',
            'test': '../data/okvqa/okvqa_val.jsonl',
            'question': '../data/okvqa/OpenEnded_mscoco_val2014_questions.json',
            'annotation': '../data/okvqa/mscoco_val2014_annotations.json',
            'metric': 'vqa_score',
            'max_new_tokens': 10,
        }
    }
    if answers is not None:
        result_file = open(results_path, 'w')
        result_file.write(json.dumps(answers))
        result_file.close()


    vqa = VQA(ds_collections[cur_dataset]['annotation'],
                ds_collections[cur_dataset]['question'])
    results = vqa.loadRes(
        resFile=results_path,
        quesFile=ds_collections[cur_dataset]['question'])
    vqa_scorer = VQAEval(vqa, results, n=2)
    vqa_scorer.evaluate()
    print(vqa_scorer.accuracy)


def get_classifier(dialogue_act, contextual=False):
    """
    Load the appropriate classifier for a given dialogue act.
    
    Parameters:
    dialogue_act: The dialogue act to classify (e.g., 'sd', 'sv', 'b', 'aa', '%', etc.)
    contextual: Whether to use contextual classifiers that consider previous utterance
    
    Returns:
    classifier: The loaded classifier model
    classify_func: A function that takes (utterance, previous_utterance=None) and returns predicted act
    """
    
    # Define mapping from dialogue acts to classifier types
    act_to_classifier = {
        'sd': 'declarative',
        'sv': 'statement_opinion', 
        'b': 'backchannel',
        'aa': 'agreement',
        '%': 'abandoned',
        # Add more mappings as needed
    }
    
    # Get classifier type or default to backchannel for unknown acts
    classifier_type = act_to_classifier.get(dialogue_act, 'backchannel')
    
    if contextual:
        # Load contextual classifiers (only available for sd, sv, and b)
        if classifier_type == 'declarative':
            from contextual_sd_classifier import load_classifier
            classifier = load_classifier()
            def classify_func(utterance, previous_utterance=""):
                return 'sd' if classifier.classify_utterance(utterance, previous_utterance, pre_cleaned=True) else 'o'
        elif classifier_type == 'statement_opinion':
            from contextual_sv_classifier import load_classifier
            classifier = load_classifier()
            def classify_func(utterance, previous_utterance=""):
                return 'sv' if classifier.classify_utterance(utterance, previous_utterance, pre_cleaned=True) else 'o'
        elif classifier_type == 'backchannel':
            from contextual_backchannel_classifier import load_classifier
            classifier = load_classifier()
            def classify_func(utterance, previous_utterance=""):
                return 'b' if classifier.classify_utterance(utterance, previous_utterance, pre_cleaned=True) else 'o'
        else:
            # For agreement and abandoned, fall back to binary classifiers since contextual versions don't exist
            print(f"Warning: Contextual classifier not available for {dialogue_act}, using binary classifier instead")
            if classifier_type == 'agreement':
                from agreement_classifier import load_classifier
                classifier = load_classifier()
                def classify_func(utterance, previous_utterance=""):
                    return 'aa' if classifier.classify_utterance(utterance) else 'o'
            elif classifier_type == 'abandoned':
                from abandoned_classifier import load_classifier
                classifier = load_classifier()
                def classify_func(utterance, previous_utterance=""):
                    return '%' if classifier.classify_utterance(utterance) else 'o'
            else:
                # Fallback to backchannel
                from backchannel_classifier import load_classifier
                classifier = load_classifier()
                def classify_func(utterance, previous_utterance=""):
                    return 'b' if classifier.classify_utterance(utterance) else 'o'
    else:
        # Load binary classifiers
        if classifier_type == 'declarative':
            from new_classifiers.declarative_classifier import load_classifier
            classifier = load_classifier()
            def classify_func(utterance, previous_utterance=""):
                return 'sd' if classifier.classify_utterance(utterance) else 'o'
        elif classifier_type == 'statement_opinion':
            from statement_opinion_classifier import load_classifier
            classifier = load_classifier()
            def classify_func(utterance, previous_utterance=""):
                return 'sv' if classifier.classify_utterance(utterance) else 'o'
        elif classifier_type == 'backchannel':
            from backchannel_classifier import load_classifier
            classifier = load_classifier()
            def classify_func(utterance, previous_utterance=""):
                return 'b' if classifier.classify_utterance(utterance) else 'o'
        elif classifier_type == 'agreement':
            from agreement_classifier import load_classifier
            classifier = load_classifier()
            def classify_func(utterance, previous_utterance=""):
                return 'aa' if classifier.classify_utterance(utterance) else 'o'
        elif classifier_type == 'abandoned':
            from abandoned_classifier import load_classifier
            classifier = load_classifier()
            def classify_func(utterance, previous_utterance=""):
                return '%' if classifier.classify_utterance(utterance) else 'o'
        else:
            # Fallback to backchannel
            from backchannel_classifier import load_classifier
            classifier = load_classifier()
            def classify_func(utterance, previous_utterance=""):
                return 'b' if classifier.classify_utterance(utterance) else 'o'
    
    return classifier, classify_func


def classify_dialogue_act(utterance, target_act, classifiers_cache=None, previous_utterance="", contextual=False):
    """
    Classify a single utterance for a specific target dialogue act.
    
    Parameters:
    utterance: The utterance to classify
    target_act: The target dialogue act to check for ('sd', 'sv', 'b', 'aa', '%', or 'o')
    classifiers_cache: Optional dict to cache loaded classifiers
    previous_utterance: Previous turn for contextual classifiers
    contextual: Whether to use contextual classifiers
    
    Returns:
    predicted_act: The predicted dialogue act ('sd', 'sv', 'b', 'aa', '%', or 'o')
    """
    
    # Initialize cache if not provided
    if classifiers_cache is None:
        classifiers_cache = {}
    
    # Check if classifier is already cached
    cache_key = f"{target_act}_{'contextual' if contextual else 'binary'}"
    if cache_key not in classifiers_cache:
        classifier, classify_func = get_classifier(target_act, contextual=contextual)
        classifiers_cache[cache_key] = (classifier, classify_func)
    
    # Use cached classifier
    classifier, classify_func = classifiers_cache[cache_key]
    
    # Classify the utterance
    if contextual:
        predicted_act = classify_func(utterance, previous_utterance)
    else:
        predicted_act = classify_func(utterance)
    
    return predicted_act

def build_reference_sets(train_dataset, dialogue_acts, size_per_act, seed=42):
    """
    Build reference sets of utterances for each dialogue act from training data. --> FOR PERPLEXITY
    
    Args:
        train_dataset: List of training examples
        dialogue_acts: List of dialogue acts to collect references for
        size_per_act: Number of reference utterances to sample per act
        seed: Random seed for reproducibility
        
    Returns:
        Dict mapping dialogue acts to lists of reference utterances
    """
    random.seed(seed)
    
    # Group training examples by dialogue act
    act_to_examples = {}
    for ex in train_dataset:
        act = ex.get('dialog_act', 'o')
        if act not in act_to_examples:
            act_to_examples[act] = []
        act_to_examples[act].append(ex)
    
    # Sample reference utterances for each act
    ref_sets = {}
    for act in dialogue_acts:
        if act not in act_to_examples or not act_to_examples[act]:
            print(f"[WARNING] No training examples found for act '{act}', using empty reference set")
            ref_sets[act] = []
            continue
            
        # Sample with replacement if we need more than we have
        examples = random.choices(act_to_examples[act], k=size_per_act) if size_per_act > len(act_to_examples[act]) else random.sample(act_to_examples[act], size_per_act)
        # Extract the response; fall back to other fields if empty
        utterances = []
        for ex in examples:
            utt = ex.get('response', '')
            if not utt:
                utt = ex.get('target_out', '') if 'target_out' in ex else ex.get('text', '')
            if utt:
                utterances.append(utt)
        ref_sets[act] = utterances

    # Print reference sets to file
    with open('reference_sets.txt', 'w') as f:
        f.write("\n[INFO] Reference sets:\n")
        for act, utterances in ref_sets.items():
            f.write(f"\n{act} ({len(utterances)} utterances):\n")
            for i, utt in enumerate(utterances, 1):
                f.write(f"  {i}. {utt}\n")
        f.write("\n") # Add blank line after reference sets
        
    return ref_sets

def build_multiple_choice_candidates(act_to_utterances, target_act, target_out, max_choices=5):
    """
    Build multiple choice candidates for a given example, sampling one utterance per dialogue act.
    
    Args:
        act_to_utterances: Dict mapping dialogue acts to lists of utterances
        target_act: The target dialogue act for this example
        target_out: The target/gold utterance to exclude from sampling
        max_choices: Maximum number of choices to include (including target act)
        
    Returns:
        Dict mapping dialogue acts to lists containing one sampled utterance each
    """
    # Get all available acts that have utterances
    available_acts = [act for act, utts in act_to_utterances.items() if utts]
    
    # Always include target act's pool (for fair comparison)
    if target_act in available_acts:
        available_acts.remove(target_act)
    
    # Sample N-1 other acts (where N is total number of acts)
    num_other_acts = min(max_choices - 1, len(available_acts))
    sampled_acts = []
    if available_acts:
        sampled_acts = random.sample(available_acts, num_other_acts)
        if target_act not in sampled_acts:
            sampled_acts.append(target_act)
    else:
        sampled_acts = [target_act] if target_act else []
    
    # Sample one utterance per act, ensuring we don't pick the target utterance
    current_ref_utterances = {}
    for act in sampled_acts:
        if not act_to_utterances.get(act):
            continue
            
        # Get utterance pool excluding the target
        utt_pool = [u for u in act_to_utterances[act] if u.strip() != target_out.strip()]
        
        if utt_pool:  # Only add if we have valid candidates
            sampled_utt = random.choice(utt_pool)
            current_ref_utterances[act] = [sampled_utt]
            
    return current_ref_utterances

def update_multiple_choice_stats(example_data, clean_act, interv_act, target_act, clean_ppl, interv_ppl, args):
    """
    Update example statistics for multiple choice evaluation.
    
    Args:
        example_data: Dict containing example-level data to update
        clean_act: Act selected by clean model
        interv_act: Act selected by intervention model
        target_act: Ground truth target act
        clean_ppl: Clean model perplexities (not used in letter-based approach)
        interv_ppl: Intervention model perplexities (not used in letter-based approach)
        args: Runtime arguments
        
    Returns:
        Tuple of (clean_patch_selected, interv_patch_selected) indicating if patch was selected
    """
    clean_patch_selected = False
    interv_patch_selected = False
    
    if args.dialogue_act is not None:
        # Track if model selected the target act
        clean_correct = clean_act == target_act
        example_data["clean_selected_correct"] = clean_correct
        example_data["clean_selected_act"] = clean_act
        
        # Track if model selected the patched act
        if clean_act == args.dialogue_act:
            clean_patch_selected = True
            example_data["clean_selected_patch"] = True
        else:
            example_data["clean_selected_patch"] = False
            
        if args.cur_mode in ("interv", "both"):
            # Track if intervention model selected correctly
            interv_correct = interv_act == target_act
            example_data["interv_selected_correct"] = interv_correct
            example_data["interv_selected_act"] = interv_act
            
            # Track if intervention model selected patched act
            if interv_act == args.dialogue_act:
                interv_patch_selected = True
                example_data["interv_selected_patch"] = True
            else:
                example_data["interv_selected_patch"] = False
            
    return clean_patch_selected, interv_patch_selected

def update_multiple_choice_summary(summary, clean_patch_count, interv_patch_count, total_examples, args):
    """
    Update summary statistics for multiple choice evaluation (patch selection rates).
    
    Args:
        summary: Dict containing summary statistics to update
        clean_patch_count: Number of times clean model selected patch
        interv_patch_count: Number of times intervention model selected patch
        total_examples: Total number of examples evaluated
        args: Runtime arguments
    """
    if args.multiple_choice and args.dialogue_act is not None:
        summary["clean_patch_selection_rate"] = clean_patch_count / total_examples
        summary["clean_patch_selections"] = clean_patch_count
        summary["total_examples"] = total_examples
        
        if args.cur_mode in ("interv", "both"):
            summary["intervention_patch_selection_rate"] = interv_patch_count / total_examples
            summary["intervention_patch_selections"] = interv_patch_count
            summary["patch_selection_rate_increase"] = summary["intervention_patch_selection_rate"] - summary["clean_patch_selection_rate"]
            summary["patch_selection_absolute_increase"] = interv_patch_count - clean_patch_count