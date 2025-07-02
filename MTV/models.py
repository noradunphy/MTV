from mtv_utils import *
from preprocess import *
from PIL import Image
import torch
import copy
import os
import pdb
import math
from baukit import TraceDict

# from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
# from llava.conversation import conv_templates, SeparatorStyle
# from llava.mm_utils import process_images, tokenizer_image_token

def load_image(image_file):
    try:
        # Remove ./ prefix if present
        if image_file.startswith('./'):
            image_file = image_file[2:]
        # Remove flowers-102/ prefix if present
        if image_file.startswith('flowers-102/'):
            image_file = image_file[12:]
        # Handle relative paths
        if not os.path.isabs(image_file):
            # If the path already starts with jpg/, just prepend data/flower
            if image_file.startswith('jpg/'):
                image_file = os.path.join('data/flower', image_file)
            # If the path doesn't start with jpg/, prepend data/flower/jpg
            else:
                image_file = os.path.join('data/flower/jpg', image_file)
        # Convert to absolute path for better error messages
        abs_path = os.path.abspath(image_file)
        # Remove MTV from path if present
        abs_path = abs_path.replace('/MTV/data/', '/data/')
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Image file not found: {abs_path}")
        image = Image.open(abs_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_file}: {str(e)}")
        return image_file
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out



class ModelHelper:
    def __init__(self):

        """
        self.model:The loaded model
        self.tokenizer: The loaded tokenizer
        self.model_config: The architecture of the model. Might need to do print(model) see how to initialize
        self.format_func: The format function for the current dataset
        self.space: Whether the model output will have a leading space
        self.cur_dataset: Name of the current dataset
        self.split_idx: The index of "layer" when you parse "attn_hook_names" with "."
        self.nonspecial_idx: The index in which the generated tokens are not special token. Used to skip special token and construct the current target output for loss calculation.
        self.zero_shot: Whether to use zero-shot evaluation for applicable datasets
        """


    #Always return a single variable. If both text and image is returned, return in tuple
    def insert_image(self, text, image_list):

        """
        Returns an object that is the input to forward and generate.
        """
        pass
    #Takes the output of insert_image
    def forward(self, model_input, labels=None):

        """
        Forwrad function wrapper
        """

        pass
    #Takes the output of insert image
    def generate(self, model_input, max_new_tokens):

        """
        Generate function wrapper
        """
        pass


class llavaOAHelper(ModelHelper):

    def __init__(self, model, tokenizer, processor, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor
        self.model_config = {"n_heads":model.model.config.num_attention_heads,
                    "n_layers":model.model.config.num_hidden_layers,
                    "resid_dim":model.model.config.hidden_size,
                    "name_or_path":model.model.config._name_or_path,
                    "attn_hook_names":[f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.config.num_hidden_layers)],
                    "layer_hook_names":[f'model.layers.{layer}' for layer in range(model.model.config.num_hidden_layers)],
                    "mlp_hook_names": [f'model.layers.{layer}.mlp.down_proj' for layer in range(model.model.config.num_hidden_layers)]}
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.nonspecial_idx = 0


    def insert_image(self, text, image_list, gt=None):

        conv_template = "qwen_1_5"
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        if gt is not None:
            prompt_question = prompt_question + gt

        input_ids = tokenizer_image_token(prompt_question, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")

        if image_list == []:
            return (input_ids, None, None)

        image_list = load_images(image_list)
        image_sizes = [image.size for image in image_list]

        image_tensors = process_images(image_list, self.processor, self.model.config)
        image_tensors = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensors]

        return (input_ids, image_tensors, image_sizes)
    

    def forward(self, model_input, labels=None):

        result = self.model(model_input[0],
            images=model_input[1],
            image_sizes=model_input[2],
            labels=labels) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    

    def generate(self, model_input, max_new_tokens):

        cont = self.model.generate(
            model_input[0],
            images=model_input[1],
            image_sizes=model_input[2],
            do_sample=False,
            temperature=0,

            max_new_tokens=max_new_tokens,
        )
        
        return self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]


class QwenHelper(ModelHelper):

    def __init__(self, model, tokenizer, cur_dataset, zero_shot=False):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = {"n_heads":model.transformer.config.num_attention_heads,
                    "n_layers":model.transformer.config.num_hidden_layers,
                    "resid_dim":model.transformer.config.hidden_size,
                    "name_or_path":model.transformer.config._name_or_path,
                    "attn_hook_names":[f'transformer.h.{layer}.attn.c_proj' for layer in range(model.transformer.config.num_hidden_layers)],
                    "layer_hook_names":[f'transformer.h.{layer}' for layer in range(model.transformer.config.num_hidden_layers)]}
        self.format_func = get_format_func(cur_dataset, zero_shot=zero_shot)
        self.space = True
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.nonspecial_idx = 0
        self.question_lookup = None
        self.zero_shot = zero_shot

    def insert_image(self, text, image_list):
        # Preprocess image paths
        processed_images = []
        for image_path in image_list:
            # Handle different dataset paths
            if self.cur_dataset == "flower":
                if image_path.startswith('jpg/'):
                    image_path = os.path.join('data/flower', image_path)
                else:
                    image_path = os.path.join('data/flower/jpg', image_path)
            elif self.cur_dataset == "cub":
                # Remove any leading ./ if present
                if image_path.startswith('./'):
                    image_path = image_path[2:]
                # Join with the correct base path, preserving CUB_200_2011 in the path
                if not image_path.startswith('CUB_200_2011/'):
                    image_path = os.path.join('CUB_200_2011', image_path)
                image_path = os.path.join('MTV/data/cub', image_path)
            # Convert to absolute path for better error messages
            abs_path = os.path.abspath(image_path)
            # Remove MTV from path if present
            abs_path = abs_path.replace('/MTV/data/', '/data/')
            if not os.path.exists(abs_path):
                raise FileNotFoundError(f"Image file not found: {abs_path}")
            processed_images.append(abs_path)

        text = text.replace("<image>", "<img></img>")
        text = text.split("</img>")

        new_text = ""
        for text_split, image in zip(text[:-1], processed_images):
            new_text += f"{text_split}{image}</img>"
        return self.tokenizer(new_text + text[-1], return_tensors='pt', padding='longest')
    

    def forward(self, model_input, labels=None):

        result = self.model(input_ids=model_input["input_ids"].to(self.model.device),
                attention_mask=model_input["attention_mask"].to(self.model.device)) # batch_size x n_tokens x vocab_size, only want last token prediction
        return result
    

    def generate(self, model_input, max_new_tokens):

        generated_output = self.model.generate(
                input_ids=model_input["input_ids"].to(self.model.device),
                attention_mask=model_input["attention_mask"].to(self.model.device),
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,
                pad_token_id=self.tokenizer.eod_id,
                eos_token_id=self.tokenizer.eod_id,)
        
        return self.tokenizer.batch_decode(generated_output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()
    
    
class ViLAHelper(ModelHelper):

    def __init__(self, model, tokenizer, image_processor, cur_dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = {"n_heads":model.llm.model.config.num_attention_heads,
                        "n_layers":model.llm.model.config.num_hidden_layers,
                        "resid_dim":model.llm.model.config.hidden_size,
                        "name_or_path":model.llm.model.config._name_or_path,
                        "attn_hook_names":[f'llm.model.layers.{layer}.self_attn.o_proj' for layer in range(model.llm.model.config.num_hidden_layers)],
                        "layer_hook_names":[f'llm.model.layers.{layer}' for layer in range(model.llm.model.config.num_hidden_layers)]}
    
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 0
        self.question_lookup = None


    ##No need to change the image token since it's the same as default
    def insert_image(self, text, image_list):

        text = text.replace("<image>", "<image>\n")

        if image_list is not None:
            images = load_images(image_list)
            images_tensor = process_images(images, self.image_processor, self.model.config).to(self.model.device, dtype=torch.float16)
        else:
            images_tensor = None

        conv_mode = "llama_3"
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
            

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        return (input_ids, images_tensor, stopping_criteria, stop_str)
    

    def forward(self, model_input, labels=None): 
    
        result = self.model(model_input[0], images=[model_input[1]], labels=labels)
        return result
    

    def generate(self, model_input, max_new_tokens):

        if model_input[1] is None:
            output = self.model.generate(
                model_input[0],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1,
                use_cache=True,
                stopping_criteria=[model_input[2]])    
        else:

            output = self.model.generate(
                    model_input[0],
                    images=[
                        model_input[1],
                    ],
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    min_new_tokens=1,
                    use_cache=True,
                    stopping_criteria=[model_input[2]])
    
        output = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        output = output.strip()
        if output.endswith(model_input[3]):
            output = output[: -len(model_input[3])]
        output = output.strip()
        return output 
    

class Idefics2Helper(ModelHelper):

    def __init__(self, model, processor, cur_dataset):
        self.model = model
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model_config = {"n_heads":model.model.text_model.config.num_attention_heads,
                        "n_layers":model.model.text_model.config.num_hidden_layers,
                        "resid_dim":model.model.text_model.config.hidden_size,
                        
                        "name_or_path":model.model.text_model.config._name_or_path,
                        "attn_hook_names":[f'model.text_model.layers.{layer}.self_attn.o_proj' for layer in range(model.model.text_model.config.num_hidden_layers)],
                        "layer_hook_names":[f'model.text_model.layers.{layer}' for layer in range(model.model.text_model.config.num_hidden_layers)]}
    
        self.format_func = get_format_func(cur_dataset)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 3
        self.nonspecial_idx = 1


    def insert_image(self, text, image_list):

        opened_images = load_images(image_list)
        inputs = self.processor(text=[text], images=[opened_images], padding=True, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        return inputs


    def forward(self, model_input, labels=None):
        result = self.model(**model_input)
        return result
    

    def generate(self, model_input, max_new_tokens):

        output = self.model.generate(
                **model_input,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                min_new_tokens=1,
                length_penalty=1,
                num_return_sequences=1,
                output_hidden_states=True,
                use_cache=True,)
        
        output = self.processor.batch_decode(output[:, model_input["input_ids"].size(1):],
                            skip_special_tokens=True)[0].strip()
        return output

#NEW
class TextModelHelper(ModelHelper):
    def __init__(self, model, tokenizer, cur_dataset, zero_shot=False):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "resid_dim": model.config.hidden_size,
            "name_or_path": model.config._name_or_path,
            "attn_hook_names": [f'model.layers.{layer}.self_attn.o_proj' for layer in range(model.config.num_hidden_layers)],
            "layer_hook_names": [f'model.layers.{layer}' for layer in range(model.config.num_hidden_layers)],
            "mlp_hook_names": [f'model.layers.{layer}.mlp.down_proj' for layer in range(model.config.num_hidden_layers)]
        }
        self.format_func = get_format_func(cur_dataset, zero_shot=zero_shot)
        self.space = False
        self.cur_dataset = cur_dataset
        self.split_idx = 2
        self.nonspecial_idx = 0
        self.zero_shot = zero_shot

    def insert_image(self, text, image_list):
        # For text-only models, we ignore image_list and just process the text
        # check: what index is the colon, is it last?
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        return inputs

    def forward(self, model_input, labels=None):
        outputs = self.model(**model_input, labels=labels, use_cache=False)
        return outputs
    
    def generate(self, model_input, max_new_tokens, return_scores=False, return_dict_in_generate=False, **generate_kwargs):
        """
        Wrapper around HuggingFace `model.generate()` that lets you
        pull out raw logits (when return_scores=True) and control
        return_dict_in_generate, plus any other HF generate args.
        """
        # Ensure we ask for dict form if we need scores; HF returns a Tensor otherwise.
        effective_return_dict = return_scores or return_dict_in_generate

        # Only pass required parameters and any additional kwargs
        generate_params = {
            **model_input,
            "max_new_tokens": max_new_tokens,
            "output_scores": return_scores,
            "return_dict_in_generate": effective_return_dict,
            **generate_kwargs
        }

        outputs = self.model.generate(**generate_params)

        # HF returns either a Tensor (when return_dict=False) or a GenerateDecoderOnlyOutput / similar.
        sequences = outputs.sequences if not isinstance(outputs, torch.Tensor) else outputs

        # strip off the prompt tokens
        input_len = model_input["input_ids"].size(1)
        gen_tokens = sequences[:, input_len:]
        decoded = self.tokenizer.batch_decode(
            gen_tokens, skip_special_tokens=True
        )[0].strip()
        from mtv_utils import extract_first_turn
        cleaned_output = extract_first_turn(decoded).lstrip('0123456789: ')

        if return_scores:
            # If we requested scores but outputs is Tensor (shouldn't happen), return None for scores.
            raw_scores = outputs.scores if not isinstance(outputs, torch.Tensor) else None
            return cleaned_output, raw_scores

        return cleaned_output
    
    def _compute_autoregressive_perplexity(self, model, tokenizer, prefix_input, target_text, device):
        """Helper method to compute perplexity using autoregressive token-by-token approach."""
        # Check if target_text is empty or None
        if not target_text or target_text.strip() == "":
            return None  # Return None for perplexity when there's no target text
            
        tg = tokenizer(target_text, return_tensors="pt", add_special_tokens=False).to(device)
        target_ids = tg['input_ids'][0]  # Remove batch dimension
        
        # Start with just the prefix input
        current_input = prefix_input.copy()
        total_loss = 0.0
        num_tokens = 0

        # Autoregressive loop over target tokens
        for i in range(len(target_ids)):
            # Get next target token
            target_token = target_ids[i:i+1].to(device)
            
            # Forward pass to get logits for next token
            outputs = model(**current_input, use_cache=False)
            logits = outputs.logits[:, -1, :]  # Last token prediction
            
            # Calculate loss for this step
            loss = torch.nn.functional.cross_entropy(
                logits, 
                target_token,
                reduction='sum'
            )
            total_loss += loss.item()
            num_tokens += 1
            
            # Add target token to input for next iteration
            current_input['input_ids'] = torch.cat([
                current_input['input_ids'],
                target_token.unsqueeze(0)
            ], dim=1)
            current_input['attention_mask'] = torch.ones_like(
                current_input['input_ids']
            )
        
        # Calculate final perplexity
        avg_loss = total_loss / num_tokens
        return math.exp(avg_loss)

    def generate_and_score(
        self,
        prefix_input,        # e.g. {"input_ids": ..., "attention_mask": ...}
        target_text,         # the gold string whose PPL you want
        max_new_tokens=10,
        intervention_fn=None, # your TraceDict edit_output function, or None
        skip_generation=False # whether to skip generation and only compute perplexity
    ):
        """
        Generate text and compute perplexity on target text, with optional intervention.
        Returns generated text and perplexity score.
        
        Args:
            prefix_input: Input tokens and attention mask
            target_text: Target text to compute perplexity on
            max_new_tokens: Maximum new tokens to generate (ignored if skip_generation=True)
            intervention_fn: Optional intervention function for attention layers
            skip_generation: Whether to skip generation and only compute perplexity
            
        Returns:
            gen_text: Generated text (empty string if skip_generation=True)
            ppl: Perplexity score on target_text
        """
        model = self.model
        tok = self.tokenizer
        device = next(model.parameters()).device

        # 1) GENERATION (if not skipped)
        if intervention_fn is not None:
            # Intervention branch - wrap model forward pass with hook
            with TraceDict(model, layers=self.model_config['attn_hook_names'], edit_output=intervention_fn):
                with torch.no_grad():
                    gen_text = "" if skip_generation else self.generate(
                        prefix_input,
                        max_new_tokens=max_new_tokens
                    )
                    
                    # 2) PPL ON TARGET via autoregressive loop
                    ppl = self._compute_autoregressive_perplexity(model, tok, prefix_input, target_text, device)
                    
        else:
            # Clean branch - no intervention
            with torch.no_grad():
                gen_text = "" if skip_generation else self.generate(
                    prefix_input,
                    max_new_tokens=max_new_tokens
                )
                
                # 2) PPL ON TARGET via autoregressive loop
                ppl = self._compute_autoregressive_perplexity(model, tok, prefix_input, target_text, device)

        return gen_text, ppl








