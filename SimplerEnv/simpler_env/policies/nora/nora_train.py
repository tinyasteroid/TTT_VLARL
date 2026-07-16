import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, PeftModel
from tqdm import tqdm
from transformers import AutoTokenizer, BatchFeature
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPredictionWithValueHead
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor



import PIL.Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, GenerationConfig
from typing import Optional, Union, Dict, Any
from qwen_vl_utils import process_vision_info
from huggingface_hub import hf_hub_download
import json
from simpler_env.policies.nora.nora import NoraForActionPrediction


import tensorflow as tf


import math

def huber_loss(e, d):
    a = (abs(e) <= d).to(torch.float32)
    b = (abs(e) > d).to(torch.float32)
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)


class NoraPolicy:
    def __init__(self, all_args, device_id: int):
        self.args = all_args
        self.device_id = device_id
        self.tpdv = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.bfloat16)
        self.tpdv_vn = dict(device=torch.device("cuda:" + str(device_id)), dtype=torch.float32)
        self.action_scale = 1.0
        '''
        # openvla: register
        self.image_processor = PrismaticImageProcessor.from_pretrained(self.args.vla_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.vla_path, trust_remote_code=True, padding_side="left")
        self.processor = PrismaticProcessor.from_pretrained(
            self.args.vla_path,
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            trust_remote_code=True
        )
        # self.action_tokenizer = ActionTokenizer(self.processor.tokenizer)
        self.vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )'''


        # --- Load Fast Tokenizer ---
        try:
            print(f"Loading fast tokenizer from: physical-intelligence/fast")
            self.fast_tokenizer = AutoProcessor.from_pretrained(
                "physical-intelligence/fast", trust_remote_code=True
            )
            # Ensure required attributes are set/exist
           
            self.fast_tokenizer.action_dim = 7 # Set default if not in config
            print("Setting action_dim  to 7.")
           
            self.fast_tokenizer.time_horizon = 1 # Set default if not in config
            print("Setting time horizon to 1.")

        except Exception as e:
            raise RuntimeError(
                f"Error loading fast tokenizer: {e}. "
            )

        # --- Load Main Processor ---
        try:
            print(f"Loading main processor from: {self.args.vla_path}")
            # Assuming the main processor is saved in the same location as the model
            self.processor = AutoProcessor.from_pretrained(
                #model_path, trust_remote_code=True
                self.args.vla_path, trust_remote_code=True
            )
        except Exception as e:
            raise RuntimeError(f"Error loading main processor from {self.args.vla_path}: {e}")

        # --- Load Main Model ---
        try:
            print(f"Loading model from: {self.args.vla_path}")
            self.vla = NoraForActionPrediction.from_pretrained(
                self.args.vla_path,
                torch_dtype=torch.bfloat16,
            #   attn_implementation="flash_attention_2", # Comment out this line if there is an error with flash attention
            )
            
            self.vla.to(self.device_id)
            self.vla.generation_config = GenerationConfig.from_pretrained(self.args.vla_path)
            self.vla.generation_config.do_sample = False


            

        except Exception as e:
            raise RuntimeError(f"Error loading model from {self.args.vla_path}: {e}")

        print("Model and processors loaded successfully.")

        #  lora
        if not self.args.vla_load_path:
            lora_config = LoraConfig(
                r=self.args.vla_lora_rank,
                lora_alpha=min(self.args.vla_lora_rank, 16),
                lora_dropout=0.0,
                target_modules=[
                    "proj", "qkv", 
                    "merger.mlp.0", "merger.mlp.2",  # vision
                    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",  # llm
                ],
                init_lora_weights="gaussian"
            )
            self.vla = get_peft_model(self.vla, lora_config)
        else:
            self.vla = PeftModel.from_pretrained(self.vla, self.args.vla_load_path, is_trainable=True)
            print(f"VLA load: {self.args.vla_load_path}")

            if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
                path = Path(self.args.vla_load_path) / "dataset_statistics.json"
                ds = json.load(open(path, "r"))
                self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]
        
        # set value head trainable
        #for name, param in self.vla.named_parameters():
        #    if "value_head" in name:
        #        param.requires_grad = True

        self.vla.print_trainable_parameters()
    
        # nora: optimizer
        self.params_vh = None
        self.params_vla = None
        self.vh_optimizer = None
        self.vla_optimizer = None
        self._setup_optimizer()

        if self.args.vla_load_path:
            training_state_path = Path(self.args.vla_load_path) / "training_state.pt"
            if training_state_path.exists():
                training_state = torch.load(training_state_path, map_location=self.tpdv["device"])

                if "vh" in training_state:
                    self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
                else:
                    print("Warning: value_head state not found in training_state")

                self._setup_optimizer()
                self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
                self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])

                print(f"Optimizer load: {self.args.vla_load_path}")
            else:
                print(f"Warning: training_state not found in {training_state_path}")

    def _setup_optimizer(self):
        self.params_vh = [p for n, p in self.vla.named_parameters() if "value_head" in n and p.requires_grad]
        self.params_vla = [p for n, p in self.vla.named_parameters() if "value_head" not in n and p.requires_grad]
        betas = (self.args.vla_optim_beta1, self.args.vla_optim_beta2)
        #self.vh_optimizer = AdamW(self.params_vh, lr=self.args.vla_vhlr, betas=betas)
        self.vla_optimizer = AdamW(self.params_vla, lr=self.args.vla_lr, betas=betas)


    def get_action(self, x: dict, deterministic) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        #TODO 1
        #这里需要改动， nora的get_action需要返回logprobs
        #看下nora 生成action的代码
        #这里主要是负责数据的处理， 以及调用vla的生成函数
        #temperature = self.args.vla_temperature_eval if deterministic else self.args.vla_temperature
        #do_sample = (temperature != 0.0)
        #features = self._preprocess_obs(x)
        image = x["image"][0].cpu().numpy() 
        image = get_preprocessed_image(image, resize_size=224)
        instruction = x["task_description"][0]
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)

        # Construct messages in the expected chat format. Note that nora expects image of size 224 by 224
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        # Apply chat template to get the text input for the model
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision information (depends on your process_vision_info function)
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs for the model using the main processor
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Move inputs to GPU
        inputs = {k: v.to(self.device_id) for k, v in inputs.items()}

        #TODO 6 Nora这里不生成values， 之后拿一个占位符 "_"， 赋值给他， 占位符不行，还是要给他赋值为0的tensor, he logprobs的形状一样就行
        #Action,返回的是生成的token id。
        #logprobs， 返回的是一个标量

        #TODO 7 Nora 这需要有自己的predict_action_batch函数， 需要返回logprobs
        #原来self.vla: OpenVLAForActionPredictionWithValueHead 里面好多类似的函数， 为了rl的各种值， 进行的改造
        #参考OpenVLAForActionPredictionWithValueHead里面的写就行。
        #values, action, logprobs = self.vla.predict_action_batch(
        #    **features,
        #    unnorm_key=self.args.vla_unnorm_key,
        #    do_sample=do_sample,
        #    temperature=temperature,
        #)
        values, action, logprobs = self.vla.predict_action_batch(**inputs)


        assert len(values.shape) == 2 and values.shape[1] == 1
        assert len(action.shape) == 2 and action.shape[0] == values.shape[0]
        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1

        return values, action, logprobs


    def evaluate_actions(self, x: dict, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        #image = x["image"][0].cpu().numpy() 
        #instruction = x["task_description"][0]
        #if not isinstance(image, PIL.Image.Image):
        #    image = PIL.Image.fromarray(image)

        # Construct messages in the expected chat format. Note that nora expects image of size 224 by 224
        # message = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {
        #                 "type": "image",
        #                 "image": image,
        #                 "resized_height": 224,
        #                 "resized_width": 224,
        #             },
        #             {"type": "text", "text": instruction},
        #         ],
        #     }
        # ] when evaluating a batch, we need to prepare multiple messages
        messages = prepare_message(x)
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )


        #here add action to the text with decoded action
        #only get the valid tokens, which are >0
        #token_len = (action[0]>0).sum().item()
        #action= action[:,:token_len]
        action_str = self.processor.tokenizer.batch_decode(action)
        
        #remove"!" and add "\n" to the end of action str.
        #! is decoded from id 0
        text = [f"{t}{a.replace('!', '')}\n"
            for t, a in zip(text, action_str)]

        # Process vision information (depends on your process_vision_info function)
        image_inputs, video_inputs = process_vision_info(messages)

        # Prepare inputs for the model using the main processor
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side="left",
            return_tensors="pt",
        )


        # or here add action to the text with  action token id?

        # Move inputs to GPU
        inputs = {k: v.to(self.device_id) for k, v in inputs.items()}


        #logprobs, entropy, values = self.vla.evaluate_action(
        #    **features,
        #    unnorm_key=self.args.vla_unnorm_key
        #)
        labels = inputs["input_ids"].clone()

        logprobs, entropy, values = self.vla.evaluate_action(**inputs, labels=labels, action=action)


        assert len(logprobs.shape) == 2 and logprobs.shape[1] == 1
        assert len(entropy.shape) == 2 and entropy.shape[1] == 1
        assert len(values.shape) == 2 and values.shape[1] == 1

        return logprobs, entropy, values

    def prep_rollout(self):
        self.vla.eval()

    def prep_training(self):
        self.vla.train()

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)

        self.vla.save_pretrained(str(path))
        training_state = {
            "vh": self.vla.value_head.state_dict(),
            "vh_optimizer": self.vh_optimizer.state_dict(),
            "vla_optimizer": self.vla_optimizer.state_dict(),
        }
        torch.save(training_state, path / "training_state.pt")

        json.dump(self.vla.base_model.norm_stats, open(path / "dataset_statistics.json", "w"))

    def load(self, path: Path):
        del self.vla
        torch.cuda.empty_cache()

        self.vla = OpenVLAForActionPredictionWithValueHead.from_pretrained(
            self.args.vla_path,
            attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="cuda:" + str(self.device_id),
            vh_mode="a0",
        )
        self.vla = PeftModel.from_pretrained(self.vla, path, is_trainable=True)
        self.vla.print_trainable_parameters()

        if self.args.vla_unnorm_key not in self.vla.base_model.norm_stats:
            ds = json.load(open(path / "dataset_statistics.json", "r"))
            self.vla.base_model.norm_stats[self.args.vla_unnorm_key] = ds[self.args.vla_unnorm_key]

        training_state_path = path / "training_state.pt"
        training_state = torch.load(training_state_path, map_location=self.tpdv["device"])

        if "vh" in training_state:
            self.vla.value_head.load_state_dict(training_state['vh'], assign=True)
        else:
            print("Warning: value_head state not found in training_state")

        self._setup_optimizer()
        self.vh_optimizer.load_state_dict(training_state['vh_optimizer'])
        self.vla_optimizer.load_state_dict(training_state['vla_optimizer'])




def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(
        img, expand_animations=False, dtype=tf.uint8
    )  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_preprocessed_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    obs = resize_image(obs, resize_size)
    return obs


def prepare_message(input_dict):
    """prepare the message for nora model"""
    batch_size = input_dict['image'].shape[0]
    messages = []
    for i in range(batch_size):
        image = input_dict["image"][i].cpu().numpy() 
        instruction = input_dict["task_description"][i]
        if not isinstance(image, PIL.Image.Image):
            image = PIL.Image.fromarray(image)
        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                        "resized_height": 224,
                        "resized_width": 224,
                    },
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        messages.append(message)
    return messages