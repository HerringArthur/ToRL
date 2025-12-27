# qwen_tool_policy.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

try:
    from transformers import BitsAndBytesConfig
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("Warning: bitsandbytes not available, 4-bit quantization disabled")


@dataclass
class DistPack:
    logits: torch.Tensor
    log_probs: torch.Tensor
    probs: torch.Tensor


def build_dist_pack(logits: torch.Tensor) -> DistPack:
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return DistPack(logits=logits, log_probs=log_probs, probs=probs)


def kl_divergence(p: DistPack, q: DistPack) -> torch.Tensor:
    """KL(p || q)"""
    return (p.probs * (p.log_probs - q.log_probs)).sum(dim=-1)


def entropy(d: DistPack) -> torch.Tensor:
    return -(d.probs * d.log_probs).sum(dim=-1)


class QwenToolPolicy(nn.Module):
    def __init__(
        self,
        model_path: str,
        num_tools: int,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: List[str] = None,
        device: Optional[torch.device] = None,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.num_tools = num_tools
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.load_in_8bit = load_in_8bit
        
        # 1. Config
        try:
            config_obj = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        except Exception as e:
            print(f"Warning: AutoConfig load failed ({e}), attempting fallback...")
            config_obj = None

        # 2. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            config=config_obj,
            trust_remote_code=True,
            use_fast=False, 
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if load_in_4bit and HAS_BNB:
            print(f"Loading base model from {model_path} in 4-bit NF4...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,  # 双重量化进一步节省内存
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map={"": device} if device else "auto",
                torch_dtype=torch.bfloat16,
            )
            # 为 k-bit 训练准备模型
            base_model = prepare_model_for_kbit_training(base_model)
            
        elif load_in_8bit and HAS_BNB:
            print(f"Loading base model from {model_path} in 8-bit...")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=quantization_config,
                trust_remote_code=True,
                device_map={"": device} if device else "auto",
            )
            base_model = prepare_model_for_kbit_training(base_model)
            
        else:
            print(f"Loading base model from {model_path} in bfloat16...")
            base_model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
            )
            if device and not (load_in_4bit or load_in_8bit):
                base_model = base_model.to(device)

        # 4. LoRA Configuration
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )

        self.model = get_peft_model(base_model, lora_config)
        
        # 5. Classifier Head
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_tools)
        
        # 确保 classifier 在正确的设备和数据类型
        if device:
            self.classifier = self.classifier.to(device)
        self.classifier = self.classifier.to(torch.bfloat16)
        
        self._device = device

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def encode_states(self, texts: List[str], device: torch.device):
        """编码状态文本，带有长度限制"""
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        return input_ids, attention_mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        last_hidden = outputs.hidden_states[-1][:, -1, :]

        last_hidden = last_hidden.to(self.classifier.weight.dtype)
        
        logits = self.classifier(last_hidden)
        return logits
    
    def get_device(self):
        return next(self.parameters()).device