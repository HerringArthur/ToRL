import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

# 检查是否安装了 bitsandbytes 用于 4-bit 量化
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    print("Warning: bitsandbytes not installed. 4-bit quantization will be disabled.")

class TrainTicketPolicyModel(nn.Module):
    """
    基于 Qwen2.5 的策略模型。
    使用 LoRA 进行参数高效微调 (PEFT)。
    支持 4-bit 量化加载以节省显存。
    """
    def __init__(self, cfg, device_map="auto"):
        super().__init__()
        self.cfg = cfg  # 保存传入的配置对象
        self.model_path = cfg.model_path
        self.device = cfg.device
        
        # =================================================================
        # 1. 加载 Tokenizer
        # =================================================================
        print(f"[Model] Loading tokenizer from {self.model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            padding_side="left" # 生成任务必须左填充
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # =================================================================
        # 2. 模型量化配置
        # =================================================================
        # 如果显存有限，尝试开启 4-bit 量化
        quantization_config = None
        if HAS_BNB and self.device == "cuda":
            # 这里简单判断：如果不是全精度模式或者显式要求量化，可以开启
            # 在本例中，我们默认根据是否有 BNB 库来决定是否尝试量化加载
            # 你也可以在 cfg 中增加一个 use_4bit 字段来控制
            print("[Model] Enabling 4-bit NF4 quantization...")
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if cfg.use_amp else torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )

        # =================================================================
        # 3. 加载基础模型 (Base Model)
        # =================================================================
        print(f"[Model] Loading base model...")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if cfg.use_amp else torch.float16,
            device_map=device_map
        )
        
        # 如果使用了量化，需要预处理模型以支持训练
        if quantization_config:
            self.base_model = prepare_model_for_kbit_training(self.base_model)

        # =================================================================
        # 4. 配置 LoRA (PEFT)
        # =================================================================
        if cfg.use_lora:
            print(f"[Model] Applying LoRA (r={cfg.lora_r}, alpha={cfg.lora_alpha})...")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=cfg.lora_r,
                lora_alpha=cfg.lora_alpha,
                lora_dropout=cfg.lora_dropout,
                target_modules=cfg.target_modules,
                bias="none"
            )
            self.model = get_peft_model(self.base_model, peft_config)
            self.model.print_trainable_parameters()
        else:
            self.model = self.base_model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """
        标准 Causal LM 前向传播
        """
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )

    def generate(self, input_ids, attention_mask=None, **kwargs):
        """
        生成函数，用于 Rollout 阶段
        """
        # 设置默认生成参数，优先使用传入的 kwargs，其次使用 cfg 中的配置
        gen_kwargs = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "do_sample": True,
            "temperature": self.cfg.temperature,
            "top_k": self.cfg.top_k,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)
        
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **gen_kwargs
        )

    def save_pretrained(self, save_directory):
        """
        保存模型（只保存 LoRA 权重，节省空间）
        """
        self.model.save_pretrained(save_directory)
        self.tokenizer.save_pretrained(save_directory)

    def get_device(self):
        return self.base_model.device