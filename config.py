"""
ToRL 训练配置
包含所有可配置参数和默认值
"""

from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path

@dataclass
class TORLConfig:
    """ToRL 训练配置类"""
    
    # 模型配置
    model_name_or_path: str = "Qwen/Qwen2.5-1.5B-Instruct"
    model_path: str = "/seu_share2/home/fenglei/sharedata/Qwen2.5-1.5B-Instruct"
    device: str = "cuda"
    
    # 数据配置
    dataset_path: str = "/seu_share2/home/fenglei/213243847/data/grpo-aco/grpo-aco/data/rl_dataset_llm_v2.json"
    batch_size: int = 1  # 目前固定为1
    max_seq_length: int = 2048
    
    # 训练配置
    num_epochs: int = 3
    learning_rate: float = 1e-6
    warmup_steps: int = 100
    save_every: int = 50
    group_size: int = 8  # GRPO的组大小
    
    # LoRA配置
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # 生成配置
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    
    # 奖励配置
    reward_success: float = 5.0
    reward_step_correct: float = 1.0
    reward_step_wrong: float = -1.0
    reward_tool_wrong: float = -2.0
    reward_args_wrong: float = -1.0
    reward_timeout: float = -0.5
    
    # 环境配置
    max_steps: int = 10
    timeout: int = 30
    
    # 输出配置
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    log_every: int = 10
    
    # 工具配置
    tools_list: List[str] = field(default_factory=lambda: [
        "arxiv_local", "bigquery", "browser", "canvas", "code_interpreter",
        "email", "github", "google-cloud", "google-map", "google-sheet",
        "howtocook", "huggingface", "k8s", "kubectl", "local",
        "memory", "notion", "pdf-tools", "playwright", "python",
        "rag", "rapidapi", "search", "shopify", "slack",
        "teams", "weather", "wikipedia"
    ])
    
    def __post_init__(self):
        """验证配置参数"""
        if self.batch_size != 1:
            print("Warning: batch_size is currently fixed to 1")
            self.batch_size = 1
            
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
            
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
            
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def save(self, path: str):
        """保存配置到文件"""
        import json
        config_dict = self.__dict__.copy()
        # 转换Path对象为字符串
        for key, value in config_dict.items():
            if isinstance(value, Path):
                config_dict[key] = str(value)
        
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, path: str):
        """从文件加载配置"""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self):
        """转换为字典"""
        return self.__dict__.copy()

# 默认配置实例
DEFAULT_CONFIG = TORLConfig()

# 快速训练配置（用于测试）
FAST_CONFIG = TORLConfig(
    num_epochs=1,
    group_size=4,
    save_every=10,
    log_every=5,
    max_steps=5
)

# 生产配置（用于正式训练）
PRODUCTION_CONFIG = TORLConfig(
    num_epochs=10,
    group_size=16,
    learning_rate=5e-7,
    save_every=100,
    log_every=20,
    max_new_tokens=1024,
    max_steps=15
)