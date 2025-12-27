import os
import sys
from pathlib import Path

# 添加当前目录到 sys.path 确保能导入 config
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from config import TORLConfig
except ImportError:
    print("❌ 无法导入 config.py，请确保该文件在当前目录下。")
    sys.exit(1)

def check_path(name: str, path_str: str, check_type: str = "exists"):
    """
    检查路径状态并打印结果
    :param name: 路径描述名称
    :param path_str: 路径字符串
    :param check_type: 检查类型 "exists" (存在), "dir" (目录), "file" (文件)
    """
    # 处理可能的 None 或空字符串
    if not path_str:
        print(f"[⚠️ 未配置] {name}")
        print(f"  路径: (空)")
        print("-" * 40)
        return

    path = Path(path_str)
    status = "✅ 正常"
    details = []
    
    # 转换为绝对路径以便查看（如果是相对路径）
    abs_path = path.absolute()

    if not path.exists():
        status = "❌ 不存在"
        details.append("路径未找到")
    else:
        if check_type == "dir" and not path.is_dir():
            status = "⚠️ 类型错误"
            details.append("应为目录但发现是文件")
        elif check_type == "file" and not path.is_file():
            status = "⚠️ 类型错误"
            details.append("应为文件但发现是目录")
        
        # 尝试读取权限检查（简单版）
        try:
            if path.is_dir():
                # 尝试列出目录
                os.listdir(path)
            elif path.is_file():
                # 尝试打开文件
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    pass
        except PermissionError:
            status = "❌ 权限拒绝"
            details.append("无读取权限")
        except Exception as e:
            status = "⚠️ 读取错误"
            details.append(str(e))

    print(f"[{status}] {name}")
    print(f"  配置路径: {path_str}")
    print(f"  绝对路径: {abs_path}")
    if details:
        print(f"  详情: {', '.join(details)}")
    print("-" * 40)

def main():
    print("正在检查配置路径...\n")
    print("-" * 40)
    
    try:
        config = TORLConfig()
    except Exception as e:
        print(f"❌ 初始化配置失败: {e}")
        return

    # 检查模型路径
    check_path("模型路径 (model_path)", config.model_path, "dir")

    # 检查数据集路径
    check_path("数据集路径 (dataset_path)", config.dataset_path, "file")

    # 检查输出目录
    check_path("输出目录 (output_dir)", config.output_dir, "dir")

    # 额外提示：如果是在 Windows 上运行且路径看起来像 Linux 路径
    if os.name == 'nt':
        linux_paths = [p for p in [config.model_path, config.dataset_path] if p and str(p).startswith('/')]
        if linux_paths:
            print("\n⚠️  检测到当前在 Windows 环境运行，但配置文件包含 Linux 风格的绝对路径。")
            print("   这可能会导致路径检查失败。请考虑在 config.py 中修改路径或使用相对路径。")

if __name__ == "__main__":
    main()
