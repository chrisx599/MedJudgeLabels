"""
测试脚本：验证环境配置和数据准备
"""
import sys
import json

def test_imports():
    """测试必要的库是否已安装"""
    print("=" * 80)
    print("测试1: 检查依赖库")
    print("=" * 80)
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('trl', 'TRL'),
        ('peft', 'PEFT'),
        ('bitsandbytes', 'BitsAndBytes'),
    ]
    
    all_ok = True
    for package, name in required_packages:
        try:
            module = __import__(package)
            version = getattr(module, '__version__', 'unknown')
            print(f"✓ {name:20s} - 版本: {version}")
        except ImportError:
            print(f"✗ {name:20s} - 未安装")
            all_ok = False
    
    # 特别检查unsloth
    try:
        import unsloth
        version = getattr(unsloth, '__version__', 'unknown')
        print(f"✓ {'Unsloth':20s} - 版本: {version}")
    except ImportError:
        print(f"✗ {'Unsloth':20s} - 未安装")
        print("\n请运行: pip install unsloth")
        all_ok = False
    
    return all_ok

def test_cuda():
    """测试CUDA是否可用"""
    print("\n" + "=" * 80)
    print("测试2: 检查CUDA")
    print("=" * 80)
    
    try:
        import torch
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA可用: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
            print(f"GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
            return True
        else:
            print("⚠ CUDA不可用，将使用CPU训练（速度会很慢）")
            return False
    except Exception as e:
        print(f"✗ 检查CUDA时出错: {e}")
        return False

def test_model_path():
    """测试模型路径是否存在"""
    print("\n" + "=" * 80)
    print("测试3: 检查模型路径")
    print("=" * 80)
    
    import os
    model_path = "/common/home/projectgrps/CS707/CS707G2/scratchDirectory/Qwen3-8B"
    
    if os.path.exists(model_path):
        print(f"✓ 模型路径存在: {model_path}")
        
        # 列出模型文件
        files = os.listdir(model_path)
        print(f"  包含 {len(files)} 个文件/目录")
        
        # 检查关键文件
        key_files = ['config.json', 'pytorch_model.bin', 'model.safetensors']
        for key_file in key_files:
            if any(key_file in f for f in files):
                print(f"  ✓ 找到 {key_file}")
        
        return True
    else:
        print(f"✗ 模型路径不存在: {model_path}")
        print("  请确认模型路径是否正确")
        return False

def test_data_path():
    """测试数据路径是否存在"""
    print("\n" + "=" * 80)
    print("测试4: 检查数据路径")
    print("=" * 80)
    
    import os
    data_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno.jsonl"
    
    if os.path.exists(data_path):
        print(f"✓ 数据文件存在: {data_path}")
        
        # 统计数据行数
        with open(data_path, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        
        print(f"  数据行数: {line_count:,}")
        
        # 读取第一行检查格式
        with open(data_path, 'r', encoding='utf-8') as f:
            first_line = f.readline()
            try:
                data = json.loads(first_line)
                print(f"  ✓ 数据格式正确")
                print(f"  包含字段: {list(data.keys())}")
            except json.JSONDecodeError:
                print(f"  ✗ 数据格式错误")
                return False
        
        return True
    else:
        print(f"✗ 数据文件不存在: {data_path}")
        return False

def test_data_loading():
    """测试数据加载"""
    print("\n" + "=" * 80)
    print("测试5: 测试数据加载")
    print("=" * 80)
    
    try:
        from prepare_data import prepare_training_data
        
        data_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno.jsonl"
        
        print("正在加载前100条数据进行测试...")
        
        # 创建临时测试文件
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 100:
                        break
                    tmp.write(line)
            tmp_path = tmp.name
        
        # 加载数据
        dataset = prepare_training_data(tmp_path)
        
        print(f"✓ 成功加载 {len(dataset)} 条数据")
        print(f"\n数据示例:")
        print("-" * 80)
        print(dataset[0]['text'][:300] + "...")
        print("-" * 80)
        
        # 清理临时文件
        import os
        os.unlink(tmp_path)
        
        return True
    except Exception as e:
        print(f"✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("Unsloth训练环境测试")
    print("=" * 80 + "\n")
    
    results = []
    
    # 运行测试
    results.append(("依赖库", test_imports()))
    results.append(("CUDA", test_cuda()))
    results.append(("模型路径", test_model_path()))
    results.append(("数据路径", test_data_path()))
    results.append(("数据加载", test_data_loading()))
    
    # 总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ 所有测试通过！可以开始训练。")
        print("\n运行训练:")
        print("  sbatch run_train.sh")
        print("\n或直接运行:")
        print("  python train_qwen3_unsloth.py")
    else:
        print("✗ 部分测试失败，请先解决上述问题。")
    print("=" * 80)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())