"""
环境测试脚本：验证train_orig文件夹的设置是否正确
"""
import sys
import json
from pathlib import Path

def test_imports():
    """测试必要的包是否已安装"""
    print("=" * 80)
    print("[1/4] 检查Python依赖...")
    print("=" * 80)
    
    packages = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'datasets': 'Datasets',
        'peft': 'PEFT',
        'trl': 'TRL',
    }
    
    missing = []
    for package, name in packages.items():
        try:
            __import__(package)
            print(f"✓ {name} ({package})")
        except ImportError:
            print(f"✗ {name} ({package}) - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n警告: 缺少以下包: {', '.join(missing)}")
        print("请运行: pip install -r requirements.txt")
        return False
    
    print("✓ 所有包已安装\n")
    return True

def test_data_files():
    """测试数据文件是否存在"""
    print("=" * 80)
    print("[2/4] 检查数据文件...")
    print("=" * 80)
    
    data_files = {
        'train': '/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_train_orig.jsonl',
        'test': '/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_test_orig.jsonl',
    }
    
    all_exist = True
    for key, path in data_files.items():
        if Path(path).exists():
            size_mb = Path(path).stat().st_size / (1024 * 1024)
            print(f"✓ {key}: {path} ({size_mb:.1f}MB)")
        else:
            print(f"✗ {key}: {path} - NOT FOUND")
            all_exist = False
    
    if not all_exist:
        print("\n错误: 某些数据文件不存在")
        return False
    
    print("✓ 所有数据文件都存在\n")
    return True

def test_data_structure():
    """测试数据文件格式"""
    print("=" * 80)
    print("[3/4] 检查数据格式...")
    print("=" * 80)
    
    train_file = '/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_train_orig.jsonl'
    
    try:
        with open(train_file, 'r', encoding='utf-8') as f:
            # 读取第一条记录
            line = f.readline()
            record = json.loads(line)
            
            required_fields = ['query', 'response', 'is_response_safe', 'response_severity_level']
            missing = [f for f in required_fields if f not in record]
            
            if missing:
                print(f"✗ 缺少字段: {missing}")
                return False
            
            print(f"✓ 数据字段完整:")
            for field in required_fields:
                value = record[field]
                if isinstance(value, str):
                    print(f"  - {field}: {value[:50]}..." if len(value) > 50 else f"  - {field}: {value}")
                else:
                    print(f"  - {field}: {value}")
            
            # 检查标签有效性
            if not isinstance(record['is_response_safe'], bool):
                print(f"✗ is_response_safe 应该是布尔值，但得到 {type(record['is_response_safe'])}")
                return False
            
            if not isinstance(record['response_severity_level'], int) or record['response_severity_level'] not in [0, 1, 2, 3]:
                print(f"✗ response_severity_level 应该是 0-3，但得到 {record['response_severity_level']}")
                return False
            
            print(f"\n✓ 数据格式正确\n")
            return True
            
    except Exception as e:
        print(f"✗ 数据格式检查失败: {e}")
        return False

def test_model_path():
    """测试模型路径"""
    print("=" * 80)
    print("[4/4] 检查模型路径...")
    print("=" * 80)
    
    model_path = '/common/home/projectgrps/CS707/CS707G2/scratchDirectory/Qwen3-8B'
    
    if Path(model_path).exists():
        config_file = Path(model_path) / 'config.json'
        if config_file.exists():
            print(f"✓ 模型目录存在: {model_path}")
            print(f"✓ config.json 存在")
            return True
        else:
            print(f"✗ config.json 不存在在: {model_path}")
            return False
    else:
        print(f"✗ 模型目录不存在: {model_path}")
        return False

def main():
    print("\n" + "=" * 80)
    print("train_orig 环境检查")
    print("=" * 80 + "\n")
    
    results = {
        '依赖检查': test_imports(),
        '数据文件': test_data_files(),
        '数据格式': test_data_structure(),
        '模型路径': test_model_path(),
    }
    
    print("\n" + "=" * 80)
    print("检查总结")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
    
    if all(results.values()):
        print("\n✓ 所有检查通过！可以开始训练")
        return 0
    else:
        print("\n✗ 某些检查失败，请解决上述问题")
        return 1

if __name__ == "__main__":
    sys.exit(main())
