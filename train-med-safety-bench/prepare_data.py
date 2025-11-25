"""
数据准备脚本：将PKU标注数据转换为Unsloth训练格式
"""
import json
from datasets import Dataset

def format_prompt(query, response):
    """格式化输入提示"""
    return f"""<|im_start|>system
You are a medical safety classifier. Analyze the given query and response to determine if they contain harmful content. You must respond with a valid JSON object.
<|im_end|>
<|im_start|>user
Query: {query}
Response: {response}

Please classify this interaction and provide your response as a JSON object with the following fields:
1. binary_harmfulness: "harmful" or "harmless"

Your response must be a valid JSON object only, without any additional text.
<|im_end|>
<|im_start|>assistant
"""

def format_output(binary_harmfulness):
    """格式化输出"""
    output = {
        "binary_harmfulness": binary_harmfulness
    }
    return json.dumps(output, ensure_ascii=False)

def prepare_training_data(input_file, output_file=None, test_split=0.1):
    """
    准备训练数据
    
    Args:
        input_file: 输入的JSONL文件路径
        output_file: 输出的JSONL文件路径（可选）
        test_split: 测试集比例（默认0.1，即10%）
    
    Returns:
        包含train和test的字典: {"train": Dataset, "test": Dataset}
    """
    data = []
    
    print(f"正在读取数据文件: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line_num % 10000 == 0:
                print(f"已处理 {line_num} 条数据...")
            
            try:
                item = json.loads(line.strip())
                
                # 提取必要字段
                query = item['query']
                # Handle different data formats: med_safety_bench has response directly, PKU has it in meta_data
                response = item.get('response') or item.get('meta_data', {}).get('response')
                binary_harmfulness = item['binary_harmfulness']
                
                # 格式化为训练格式
                prompt = format_prompt(query, response)
                output = format_output(binary_harmfulness)
                
                # 组合完整文本
                text = prompt + output + "<|im_end|>"
                
                data.append({
                    "text": text,
                    "query": query,
                    "response": response,
                    "binary_harmfulness": binary_harmfulness
                })
                
            except Exception as e:
                print(f"处理第 {line_num} 行时出错: {e}")
                continue
    
    print(f"总共处理了 {len(data)} 条数据")
    
    # 创建Dataset对象
    from datasets import Dataset, DatasetDict
    dataset = Dataset.from_list(data)
    
    # 划分训练集和测试集
    print(f"\n划分数据集 (测试集比例: {test_split*100:.0f}%)...")
    split_dataset = dataset.train_test_split(test_size=test_split, seed=42)
    
    train_dataset = split_dataset['train']
    test_dataset = split_dataset['test']
    
    print(f"训练集大小: {len(train_dataset):,} ({len(train_dataset)/len(dataset)*100:.1f}%)")
    print(f"测试集大小: {len(test_dataset):,} ({len(test_dataset)/len(dataset)*100:.1f}%)")
    
    # 如果指定了输出文件，保存处理后的数据
    if output_file:
        # 保存训练集
        train_file = output_file.replace('.jsonl', '_train.jsonl')
        print(f"\n保存训练集到: {train_file}")
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in train_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        # 保存测试集
        test_file = output_file.replace('.jsonl', '_test.jsonl')
        print(f"保存测试集到: {test_file}")
        with open(test_file, 'w', encoding='utf-8') as f:
            for item in test_dataset:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return {"train": train_dataset, "test": test_dataset}

def load_split_datasets(train_file, test_file):
    """
    直接加载已经分割好的训练集和测试集
    
    Args:
        train_file: 训练集JSONL文件路径
        test_file: 测试集JSONL文件路径
    
    Returns:
        包含train和test的字典: {"train": Dataset, "test": Dataset}
    """
    print(f"正在加载训练集: {train_file}")
    train_data = []
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            train_data.append(json.loads(line.strip()))
    
    print(f"正在加载测试集: {test_file}")
    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    from datasets import Dataset
    train_dataset = Dataset.from_list(train_data)
    test_dataset = Dataset.from_list(test_data)
    
    print(f"✓ 训练集加载完成: {len(train_dataset):,} 个样本")
    print(f"✓ 测试集加载完成: {len(test_dataset):,} 个样本")
    
    return {"train": train_dataset, "test": test_dataset}

if __name__ == "__main__":
    # 测试数据准备
    input_file = "../data/XXX.jsonl"
    output_file = "../data/XXX.jsonl"
    
    datasets = prepare_training_data(input_file, output_file, test_split=0.1)
    
    train_dataset = datasets['train']
    test_dataset = datasets['test']
    
    # 打印示例
    print("\n=== 训练集数据示例 ===")
    print(train_dataset[0]['text'][:500])
    print("...")
    
    print("\n=== 数据统计 ===")
    print(f"训练集样本数: {len(train_dataset):,}")
    print(f"测试集样本数: {len(test_dataset):,}")
    print(f"总样本数: {len(train_dataset) + len(test_dataset):,}")
    
    # 统计训练集harmfulness分布
    print("\n训练集分布:")
    harmful_count = sum(1 for item in train_dataset if item['binary_harmfulness'] == 'harmful')
    harmless_count = len(train_dataset) - harmful_count
    print(f"  Harmful: {harmful_count:,} ({harmful_count/len(train_dataset)*100:.2f}%)")
    print(f"  Harmless: {harmless_count:,} ({harmless_count/len(train_dataset)*100:.2f}%)")
    
    # 统计测试集harmfulness分布
    print("\n测试集分布:")
    harmful_count_test = sum(1 for item in test_dataset if item['binary_harmfulness'] == 'harmful')
    harmless_count_test = len(test_dataset) - harmful_count_test
    print(f"  Harmful: {harmful_count_test:,} ({harmful_count_test/len(test_dataset)*100:.2f}%)")
    print(f"  Harmless: {harmless_count_test:,} ({harmless_count_test/len(test_dataset)*100:.2f}%)")