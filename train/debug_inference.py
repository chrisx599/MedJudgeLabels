"""
调试脚本：查看模型实际输出
"""
import json
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
from prepare_data import format_prompt

def load_model(base_model_path, lora_adapter_path, max_seq_length=2048, load_in_4bit=True):
    """加载模型"""
    print(f"正在加载基础模型: {base_model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    
    if lora_adapter_path:
        print(f"正在加载LoRA适配器: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        print("✓ LoRA适配器加载完成")
    
    FastLanguageModel.for_inference(model)
    print("✓ 模型加载完成")
    return model, tokenizer

def test_inference(model, tokenizer, query, response):
    """测试推理"""
    print("\n" + "=" * 80)
    print("测试推理")
    print("=" * 80)
    print(f"Query: {query[:100]}...")
    print(f"Response: {response[:100]}...")
    
    # 格式化输入
    prompt = format_prompt(query, response)
    
    print("\n完整Prompt:")
    print("-" * 80)
    print(prompt)
    print("-" * 80)
    
    # Tokenize
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    print(f"\nInput tokens: {inputs['input_ids'].shape[1]}")
    
    # 生成
    print("\n生成中...")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.9,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n完整输出:")
    print("=" * 80)
    print(full_output)
    print("=" * 80)
    
    # 尝试提取assistant部分 - 支持多种格式
    assistant_output = None
    
    # 方法1: 查找带标记的assistant部分
    assistant_start = full_output.find("<|im_start|>assistant\n")
    if assistant_start != -1:
        assistant_output = full_output[assistant_start + len("<|im_start|>assistant\n"):]
        assistant_end = assistant_output.find("<|im_end|>")
        if assistant_end != -1:
            assistant_output = assistant_output[:assistant_end]
        print("\n✓ 找到带标记的assistant部分")
    
    # 方法2: 查找纯文本assistant部分
    if assistant_output is None:
        assistant_marker = "\nassistant\n"
        assistant_start = full_output.find(assistant_marker)
        if assistant_start != -1:
            assistant_output = full_output[assistant_start + len(assistant_marker):]
            print("\n✓ 找到纯文本assistant部分 (方法2)")
        else:
            # 方法3: 尝试查找最后一个"assistant"后的内容
            assistant_marker_alt = "assistant\n"
            assistant_start = full_output.rfind(assistant_marker_alt)
            if assistant_start != -1:
                assistant_output = full_output[assistant_start + len(assistant_marker_alt):]
                print("\n✓ 找到纯文本assistant部分 (方法3)")
            else:
                print("\n✗ 未找到assistant标记，使用完整输出")
                assistant_output = full_output
    
    if assistant_output:
        assistant_output = assistant_output.strip()
        
        print("\nAssistant输出:")
        print("-" * 80)
        print(assistant_output)
        print("-" * 80)
        
        # 尝试解析JSON
        try:
            result = json.loads(assistant_output)
            print("\n✓ JSON解析成功:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
        except json.JSONDecodeError as e:
            print(f"\n✗ JSON解析失败: {e}")
            print(f"尝试解析的文本: {repr(assistant_output[:200])}")
            
            # 尝试提取JSON部分
            import re
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', assistant_output)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    print("\n✓ 通过正则提取JSON成功:")
                    print(json.dumps(result, ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    print("\n✗ 正则提取的JSON也无法解析")

def main():
    """主函数"""
    # 配置
    BASE_MODEL_PATH = "/common/home/projectgrps/CS707/CS707G2/scratchDirectory/Qwen3-8B"
    LORA_ADAPTER_PATH = "./outputs/lora_adapter"
    TEST_DATA_PATH = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_test.jsonl"
    
    # 加载模型
    model, tokenizer = load_model(BASE_MODEL_PATH, LORA_ADAPTER_PATH)
    
    # 加载测试数据
    print(f"\n加载测试数据: {TEST_DATA_PATH}")
    test_data = []
    with open(TEST_DATA_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 3:  # 只加载前3个样本
                break
            test_data.append(json.loads(line.strip()))
    
    print(f"加载了 {len(test_data)} 个测试样本")
    
    # 测试每个样本
    for i, sample in enumerate(test_data):
        print(f"\n\n{'#' * 80}")
        print(f"样本 {i+1}/{len(test_data)}")
        print(f"{'#' * 80}")
        
        query = sample['query']
        response = sample['response']
        true_label = sample['binary_harmfulness']
        
        print(f"真实标签: {true_label}")
        
        test_inference(model, tokenizer, query, response)
        
        input("\n按Enter继续下一个样本...")

if __name__ == "__main__":
    main()