"""
标注脚本：使用微调的LoRA适配器对PKU数据进行标注
基于train/inference.py的推理能力和anno_prompt.txt的标注规则

功能:
- 加载微调的LoRA适配器 (train/outputs/lora_adapter)
- 对pku_anno_formatted_test.jsonl进行推理
- 输出标注结果到ft-pku_pku_anno_formatted_test/目录
"""
import json
import torch
from unsloth import FastLanguageModel
from peft import PeftModel
import argparse
import os
from tqdm import tqdm
from datetime import datetime
import re

def format_anno_prompt(query, response, prompt_template_path):
    """
    根据anno_prompt.txt模板格式化标注提示
    
    Args:
        query: 用户查询
        response: 模型响应
        prompt_template_path: 标注提示模板文件路径
    
    Returns:
        格式化后的提示（包含Qwen3聊天模板）
    """
    # 读取提示模板
    with open(prompt_template_path, 'r', encoding='utf-8') as f:
        template = f.read()
    
    # 替换模板中的占位符
    prompt = template.replace('{{ query }}', query).replace('{{ response }}', response)
    
    # 使用Qwen3的聊天格式
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    
    return formatted_prompt

def load_model(base_model_path, lora_adapter_path=None, max_seq_length=2048, load_in_4bit=True):
    """
    加载模型（支持原始模型 + LoRA适配器）
    
    Args:
        base_model_path: 原始模型路径
        lora_adapter_path: LoRA适配器路径
        max_seq_length: 最大序列长度
        load_in_4bit: 是否使用4-bit量化
    
    Returns:
        model, tokenizer
    """
    print(f"正在加载基础模型: {base_model_path}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    
    # 如果提供了LoRA适配器路径，加载适配器
    if lora_adapter_path:
        print(f"正在加载LoRA适配器: {lora_adapter_path}")
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        print("✓ LoRA适配器加载完成")
    
    # 启用推理模式
    FastLanguageModel.for_inference(model)
    
    print("✓ 模型加载完成")
    return model, tokenizer

def parse_annotation_output(output_text):
    """
    解析模型标注输出，提取JSON结果
    
    Args:
        output_text: 模型生成的完整文本
    
    Returns:
        解析后的字典，如果解析失败返回None
    """
    # 提取assistant的响应
    assistant_output = None
    
    # 方法1: 查找带标记的assistant部分
    assistant_start = output_text.find("<|im_start|>assistant\n")
    if assistant_start != -1:
        assistant_output = output_text[assistant_start + len("<|im_start|>assistant\n"):]
        assistant_end = assistant_output.find("<|im_end|>")
        if assistant_end != -1:
            assistant_output = assistant_output[:assistant_end]
    
    # 方法2: 查找纯文本assistant部分
    if assistant_output is None:
        assistant_marker = "\nassistant\n"
        assistant_start = output_text.find(assistant_marker)
        if assistant_start != -1:
            assistant_output = output_text[assistant_start + len(assistant_marker):]
        else:
            assistant_marker_alt = "assistant\n"
            assistant_start = output_text.rfind(assistant_marker_alt)
            if assistant_start != -1:
                assistant_output = output_text[assistant_start + len(assistant_marker_alt):]
            else:
                assistant_output = output_text
    
    # 清理输出
    assistant_output = assistant_output.strip()
    
    # 清理markdown代码块标记
    if assistant_output.startswith("```json"):
        assistant_output = assistant_output[7:]
    if assistant_output.startswith("```"):
        assistant_output = assistant_output[3:]
    if assistant_output.endswith("```"):
        assistant_output = assistant_output[:-3]
    
    assistant_output = assistant_output.strip()
    
    # 尝试解析JSON - 首先尝试整个文本
    try:
        result = json.loads(assistant_output)
        return result
    except json.JSONDecodeError:
        pass
    
    # 尝试提取JSON部分（可能有额外文本前后）
    # 寻找第一个 { 和最后一个 }
    first_brace = assistant_output.find('{')
    last_brace = assistant_output.rfind('}')
    
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        json_str = assistant_output[first_brace:last_brace+1]
        try:
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError:
            pass
    
    # 最后尝试正则表达式提取嵌套JSON
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', assistant_output)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            return result
        except json.JSONDecodeError:
            return None
    
    return None

def annotate_single(model, tokenizer, query, response, anno_prompt_path, 
                   max_new_tokens=256, temperature=0.3):
    """
    对单个样本进行标注
    
    Args:
        model: 模型
        tokenizer: tokenizer
        query: 用户查询
        response: 模型响应
        anno_prompt_path: 标注提示模板路径
        max_new_tokens: 最大生成token数
        temperature: 温度参数
    
    Returns:
        标注结果字典
    """
    # 格式化提示
    prompt = format_anno_prompt(query, response, anno_prompt_path)
    
    # Tokenize
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 解析标注结果
    result = parse_annotation_output(full_output)
    
    return result

def batch_annotate(model, tokenizer, data_list, anno_prompt_path, 
                  output_jsonl, output_csv, max_new_tokens=256, temperature=0.3):
    """
    批量标注数据
    
    Args:
        model: 模型
        tokenizer: tokenizer
        data_list: 数据列表
        anno_prompt_path: 标注提示模板路径
        output_jsonl: 输出JSONL文件路径
        output_csv: 输出CSV文件路径
        max_new_tokens: 最大生成token数
        temperature: 温度参数
    
    Returns:
        统计信息字典
    """
    results = []
    stats = {
        'total': len(data_list),
        'successful': 0,
        'failed': 0,
        'harmless': 0,
        'harmful': 0,
        'severity_dist': {'mild': 0, 'moderate': 0, 'severe': 0}
    }
    
    print(f"\n开始标注 {len(data_list)} 条数据...")
    print("=" * 80)
    
    # 保存JSONL文件
    with open(output_jsonl, 'w', encoding='utf-8') as jsonl_f:
        # 保存CSV文件头
        csv_lines = []
        csv_header = "query,response,binary_harmfulness,severity,harm_types,explanation\n"
        csv_lines.append(csv_header)
        
        # 处理每条数据
        for i, item in enumerate(tqdm(data_list, desc="标注进度")):
            query = item.get('query', '')
            response = item.get('response', '')
            
            # 执行标注
            annotation = annotate_single(model, tokenizer, query, response, anno_prompt_path,
                                        max_new_tokens, temperature)
            
            # 准备输出记录
            output_record = {
                'query': query,
                'response': response,
                'annotation': annotation,
                'annotation_success': annotation is not None
            }
            
            if annotation:
                stats['successful'] += 1
                
                # 统计标注结果
                binary_harm = annotation.get('binary_harmfulness', '')
                if binary_harm == 'harmless':
                    stats['harmless'] += 1
                elif binary_harm == 'harmful':
                    stats['harmful'] += 1
                
                severity = annotation.get('severity')
                if severity in stats['severity_dist']:
                    stats['severity_dist'][severity] += 1
                
                # 准备CSV行
                harm_types_str = ','.join(annotation.get('harm_types', []))
                csv_line = f'"{query}","{response}",{binary_harm},{severity if severity else "null"},"{harm_types_str}","{annotation.get("explanation", "")}"\n'
                csv_lines.append(csv_line)
            else:
                stats['failed'] += 1
            
            # 保存JSONL
            jsonl_f.write(json.dumps(output_record, ensure_ascii=False) + '\n')
            results.append(output_record)
    
    # 保存CSV
    with open(output_csv, 'w', encoding='utf-8') as csv_f:
        csv_f.writelines(csv_lines)
    
    return results, stats

def print_statistics(stats):
    """打印标注统计信息"""
    print("\n" + "=" * 80)
    print("标注统计")
    print("=" * 80)
    
    print(f"\n总样本数:          {stats['total']}")
    print(f"成功标注:          {stats['successful']} ({stats['successful']/stats['total']*100:.1f}%)")
    print(f"失败标注:          {stats['failed']} ({stats['failed']/stats['total']*100:.1f}%)")
    
    if stats['successful'] > 0:
        print(f"\n标注分布:")
        print(f"  Harmless:        {stats['harmless']} ({stats['harmless']/stats['successful']*100:.1f}%)")
        print(f"  Harmful:         {stats['harmful']} ({stats['harmful']/stats['successful']*100:.1f}%)")
        
        print(f"\n严重程度分布:")
        for severity, count in stats['severity_dist'].items():
            if count > 0:
                print(f"  {severity}:        {count}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='使用微调模型对数据进行标注')
    
    parser.add_argument('--base_model_path', type=str,
                        default='/common/home/projectgrps/CS707/CS707G2/scratchDirectory/Qwen3-8B',
                        help='基础模型路径')
    parser.add_argument('--lora_adapter_path', type=str,
                        default='/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/train-med-safety-bench/outputs/lora_adapter',
                        help='LoRA适配器路径')
    parser.add_argument('--test_data_path', type=str,
                        default='/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/med_safety_bench_formatted_test.jsonl',
                        help='测试数据路径')
    parser.add_argument('--anno_prompt_path', type=str,
                        default='/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/annotation/anno_prompt.txt',
                        help='标注提示模板路径')
    parser.add_argument('--output_dir', type=str,
                        default='/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/annotation/ft-pku_pku_anno_formatted_test/medsafety',
                        help='输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大样本数（默认：全部）')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='最大序列长度')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                        help='使用4-bit量化')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("微调模型标注脚本")
    print("=" * 80)
    print(f"基础模型:          {args.base_model_path}")
    print(f"LoRA适配器:        {args.lora_adapter_path}")
    print(f"测试数据:          {args.test_data_path}")
    print(f"标注提示:          {args.anno_prompt_path}")
    print(f"输出目录:          {args.output_dir}")
    print()
    
    # 1. 加载模型
    model, tokenizer = load_model(
        args.base_model_path,
        args.lora_adapter_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit
    )
    
    # 2. 加载数据
    print(f"\n正在加载测试数据: {args.test_data_path}")
    test_data = []
    with open(args.test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    print(f"✓ 加载了 {len(test_data)} 条数据")
    
    # 3. 执行标注 - 添加时间戳避免覆盖
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_jsonl = os.path.join(args.output_dir, f'ft-pku_pku_anno_formatted_test_labels_{timestamp}.jsonl')
    output_csv = os.path.join(args.output_dir, f'ft-pku_pku_anno_formatted_test_labels_{timestamp}.csv')
    
    results, stats = batch_annotate(
        model, tokenizer, test_data, args.anno_prompt_path,
        output_jsonl, output_csv
    )
    
    # 4. 打印统计信息
    print_statistics(stats)
    
    # 5. 保存统计信息
    stats_file = os.path.join(args.output_dir, 'annotation_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'base_model': args.base_model_path,
            'lora_adapter': args.lora_adapter_path,
            'test_data': args.test_data_path,
            'statistics': stats
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 标注完成！")
    print(f"  JSONL文件: {output_jsonl}")
    print(f"  CSV文件:   {output_csv}")
    print(f"  统计信息:  {stats_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
