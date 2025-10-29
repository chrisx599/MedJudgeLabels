#!/usr/bin/env python3
"""
vLLM Batch Inference Script for Medical Safety Adjudication
使用Qwen3-Guard模型进行医疗安全性评估的批量推理脚本
评估聊天机器人响应的安全性，包括有害性、严重程度和危害类型
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def load_prompt_template(prompt_file: str) -> str:
    """加载prompt模板文件"""
    with open(prompt_file, 'r', encoding='utf-8') as f:
        return f.read()


def prepare_prompts(queries: List[dict], prompt_template: str) -> List[str]:
    """准备批量推理的prompts"""
    prompts = []
    for query in queries:
        question = query["query"]
        response = query["response"]
        # 将查询添加到prompt模板中
        full_prompt = f"{prompt_template}\n\nHuman Query:\n{question}\nChatbot Response:{response}"
        prompts.append(full_prompt)
    return prompts


def load_queries_from_file(input_file: str) -> List[str]:
    """从文件加载查询数据
    支持的格式:
    - 纯文本文件 (每行一个查询)
    - JSON文件 (包含查询列表)
    - JSONL文件 (每行一个JSON对象)
    """
    file_path = Path(input_file)
    
    if file_path.suffix == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'queries' in data:
                return data['queries']
            else:
                raise ValueError("JSON文件格式不正确，应该是列表或包含'queries'键的字典")
    
    elif file_path.suffix == '.jsonl':
        queries = []
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
            # queries = [item['prompt'] for item in data]
        return data
    
    else:  # 纯文本文件
        with open(file_path, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        return queries


def save_results(results: List[Dict[str, Any]], output_file: str):
    """保存分类结果到文件（完整保存）"""
    file_path = Path(output_file)
    
    if file_path.suffix == '.jsonl':
        with open(file_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    else:  # 默认保存为JSON
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"结果已保存到: {output_file}")


def append_results(results: List[Dict[str, Any]], output_file: str):
    """追加保存结果到JSONL文件（用于增量保存）"""
    file_path = Path(output_file)
    
    with open(file_path, 'a', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def run_batch_inference(
    model_name,
    queries,
    prompt_template,
    output_file,
    batch_size=1,
    max_tokens=512,
    **kwargs
):
    print("正在加载模型和tokenizer...")
    # 初始化tokenizer和模型
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        use_fast=False
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        config_overrides={"model_type": "baichuan"},
        torch_dtype="auto"
    )
    
    print("模型加载完成，开始处理...")
    prompts = prepare_prompts(queries, prompt_template)
    results = []
    
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        try:
            # 直接使用文本输入
            model_inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(model.device)
            
            # 简化生成参数
            with torch.inference_mode():
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 处理生成的文本
            generated_texts = tokenizer.batch_decode(outputs[:, model_inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 处理结果
            for query, response in zip(queries[i:i + batch_size], generated_texts):
                try:
                    result = json.loads(response)
                except json.JSONDecodeError:
                    result = {"error": "Invalid JSON response", "raw_response": response}
                
                full_result = {
                    "query": query["query"],
                    "response": query["response"],
                    **result
                }
                results.append(full_result)
                
                if output_file and output_file.endswith('.jsonl'):
                    append_results([full_result], output_file)
                    
        except Exception as e:
            print(f"批处理出错: {str(e)}")
            continue
            
    return results


def main():
    parser = argparse.ArgumentParser(
        description="使用vLLM和Qwen3-Guard进行医疗安全性评估的批量推理"
    )
    
    # 必需参数
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入文件路径 (支持.txt, .json, .jsonl格式)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出文件路径 (推荐使用.json或.jsonl格式)"
    )
    
    # 可选参数
    parser.add_argument(
        "--model",
        type=str,
        default="/common/home/projectgrps/CS707/CS707G2/scratchDirectory/Baichuan-M1-14B-Instruct",
        help="模型名称或路径 (默认: Qwen/Qwen2.5-3B-Instruct)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="annotation/anno_prompt.txt",
        help="Prompt模板文件路径 (默认: annotation/anno_prompt.txt)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="采样温度 (默认: 0.0, 确定性输出)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="最大生成token数 (默认: 100)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="张量并行大小 (默认: 1)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU内存利用率 (默认: 0.9)"
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=256,
        help="vLLM调度器可以同时处理的最大序列数，即batch size (默认: 256)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="手动分批处理的批次大小，用于处理超大数据集 (默认: None，一次处理所有数据)"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="最大模型序列长度（prompt + output），用于控制KV cache内存 (默认: 8192)"
    )
    
    args = parser.parse_args()
    
    # 加载prompt模板
    print(f"加载prompt模板: {args.prompt}")
    prompt_template = load_prompt_template(args.prompt)
    
    # 加载查询数据
    print(f"加载查询数据: {args.input}")
    queries = load_queries_from_file(args.input)
    print(f"共加载 {len(queries)} 个查询")
    
    # 运行批量推理（JSONL格式会在推理过程中增量保存）
    results = run_batch_inference(
        model_name=args.model,
        queries=queries,
        prompt_template=prompt_template,
        output_file=args.output,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
    )
    
    # 如果不是JSONL格式，在最后保存所有结果
    output_path = Path(args.output)
    if output_path.suffix != '.jsonl':
        save_results(results, args.output)
    else:
        print(f"\n所有结果已增量保存到: {args.output}")
    
    # 打印统计信息
    harmless_count = sum(1 for r in results if r['binary_harmfulness'] == 'harmless')
    harmful_count = sum(1 for r in results if r['binary_harmfulness'] == 'harmful')
    unknown_count = sum(1 for r in results if r['binary_harmfulness'] == 'unknown')
    
    # 统计严重程度
    mild_count = sum(1 for r in results if r.get('severity') == 'mild')
    moderate_count = sum(1 for r in results if r.get('severity') == 'moderate')
    severe_count = sum(1 for r in results if r.get('severity') == 'severe')
    
    print("\n=== 安全性评估统计 ===")
    print(f"总查询数: {len(results)}")
    print(f"无害 (harmless): {harmless_count} ({harmless_count/len(results)*100:.1f}%)")
    print(f"有害 (harmful): {harmful_count} ({harmful_count/len(results)*100:.1f}%)")
    if unknown_count > 0:
        print(f"未知 (unknown): {unknown_count} ({unknown_count/len(results)*100:.1f}%)")
    
    if harmful_count > 0:
        print(f"\n=== 有害响应严重程度分布 ===")
        print(f"轻度 (mild): {mild_count} ({mild_count/harmful_count*100:.1f}%)")
        print(f"中度 (moderate): {moderate_count} ({moderate_count/harmful_count*100:.1f}%)")
        print(f"严重 (severe): {severe_count} ({severe_count/harmful_count*100:.1f}%)")
    
    print("\n推理完成！")


if __name__ == "__main__":
    main()