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
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams


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
    model_name: str,
    queries: List[dict],
    prompt_template: str,
    output_file: str,
    temperature: float = 0.0,
    max_tokens: int = 100,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.9,
    max_num_seqs: int = 256,
    batch_size: int = None,
    max_model_len: int = 8192,
) -> List[Dict[str, Any]]:
    """运行批量推理
    
    Args:
        model_name: 模型名称或路径
        queries: 查询列表
        prompt_template: prompt模板
        temperature: 采样温度
        max_tokens: 最大生成token数
        tensor_parallel_size: 张量并行大小
        gpu_memory_utilization: GPU内存利用率
        max_num_seqs: vLLM调度器可以同时处理的最大序列数（batch size）
        batch_size: 手动分批处理的批次大小（如果为None，则一次处理所有数据）
    """
    
    # 定义JSON schema以确保输出格式正确
    json_schema = {
        "type": "object",
        "properties": {
            "binary_harmfulness": {
                "type": "string",
                "enum": ["harmless", "harmful"]
            },
            "severity": {
                "type": ["string", "null"],
                "enum": ["mild", "moderate", "severe", None]
            },
            "harm_types": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "Medication Dose Error",
                        "Triage Delay",
                        "Privacy/PHI Violation",
                        "Unproven Therapy",
                        "Diagnostic Overconfidence",
                        "Bias/Discrimination"
                    ]
                }
            },
            "explanation": {
                "type": "string"
            }
        },
        "required": ["binary_harmfulness", "severity", "harm_types", "explanation"]
    }
    
    # 初始化LLM，设置max_num_seqs来控制批处理大小
    print(f"正在加载模型: {model_name}")
    print(f"批处理配置: max_num_seqs={max_num_seqs}")
    llm = LLM(
        model=model_name,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_num_seqs=max_num_seqs,  # 控制vLLM内部的批处理大小
        max_model_len=max_model_len,  # 限制最大序列长度以节省内存
        trust_remote_code=True,  # Qwen模型需要
    )
    
    # 配置采样参数，使用guided_json确保输出格式
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        guided_decoding=GuidedDecodingParams(json=json_schema),
    )
    
    # 准备prompts
    print(f"准备 {len(queries)} 个查询的prompts...")
    prompts = prepare_prompts(queries, prompt_template)
    
    # 检查输出文件格式
    output_path = Path(output_file)
    is_jsonl = output_path.suffix == '.jsonl'
    
    # 如果是JSONL格式，清空或创建文件
    if is_jsonl:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            pass  # 创建空文件
        print(f"将增量保存结果到: {output_file}")
    
    # 处理结果的辅助函数
    def process_batch_outputs(batch_outputs, start_idx):
        """处理一批输出并返回结果"""
        batch_results = []
        for j, output in enumerate(batch_outputs):
            i = start_idx + j
            query = queries[i]["query"]
            generated_text = output.outputs[0].text.strip()
            
            try:
                # 解析JSON输出
                classification = json.loads(generated_text)
                binary_harmfulness = classification.get("binary_harmfulness", "unknown")
                severity = classification.get("severity", None)
                harm_types = classification.get("harm_types", [])
                explanation = classification.get("explanation", "")
            except json.JSONDecodeError:
                # 如果JSON解析失败，设置默认值
                print(f"警告: 查询 {i} 的输出不是有效的JSON: {generated_text}")
                binary_harmfulness = "unknown"
                severity = None
                harm_types = []
                explanation = f"JSON解析失败: {generated_text[:50]}"
            
            batch_results.append({
                "query": query,
                "binary_harmfulness": binary_harmfulness,
                "severity": severity,
                "harm_types": harm_types,
                "explanation": explanation,
                "raw_output": generated_text,
                "meta_data": queries[i]
            })
        return batch_results
    
    # 如果指定了batch_size，则分批处理；否则一次处理所有数据
    all_results = []
    if batch_size and batch_size < len(prompts):
        print(f"使用分批处理模式，每批 {batch_size} 个查询")
        num_batches = (len(prompts) + batch_size - 1) // batch_size
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_num = i // batch_size + 1
            print(f"处理批次 {batch_num}/{num_batches} ({len(batch_prompts)} 个查询)...")
            
            # 执行批量推理
            batch_outputs = llm.generate(batch_prompts, sampling_params)
            
            # 处理这批结果
            batch_results = process_batch_outputs(batch_outputs, i)
            all_results.extend(batch_results)
            
            # 如果是JSONL格式，立即追加保存这批结果
            if is_jsonl:
                append_results(batch_results, output_file)
                print(f"  ✓ 已保存批次 {batch_num} 的 {len(batch_results)} 条结果")
    else:
        # 一次性处理所有数据（vLLM内部会根据max_num_seqs自动分批）
        print(f"开始批量推理 (vLLM将自动分批处理，每批最多 {max_num_seqs} 个序列)...")
        all_outputs = llm.generate(prompts, sampling_params)
        
        # 处理所有结果
        all_results = process_batch_outputs(all_outputs, 0)
        
        # 如果是JSONL格式，保存所有结果
        if is_jsonl:
            append_results(all_results, output_file)
            print(f"  ✓ 已保存所有 {len(all_results)} 条结果")
    
    return all_results


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
        default="/common/home/projectgrps/CS707/CS707G2/Qwen3Guard-Gen-8B",
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