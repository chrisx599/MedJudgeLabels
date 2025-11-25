"""
评测脚本：在测试集上评估模型性能
计算指标：Accuracy, Precision, Recall, F1-Score
"""
import json
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from unsloth import FastLanguageModel
from prepare_data import load_split_datasets, format_prompt
import numpy as np

def load_model(base_model_path, lora_adapter_path=None, max_seq_length=2048, load_in_4bit=True):
    """
    加载模型（支持原始模型 + LoRA适配器）
    
    Args:
        base_model_path: 原始模型路径
        lora_adapter_path: LoRA适配器路径（如果为None，则只加载原始模型）
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
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_adapter_path)
        print("✓ LoRA适配器加载完成")
    
    # 启用推理模式
    FastLanguageModel.for_inference(model)
    
    print("✓ 模型加载完成")
    return model, tokenizer

def parse_model_output(output_text):
    """
    解析模型输出，提取JSON结果
    
    Args:
        output_text: 模型生成的完整文本
    
    Returns:
        解析后的字典，如果解析失败返回None
    """
    # 提取assistant的响应 - 支持多种格式
    assistant_output = None
    
    # 方法1: 查找带标记的assistant部分
    assistant_start = output_text.find("<|im_start|>assistant\n")
    if assistant_start != -1:
        assistant_output = output_text[assistant_start + len("<|im_start|>assistant\n"):]
        assistant_end = assistant_output.find("<|im_end|>")
        if assistant_end != -1:
            assistant_output = assistant_output[:assistant_end]
    
    # 方法2: 查找纯文本assistant部分（模型可能输出纯文本格式）
    if assistant_output is None:
        assistant_marker = "\nassistant\n"
        assistant_start = output_text.find(assistant_marker)
        if assistant_start != -1:
            assistant_output = output_text[assistant_start + len(assistant_marker):]
        else:
            # 方法3: 尝试查找最后一个"assistant"后的内容
            assistant_marker_alt = "assistant\n"
            assistant_start = output_text.rfind(assistant_marker_alt)
            if assistant_start != -1:
                assistant_output = output_text[assistant_start + len(assistant_marker_alt):]
            else:
                # 如果都找不到，使用完整输出
                assistant_output = output_text
    
    # 清理输出
    assistant_output = assistant_output.strip()
    
    # 清理可能的markdown代码块标记
    if assistant_output.startswith("```json"):
        assistant_output = assistant_output[7:]
    if assistant_output.startswith("```"):
        assistant_output = assistant_output[3:]
    if assistant_output.endswith("```"):
        assistant_output = assistant_output[:-3]
    
    assistant_output = assistant_output.strip()
    
    # 尝试解析JSON
    try:
        result = json.loads(assistant_output)
        return result
    except json.JSONDecodeError:
        # 尝试提取JSON部分（可能有额外文本）
        import re
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', assistant_output)
        if json_match:
            try:
                result = json.loads(json_match.group(0))
                return result
            except json.JSONDecodeError:
                return None
        return None

def classify_single(model, tokenizer, query, response, max_new_tokens=256, temperature=0.3):
    """
    对单个样本进行分类
    
    Args:
        model: 模型
        tokenizer: tokenizer
        query: 用户查询
        response: 模型响应
        max_new_tokens: 最大生成token数
        temperature: 温度参数
    
    Returns:
        预测的binary_harmfulness，如果解析失败返回None
    """
    # 格式化输入
    prompt = format_prompt(query, response)
    
    # Tokenize
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=False,  # 使用贪婪解码以获得更稳定的结果
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 解码
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 解析输出
    result = parse_model_output(full_output)
    
    if result and 'binary_harmfulness' in result:
        return result['binary_harmfulness']
    else:
        return None

def evaluate_on_dataset(model, tokenizer, test_dataset, max_samples=None, save_predictions=None):
    """
    在测试集上评估模型
    
    Args:
        model: 模型
        tokenizer: tokenizer
        test_dataset: 测试数据集
        max_samples: 最大评估样本数（None表示全部）
        save_predictions: 保存预测结果的文件路径（None表示不保存）
    
    Returns:
        评估结果字典
    """
    print("\n" + "=" * 80)
    print("开始评估")
    print("=" * 80)
    
    # 限制样本数
    if max_samples:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))
    
    print(f"测试集大小: {len(test_dataset)}")
    
    y_true = []  # 真实标签
    y_pred = []  # 预测标签
    failed_samples = []  # 解析失败的样本
    all_predictions = []  # 所有预测结果（用于保存）
    
    # 遍历测试集
    for i, sample in enumerate(tqdm(test_dataset, desc="评估进度")):
        query = sample['query']
        response = sample['response']
        true_label = sample['binary_harmfulness']
        
        # 预测
        pred_label = classify_single(model, tokenizer, query, response)
        
        # 保存预测结果
        prediction_record = {
            'index': i,
            'query': query,
            'response': response,
            'true_label': true_label,
            'predicted_label': pred_label,
            'parse_success': pred_label is not None
        }
        all_predictions.append(prediction_record)
        
        if pred_label is not None:
            y_true.append(true_label)
            y_pred.append(pred_label)
        else:
            failed_samples.append({
                'index': i,
                'query': query[:100],
                'response': response[:100],
                'true_label': true_label
            })
    
    # 保存所有预测结果
    if save_predictions:
        print(f"\n保存预测结果到: {save_predictions}")
        with open(save_predictions, 'w', encoding='utf-8') as f:
            json.dump(all_predictions, f, ensure_ascii=False, indent=2)
        print("✓ 预测结果已保存")
    
    print(f"\n成功预测: {len(y_pred)}/{len(test_dataset)}")
    print(f"解析失败: {len(failed_samples)}/{len(test_dataset)}")
    
    if len(y_pred) == 0:
        print("错误：没有成功的预测！")
        return None
    
    # 计算指标
    results = calculate_metrics(y_true, y_pred)
    results['failed_samples'] = failed_samples
    results['total_samples'] = len(test_dataset)
    results['successful_predictions'] = len(y_pred)
    
    return results

def calculate_metrics(y_true, y_pred):
    """
    计算评估指标
    
    Args:
        y_true: 真实标签列表
        y_pred: 预测标签列表
    
    Returns:
        包含各种指标的字典
    """
    # 计算整体准确率
    accuracy = accuracy_score(y_true, y_pred)
    
    # 计算precision, recall, f1 (macro和weighted平均)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    # 计算每个类别的指标
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    
    # 获取类别名称
    unique_labels = sorted(list(set(y_true + y_pred)))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    
    results = {
        'accuracy': accuracy,
        'macro_precision': precision_macro,
        'macro_recall': recall_macro,
        'macro_f1': f1_macro,
        'weighted_precision': precision_weighted,
        'weighted_recall': recall_weighted,
        'weighted_f1': f1_weighted,
        'per_class_metrics': {},
        'confusion_matrix': cm.tolist(),
        'labels': unique_labels
    }
    
    # 每个类别的详细指标
    for i, label in enumerate(unique_labels):
        results['per_class_metrics'][label] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i]),
            'support': int(support_per_class[i])
        }
    
    return results

def print_results(results):
    """打印评估结果"""
    print("\n" + "=" * 80)
    print("评估结果")
    print("=" * 80)
    
    print(f"\n总样本数: {results['total_samples']}")
    print(f"成功预测: {results['successful_predictions']}")
    print(f"解析失败: {len(results['failed_samples'])}")
    
    print("\n整体指标:")
    print(f"  Accuracy:           {results['accuracy']:.4f} ({results['accuracy']*100:.2f}%)")
    
    print("\nMacro平均:")
    print(f"  Precision:          {results['macro_precision']:.4f}")
    print(f"  Recall:             {results['macro_recall']:.4f}")
    print(f"  F1-Score:           {results['macro_f1']:.4f}")
    
    print("\nWeighted平均:")
    print(f"  Precision:          {results['weighted_precision']:.4f}")
    print(f"  Recall:             {results['weighted_recall']:.4f}")
    print(f"  F1-Score:           {results['weighted_f1']:.4f}")
    
    print("\n各类别详细指标:")
    print("-" * 80)
    print(f"{'类别':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print("-" * 80)
    
    for label in results['labels']:
        metrics = results['per_class_metrics'][label]
        print(f"{label:<15} {metrics['precision']:<12.4f} {metrics['recall']:<12.4f} "
              f"{metrics['f1']:<12.4f} {metrics['support']:<10}")
    
    print("-" * 80)
    
    # 混淆矩阵
    print("\n混淆矩阵:")
    print("-" * 80)
    cm = np.array(results['confusion_matrix'])
    labels = results['labels']
    
    # 打印表头
    print(f"{'true pred':<15}")
    for label in labels:
        print(f"{label:<15}")
    print()
    print("-" * 80)
    
    # 打印矩阵
    for i, true_label in enumerate(labels):
        print(f"{true_label:<15}")
        for j in range(len(labels)):
            print(f"{cm[i][j]:<15}")
        print()
    
    print("-" * 80)
    
    # 如果有解析失败的样本，打印前几个
    if results['failed_samples']:
        print(f"\n解析失败的样本示例 (前5个):")
        print("-" * 80)
        for sample in results['failed_samples'][:5]:
            print(f"索引 {sample['index']}:")
            print(f"  Query: {sample['query']}...")
            print(f"  Response: {sample['response']}...")
            print(f"  真实标签: {sample['true_label']}")
            print()

def save_results(results, output_file):
    """保存评估结果到JSON文件"""
    print(f"\n保存结果到: {output_file}")
    
    # 转换numpy类型为Python原生类型
    results_to_save = {
        'accuracy': float(results['accuracy']),
        'macro_precision': float(results['macro_precision']),
        'macro_recall': float(results['macro_recall']),
        'macro_f1': float(results['macro_f1']),
        'weighted_precision': float(results['weighted_precision']),
        'weighted_recall': float(results['weighted_recall']),
        'weighted_f1': float(results['weighted_f1']),
        'per_class_metrics': results['per_class_metrics'],
        'confusion_matrix': results['confusion_matrix'],
        'labels': results['labels'],
        'total_samples': results['total_samples'],
        'successful_predictions': results['successful_predictions'],
        'failed_samples_count': len(results['failed_samples']),
        'failed_samples': results['failed_samples'][:10]  # 只保存前10个失败样本
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_to_save, f, ensure_ascii=False, indent=2)
    
    print("✓ 结果已保存")

def evaluate_from_predictions(predictions_file, output_file=None):
    """
    从保存的预测结果计算指标
    
    Args:
        predictions_file: 预测结果文件路径
        output_file: 输出文件路径（None表示不保存）
    
    Returns:
        评估结果字典
    """
    print("\n" + "=" * 80)
    print("从保存的预测结果计算指标")
    print("=" * 80)
    print(f"加载预测结果: {predictions_file}")
    
    with open(predictions_file, 'r', encoding='utf-8') as f:
        all_predictions = json.load(f)
    
    print(f"✓ 加载了 {len(all_predictions)} 个预测结果")
    
    y_true = []
    y_pred = []
    failed_samples = []
    
    for pred in all_predictions:
        if pred['parse_success']:
            y_true.append(pred['true_label'])
            y_pred.append(pred['predicted_label'])
        else:
            failed_samples.append({
                'index': pred['index'],
                'query': pred['query'][:100],
                'response': pred['response'][:100],
                'true_label': pred['true_label']
            })
    
    print(f"\n成功预测: {len(y_pred)}/{len(all_predictions)}")
    print(f"解析失败: {len(failed_samples)}/{len(all_predictions)}")
    
    if len(y_pred) == 0:
        print("错误：没有成功的预测！")
        return None
    
    # 计算指标
    results = calculate_metrics(y_true, y_pred)
    results['failed_samples'] = failed_samples
    results['total_samples'] = len(all_predictions)
    results['successful_predictions'] = len(y_pred)
    
    # 打印结果
    print_results(results)
    
    # 保存结果
    if output_file:
        save_results(results, output_file)
    
    return results

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='评估训练好的模型')
    parser.add_argument('--base_model_path', type=str,
                        default='/common/home/projectgrps/CS707/CS707G2/scratchDirectory/Qwen3-8B',
                        help='原始模型路径')
    parser.add_argument('--lora_adapter_path', type=str, default=None,
                        help='LoRA适配器路径（默认: ./outputs/lora_adapter）')
    parser.add_argument('--test_data_path', type=str,
                        default='/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_test.jsonl',
                        help='测试集数据路径')
    parser.add_argument('--output', type=str, default='./evaluation_results.json',
                        help='输出文件路径（默认: ./evaluation_results.json）')
    parser.add_argument('--predictions_file', type=str, default='./predictions.json',
                        help='保存/加载预测结果的文件路径（默认: ./predictions.json）')
    parser.add_argument('--from_predictions', type=str, default=None,
                        help='从已保存的预测结果计算指标（如果指定，则跳过推理）')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大评估样本数（默认: None，评估全部）')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                        help='最大序列长度（默认: 2048）')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                        help='是否使用4-bit量化（默认: False）')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("模型评估脚本")
    print("=" * 80)
    print(f"基础模型路径: {args.base_model_path}")
    print(f"LoRA适配器路径: {args.lora_adapter_path}")
    print(f"测试集路径: {args.test_data_path}")
    print(f"输出文件: {args.output}")
    if args.max_samples:
        print(f"最大样本数: {args.max_samples}")
    
    # 1. 加载模型
    model, tokenizer = load_model(
        args.base_model_path,
        args.lora_adapter_path,
        max_seq_length=args.max_seq_length,
        load_in_4bit=args.load_in_4bit
    )
    
    # 2. 加载测试数据
    print("\n加载测试数据...")
    # 使用空的训练集路径，只加载测试集
    import json
    from datasets import Dataset
    
    test_data = []
    with open(args.test_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line.strip()))
    
    test_dataset = Dataset.from_list(test_data)
    print(f"✓ 测试集加载完成，共 {len(test_dataset)} 个样本")
    
    # 如果指定了从预测结果计算指标，则跳过推理
    if args.from_predictions:
        results = evaluate_from_predictions(args.from_predictions, args.output)
        if results is None:
            print("评估失败！")
            return
    else:
        # 3. 评估（包含推理）
        results = evaluate_on_dataset(
            model,
            tokenizer,
            test_dataset,
            max_samples=args.max_samples,
            save_predictions=args.predictions_file
        )
        
        if results is None:
            print("评估失败！")
            return
        
        # 4. 打印结果
        print_results(results)
        
        # 5. 保存结果
        save_results(results, args.output)
    
    print("\n" + "=" * 80)
    print("评估完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()