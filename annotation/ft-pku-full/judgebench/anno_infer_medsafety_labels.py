"""
使用 LoRA 训练参数对 medsafety_labels.jsonl 进行推理标注
参考:
- annotation/ft-pku_pku_anno_formatted_test/anno_infer_ft_pku.py
- annotation/gpt-5-mini/judgebench-test-eval/run_medsafety_labels_async.py
"""

import os
import json
import torch
from tqdm import tqdm
from datetime import datetime
from unsloth import FastLanguageModel
from peft import PeftModel


def format_prompt(query, response, prompt_template_path):
    """根据 gpt-5-mini 标注模板格式化输入"""
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        template = f.read()
    # 用模板占位符替换
    prompt = template.replace("{{ query }}", query).replace("{{ response }}", response)
    formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    return formatted_prompt


def load_lora_model(base_model_path, lora_path, max_seq_length=4096, load_in_4bit=True):
    """加载基础模型和LoRA适配器"""
    print(f"正在加载基础模型: {base_model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )

    print(f"正在加载LoRA适配器: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path)
    FastLanguageModel.for_inference(model)
    print("✓ 模型加载完成")
    return model, tokenizer


def parse_output(text):
    """解析模型输出为JSON"""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        try:
            first, last = text.find("{"), text.rfind("}")
            if first != -1 and last != -1:
                return json.loads(text[first : last + 1])
        except Exception:
            return None
    return None


def infer_single(model, tokenizer, query, response, prompt_path):
    """单样本推理"""
    prompt = format_prompt(query, response, prompt_path)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.2,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    result = parse_output(decoded)
    return result


def main():
    base_model = "/common/home/projectgrps/CS707/CS707G2/scratchDirectory/Qwen3-8B"
    lora_adapter_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/train/outputs/lora_adapter"
    input_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/medsafety_labels.jsonl"
    prompt_path = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/annotation/medjudge_unified_prd_v1.4.txt"

    # 输出路径对齐 run_medsafety_labels_async.py 结构
    output_dir = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/annotation/ft-pku_pku_anno_formatted_test/judgebench/artifacts"

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_jsonl = os.path.join(output_dir, f"medsafety_labels_infer_{timestamp}.jsonl")
    output_csv = os.path.join(output_dir, f"medsafety_labels_infer_{timestamp}.csv")

    print("=" * 80)
    print("使用LoRA模型进行MedSafety标注推理")
    print("=" * 80)
    print(f"数据文件: {input_path}")
    print(f"Prompt 模板: {prompt_path}")
    print(f"输出目录: {output_dir}")
    print()

    # 1. 加载模型
    model, tokenizer = load_lora_model(base_model, lora_adapter_path, max_seq_length=4096, load_in_4bit=True)

    # 2. 读取数据
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line.strip()) for line in f]
    print(f"✓ 加载 {len(data)} 条样本")

    # 3. 批量推理标注
    results_jsonl = []
    csv_lines = ["id,query,response,query_risk_level-gpt,respond_type-gpt,respond_stdtype-gpt\n"]

    batch_size = 8  # 默认批次大小，可根据显存动态调整
    total = len(data)

    for start in tqdm(range(0, total, batch_size), desc="MedSafety 批量推理中"):
        batch = data[start : start + batch_size]
        prompts = [format_prompt(item["query"], item["response"], prompt_path) for item in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                temperature=0.2,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs['input_ids'].shape[1]
        generated_tokens_batch = outputs[:, input_length:]
        decoded_batch = tokenizer.batch_decode(generated_tokens_batch, skip_special_tokens=True)

        # 修复解析逻辑：正确截取 assistant 输出并确保 JSON 能被解析
        for item, decoded_text in zip(batch, decoded_batch):
            rid, q, r = item["id"], item["query"], item["response"]

            decoded_section = decoded_text.strip()

            # 清理 markdown 包装
            if decoded_section.startswith("```json"):
                decoded_section = decoded_section[7:]
            if decoded_section.startswith("```"):
                decoded_section = decoded_section[3:]
            if decoded_section.endswith("```"):
                decoded_section = decoded_section[:-3]
            decoded_section = decoded_section.strip()

            # 提取 JSON 部分并解析结果
            try:
                # 改进解析逻辑：参考 anno_infer_ft_pku.py 的 parse_annotation_output 方法
                import re
                decoded_text_clean = decoded_section.strip()

                # Step 1: 定位 assistant 输出（已在 decoded_section 中）
                # 清理markdown代码块标记
                if decoded_text_clean.startswith("```json"):
                    decoded_text_clean = decoded_text_clean[7:]
                if decoded_text_clean.startswith("```"):
                    decoded_text_clean = decoded_text_clean[3:]
                if decoded_text_clean.endswith("```"):
                    decoded_text_clean = decoded_text_clean[:-3]
                decoded_text_clean = decoded_text_clean.strip()

                infer_result = None
                # Step 2: 优先尝试整体JSON解析
                try:
                    infer_result = json.loads(decoded_text_clean)
                except json.JSONDecodeError:
                    pass

                # Step 3: 若失败，尝试用首尾花括号截取
                if not infer_result:
                    first_brace = decoded_text_clean.find('{')
                    last_brace = decoded_text_clean.rfind('}')
                    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                        json_str = decoded_text_clean[first_brace:last_brace+1]
                        try:
                            infer_result = json.loads(json_str)
                        except json.JSONDecodeError:
                            pass

                # Step 4: 最后使用正则匹配嵌套JSON
                if not infer_result:
                    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', decoded_text_clean)
                    if json_match:
                        try:
                            infer_result = json.loads(json_match.group(0))
                        except json.JSONDecodeError:
                            infer_result = None
            except Exception as e:
                print(f"[解析失败] {rid} : {e}")
                print(f"原始输出片段: {decoded_section[:200]}")
                infer_result = None

            # 统一输出格式
            result_record = {
                "id": rid,
                "query": q,
                "response": r,
                "query_risk_level-human": item.get("query_risk_level-human"),
                "respond_type-human": item.get("respond_type-human"),
                "query_risk_level-gpt": None,
                "respond_type-gpt": None,
                "respond_stdtype-gpt": None,
            }

            # 如果成功解析到结果，提取字段
            if infer_result:
                result_record["query_risk_level-gpt"] = infer_result.get("query_risk_level-gpt")
                result_record["respond_type-gpt"] = infer_result.get("respond_type-gpt")
                result_record["respond_stdtype-gpt"] = infer_result.get("respond_stdtype-gpt")
                
                # 如果解析后缺少字段，打印前 80 chars 方便调试
                missing = [k for k in ["query_risk_level-gpt", "respond_type-gpt", "respond_stdtype-gpt"] if k not in infer_result]
                if missing:
                    print(f"[警告] {rid} 模型输出缺少字段 {missing}，原始输出截断预览：{json.dumps(infer_result)[:80]}")
            else:
                # 如果没有成功解析到结果，则保持 None
                result_record["query_risk_level-gpt"] = None
                result_record["respond_type-gpt"] = None
                result_record["respond_stdtype-gpt"] = None

            # 增加 raw 模型输出，便于追踪
            result_record["raw_response"] = decoded_section

            results_jsonl.append(result_record)

            # CSV 输出
            csv_line = f"{rid},{q.replace(',', '，')},{r.replace(',', '，')},{result_record['query_risk_level-gpt'] or ''},{result_record['respond_type-gpt'] or ''},{result_record['respond_stdtype-gpt'] or ''}\n"
            csv_lines.append(csv_line)

            # JSONL 保存原始输出字段
            with open(output_jsonl, "a", encoding="utf-8") as jf:
                jf.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                jf.flush()

    # 4. 写出CSV
    with open(output_csv, "w", encoding="utf-8") as cf:
        cf.writelines(csv_lines)

    print("\n✓ 推理标注完成！")
    print(f"JSONL 输出: {output_jsonl}")
    print(f"CSV 输出: {output_csv}")
    print("=" * 80)


if __name__ == "__main__":
    main()