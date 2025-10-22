# vLLM Qwen3-4B 医疗查询分类推理脚本

这个脚本使用vLLM和Qwen3-4B模型进行批量推理，用于分类医疗相关查询。

## 功能特点

- ✅ 使用vLLM进行高效批量推理
- ✅ 支持JSON schema约束，确保输出格式正确
- ✅ 支持多种输入格式（.txt, .json, .jsonl）
- ✅ **增量保存**：JSONL格式支持每批推理后立即保存，防止数据丢失
- ✅ 自动统计分类结果
- ✅ 支持GPU张量并行加速

## 安装依赖

```bash
pip install vllm
```

## 使用方法

### 基本用法

```bash
python classify/vllm_qwen3_4b_infer.py \
    --input classify/example_queries.txt \
    --output classify/results.json
```

### 完整参数说明

```bash
python classify/vllm_qwen3_4b_infer.py \
    --input <输入文件路径> \
    --output <输出文件路径> \
    --model <模型名称或路径> \
    --prompt <prompt模板文件路径> \
    --temperature <采样温度> \
    --max-tokens <最大生成token数> \
    --tensor-parallel-size <张量并行大小> \
    --gpu-memory-utilization <GPU内存利用率> \
    --max-num-seqs <批处理大小> \
    --batch-size <手动分批大小>
```

### 参数详解

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 输入文件路径（必需） | - |
| `--output` | 输出文件路径（必需） | - |
| `--model` | 模型名称或路径 | `Qwen/Qwen2.5-3B-Instruct` |
| `--prompt` | Prompt模板文件路径 | `classify/classify_prompt.txt` |
| `--temperature` | 采样温度（0.0为确定性输出） | `0.0` |
| `--max-tokens` | 最大生成token数 | `100` |
| `--tensor-parallel-size` | 张量并行大小（多GPU） | `1` |
| `--gpu-memory-utilization` | GPU内存利用率 | `0.9` |
| `--max-num-seqs` | vLLM调度器的最大序列数（batch size） | `256` |
| `--batch-size` | 手动分批处理的批次大小（可选） | `None` |
| `--max-model-len` | 最大模型序列长度（控制KV cache内存） | `8192` |

## 输入格式

脚本支持三种输入格式：

### 1. 纯文本文件 (.txt)

每行一个查询：

```text
What are the symptoms of pneumonia?
What is the capital of Japan?
How to treat a broken bone?
```

### 2. JSON文件 (.json)

查询列表：

```json
[
  "What are the symptoms of pneumonia?",
  "What is the capital of Japan?",
  "How to treat a broken bone?"
]
```

或包含'queries'键的字典：

```json
{
  "queries": [
    "What are the symptoms of pneumonia?",
    "What is the capital of Japan?"
  ]
}
```

### 3. JSONL文件 (.jsonl)

每行一个JSON对象：

```jsonl
{"query": "What are the symptoms of pneumonia?"}
{"query": "What is the capital of Japan?"}
{"query": "How to treat a broken bone?"}
```

或简单字符串：

```jsonl
"What are the symptoms of pneumonia?"
"What is the capital of Japan?"
```

## 输出格式

### JSON格式（.json）
输出为JSON数组，包含所有查询的分类结果：

```json
[
  {
    "query": "What are the symptoms of pneumonia?",
    "label": "medical",
    "raw_output": "{\"label\": \"medical\"}"
  },
  {
    "query": "What is the capital of Japan?",
    "label": "others",
    "raw_output": "{\"label\": \"others\"}"
  }
]
```

### JSONL格式（.jsonl）- **推荐用于大数据集**
每行保存一个结果，**支持增量保存**：

```jsonl
{"query": "What are the symptoms of pneumonia?", "label": "medical", "raw_output": "{\"label\": \"medical\"}"}
{"query": "What is the capital of Japan?", "label": "others", "raw_output": "{\"label\": \"others\"}"}
```

**重要特性**：
- ✅ **防止数据丢失**：每批推理完成后立即追加保存到文件
- ✅ **断点续传友好**：即使中途中断，已处理的数据也已保存
- ✅ **内存友好**：不需要在内存中保存所有结果

## 使用示例

### 示例1：使用默认模型和参数

```bash
python classify/vllm_qwen3_4b_infer.py \
    --input classify/example_queries.txt \
    --output classify/results.json
```

### 示例2：使用自定义模型

```bash
python classify/vllm_qwen3_4b_infer.py \
    --input data/queries.jsonl \
    --output results/classifications.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct
```

### 示例3：控制批处理大小

```bash
# 设置vLLM内部批处理大小为128
python classify/vllm_qwen3_4b_infer.py \
    --input large_dataset.txt \
    --output results.json \
    --max-num-seqs 128
```

### 示例4：手动分批处理大数据集（推荐使用JSONL格式）

```bash
# 每次处理1000个查询，每批完成后立即保存
# 使用.jsonl格式可以防止中途意外导致数据丢失
python classify/vllm_qwen3_4b_infer.py \
    --input huge_dataset.txt \
    --output results.jsonl \
    --batch-size 1000 \
    --max-num-seqs 256
```

**增量保存说明**：
- 使用`.jsonl`格式时，每批推理完成后会立即追加保存到文件
- 即使程序中途中断，已完成的批次数据不会丢失
- 适合处理大规模数据集（数万到数百万条）

### 示例5：多GPU推理

```bash
python classify/vllm_qwen3_4b_infer.py \
    --input large_dataset.txt \
    --output results.json \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.95 \
    --max-num-seqs 512
```

### 示例6：使用本地模型

```bash
python classify/vllm_qwen3_4b_infer.py \
    --input queries.txt \
    --output results.json \
    --model /path/to/local/qwen-model
```

## 分类标签

脚本会将查询分类为以下两类：

- **medical**: 与医疗、健康或生物医学相关的查询
- **others**: 其他类型的查询

分类规则详见 [`classify_prompt.txt`](classify_prompt.txt)。

## Batch Inference说明

脚本支持两种批处理模式：

### 1. vLLM内部批处理（推荐）

通过`--max-num-seqs`参数控制vLLM调度器可以同时处理的最大序列数。这是**真正的batch inference**，vLLM会自动将多个请求组合成批次进行并行处理。

```bash
# 设置批处理大小为256（默认值）
python classify/vllm_qwen3_4b_infer.py \
    --input data.txt \
    --output results.json \
    --max-num-seqs 256
```

**优点**：
- ✅ 真正的并行批处理，性能最优
- ✅ vLLM自动优化内存和计算
- ✅ 支持continuous batching，动态调度

**调优建议**：
- 增大`max-num-seqs`可以提高吞吐量，但会占用更多GPU内存
- 如果遇到OOM，减小`max-num-seqs`或降低`gpu-memory-utilization`
- 典型值：64-512，取决于模型大小和GPU内存

### 2. 手动分批处理（可选）

通过`--batch-size`参数手动将大数据集分成多个批次处理。这适用于：
- 超大数据集（数百万条）
- 需要中间保存结果
- 内存受限的场景

```bash
# 每次处理1000个查询
python classify/vllm_qwen3_4b_infer.py \
    --input huge_data.txt \
    --output results.json \
    --batch-size 1000 \
    --max-num-seqs 256
```

**注意**：`batch-size`和`max-num-seqs`是两个不同的概念：
- `batch-size`：手动分批，每批包含多少个查询
- `max-num-seqs`：vLLM内部批处理大小，每批内部并行处理多少个序列

## 性能优化建议

1. **批处理大小**:
   - 使用`--max-num-seqs`控制vLLM的batch size（默认256）
   - 增大此值可提高吞吐量，但需要更多GPU内存
   - 建议范围：64-512，根据GPU内存调整

2. **GPU内存**:
   - 如果遇到OOM错误，降低`--gpu-memory-utilization`参数（如0.7-0.8）
   - 或者减小`--max-num-seqs`参数

3. **多GPU**:
   - 对于大模型，使用`--tensor-parallel-size`参数启用多GPU并行
   - 同时可以增大`--max-num-seqs`以充分利用多GPU性能

4. **温度设置**:
   - 使用`temperature=0.0`获得确定性输出，适合分类任务

5. **大数据集处理**:
   - 对于超大数据集，使用`--batch-size`参数分批处理
   - 例如：`--batch-size 10000 --max-num-seqs 256`

## 故障排除

### 问题1: CUDA Out of Memory

**解决方案**: 降低GPU内存利用率

```bash
--gpu-memory-utilization 0.7
```

### 问题2: 模型加载失败

**解决方案**: 确保模型名称正确，或使用本地模型路径

```bash
--model /path/to/local/model
```

### 问题3: JSON解析错误

脚本会自动处理JSON解析错误，并尝试从文本中提取标签。如果频繁出现此问题，可能需要调整prompt模板。

## 技术细节

- **框架**: vLLM (高性能LLM推理引擎)
- **模型**: Qwen系列模型（支持所有Qwen模型）
- **输出约束**: 使用JSON schema确保输出格式正确
- **并行化**: 支持张量并行和批量推理

## 许可证

本脚本遵循Apache 2.0许可证。