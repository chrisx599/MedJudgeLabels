"""
使用Unsloth微调Qwen3-8B模型进行医疗安全分类（原始标签版本）
训练标签：is_response_safe（二分类）和 response_severity_level（多分类）
"""
import os
import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from prepare_data import load_split_datasets

# ============ 配置参数 ============
MODEL_PATH = "/common/home/projectgrps/CS707/CS707G2/scratchDirectory/Qwen3-8B"
TRAIN_DATA_PATH = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_train_orig.jsonl"
TEST_DATA_PATH = "/common/home/projectgrps/CS707/CS707G2/MedJudgeLabels/data/pku_anno_formatted_test_orig.jsonl"
OUTPUT_DIR = "./outputs"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA配置
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# 训练配置
PER_DEVICE_TRAIN_BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
WARMUP_STEPS = 10
NUM_TRAIN_EPOCHS = 1
LEARNING_RATE = 2e-4
LOGGING_STEPS = 10
SAVE_STEPS = 100
MAX_STEPS = -1  # -1表示训练完整个epoch

def main():
    print("=" * 80)
    print("开始Qwen3-8B模型微调（原始安全标签版本）")
    print("=" * 80)
    
    # ============ 1. 加载模型和tokenizer ============
    print("\n[1/5] 加载模型和tokenizer...")
    print(f"模型路径: {MODEL_PATH}")
    print(f"最大序列长度: {MAX_SEQ_LENGTH}")
    print(f"4-bit量化: {LOAD_IN_4BIT}")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_PATH,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=None,  # 自动检测
        load_in_4bit=LOAD_IN_4BIT,
    )
    
    print(f"✓ 模型加载成功")
    print(f"  模型类型: {type(model).__name__}")
    print(f"  Tokenizer类型: {type(tokenizer).__name__}")
    
    # ============ 2. 应用LoRA ============
    print("\n[2/5] 应用LoRA配置...")
    print(f"LoRA rank (r): {LORA_R}")
    print(f"LoRA alpha: {LORA_ALPHA}")
    print(f"LoRA dropout: {LORA_DROPOUT}")
    print(f"目标模块: {TARGET_MODULES}")
    
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # 使用unsloth优化的梯度检查点
        random_state=3407,
        max_seq_length=MAX_SEQ_LENGTH,
        use_rslora=False,
        loftq_config=None,
    )
    
    print("✓ LoRA配置完成")
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"  可训练参数: {trainable_params:,} ({trainable_params/all_params*100:.2f}%)")
    print(f"  总参数: {all_params:,}")
    
    # ============ 3. 加载数据集 ============
    print("\n[3/5] 加载训练数据...")
    print(f"训练集路径: {TRAIN_DATA_PATH}")
    print(f"测试集路径: {TEST_DATA_PATH}")
    print("\n标签说明:")
    print("  is_response_safe: true (harmless) or false (harmful)")
    print("  response_severity_level:")
    print("    - 0: safe (harmless)")
    print("    - 1: minor (mild harm)")
    print("    - 2: moderate (moderate harm)")
    print("    - 3: severe (severe harm)")
    
    # 直接加载已分割好的数据集
    datasets = load_split_datasets(TRAIN_DATA_PATH, TEST_DATA_PATH)
    train_dataset = datasets['train']
    eval_dataset = datasets['test']
    
    print(f"\n✓ 数据准备完成")
    print(f"  训练集大小: {len(train_dataset):,}")
    print(f"  测试集大小: {len(eval_dataset):,}")
    
    # 统计标签分布
    safe_count = sum(1 for item in train_dataset if item['is_response_safe'])
    unsafe_count = len(train_dataset) - safe_count
    print(f"\n训练集标签分布:")
    print(f"  Safe (is_response_safe=True): {safe_count:,} ({safe_count/len(train_dataset)*100:.1f}%)")
    print(f"  Unsafe (is_response_safe=False): {unsafe_count:,} ({unsafe_count/len(train_dataset)*100:.1f}%)")
    
    severity_dist = {}
    for item in train_dataset:
        level = item['response_severity_level']
        severity_dist[level] = severity_dist.get(level, 0) + 1
    print(f"\n  Severity Level分布:")
    for level in sorted(severity_dist.keys()):
        count = severity_dist[level]
        print(f"    Level {level}: {count:,} ({count/len(train_dataset)*100:.1f}%)")
    
    # 打印数据示例
    print("\n数据示例:")
    print("-" * 80)
    example = train_dataset[0]['text']
    print(example[:600] + "..." if len(example) > 600 else example)
    print("-" * 80)
    
    # ============ 4. 配置训练参数 ============
    print("\n[4/5] 配置训练参数...")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        warmup_steps=WARMUP_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        learning_rate=LEARNING_RATE,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=LOGGING_STEPS,
        logging_dir=f"{OUTPUT_DIR}/logs",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_steps=SAVE_STEPS,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        report_to="none",  # 不使用wandb等
    )
    
    print("✓ 训练参数配置完成")
    print(f"  批次大小: {PER_DEVICE_TRAIN_BATCH_SIZE}")
    print(f"  梯度累积步数: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  有效批次大小: {PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  学习率: {LEARNING_RATE}")
    print(f"  训练轮数: {NUM_TRAIN_EPOCHS}")
    print(f"  优化器: adamw_8bit")
    print(f"  精度: {'bf16' if torch.cuda.is_bf16_supported() else 'fp16'}")
    
    # ============ 5. 开始训练 ============
    print("\n[5/5] 开始训练...")
    
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        packing=False,  # 不使用packing，因为我们的数据已经格式化好了
        args=training_args,
    )
    
    print("\n开始训练...")
    print("=" * 80)
    
    trainer_stats = trainer.train()
    
    print("\n" + "=" * 80)
    print("训练完成！")
    print("=" * 80)
    print(f"训练统计:")
    print(f"  总步数: {trainer_stats.global_step}")
    print(f"  训练损失: {trainer_stats.training_loss:.4f}")
    
    # ============ 6. 保存模型 ============
    print("\n保存模型...")
    
    # 只保存LoRA适配器（不合并模型，避免Unsloth的bug）
    model.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    tokenizer.save_pretrained(f"{OUTPUT_DIR}/lora_adapter")
    print(f"✓ LoRA适配器已保存到: {OUTPUT_DIR}/lora_adapter")
    print(f"  推理时需要同时加载原始模型和LoRA适配器")
    
    print("\n" + "=" * 80)
    print("所有任务完成！")
    print("=" * 80)

if __name__ == "__main__":
    main()
