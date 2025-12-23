#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PEFT (LoRA) 파인튜닝 - SFTTrainer + DeepSpeed ZeRO + Unsloth"""

import os
import json
import torch
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Any

warnings.filterwarnings('ignore')

from unsloth import is_bfloat16_supported, FastLanguageModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from safetensors.torch import load_file, save_file
import wandb, weave
from unsloth import unsloth_train




def load_jsonl_or_json(path: str) -> List[Dict]:
    """JSONL 또는 JSON 파일 로드"""
    data = []
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    else:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return data


def normalize_example(item: Dict[str, Any]) -> Dict[str, Any] | None:
    """다양한 스키마를 {question, answer} 형식으로 정규화"""
    q, a = None, None
    
    # {"qas": [...]} 형식
    if 'qas' in item and item['qas']:
        qa = item['qas'][0]
        q, a = qa.get('question'), qa.get('answer')
    
    # {"question": ..., "answer": ...} 형식
    elif item.get('question') and item.get('answer'):
        q, a = item['question'], item['answer']
    
    # {"input": {"question": ...}, "output": {"answer": ...}} 형식
    elif isinstance(item.get('input'), dict) and isinstance(item.get('output'), dict):
        q = item['input'].get('question')
        a = item['output'].get('answer')
    
    # 빈 문자열 체크
    if q and a and str(q).strip() and str(a).strip():
        return {'question': str(q).strip(), 'answer': str(a).strip()}
    
    return None


def load_dataset_from_path(path: str, max_samples: int = None) -> Dataset:
    """데이터셋 로드 및 정규화 (question, answer 필드 유지)"""
    data = load_jsonl_or_json(path)
    
    if max_samples and len(data) > max_samples:
        data = data[:max_samples]
    
    data = [normalize_example(ex) for ex in data]
    data = [ex for ex in data if ex is not None]
    
    return Dataset.from_list(data)


def create_formatting_func(tokenizer, max_seq_length=2048):
    """question, answer → conversations → text + input_ids 변환"""
    def formatting_prompts_func(examples):
        # 시스템 프롬프트
        system_content = """다음 질문에 대해 정확하고 상세하게 답변해주세요.

규칙:
1. 질문의 핵심을 파악하여 명확하게 답변하세요.
2. 기술적인 용어가 있다면 정확하게 설명하세요.
3. 논리적이고 체계적인 답변을 작성하세요."""
        
        # question, answer를 conversations 형식으로 변환
        convos = []
        for question, answer in zip(examples.get('question', []), examples.get('answer', [])):
            # 빈 값 체크
            if not question or not answer or not str(question).strip() or not str(answer).strip():
                continue
            
            # Chat message 구성 (system, user, assistant 순서)
            messages = [
                {"role": "system", "content": str(system_content)},
                {"role": "user", "content": str(question).strip()},
                {"role": "assistant", "content": str(answer).strip()}
            ]
            convos.append(messages)
        
        # conversations를 text로 변환 (removeprefix로 bos 토큰 제거)
        texts = []
        for convo in convos:
            text = tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False)
            # bos 토큰 제거 (있는 경우)
            if text.startswith('<bos>'):
                text = text.removeprefix('<bos>')
            texts.append(text)
        
        return {"text": texts}
    return formatting_prompts_func


def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일 로드"""
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일이 필요합니다: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='PEFT 학습')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--deepspeed', type=str)
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    
    config = load_config(args.config)
    if args.deepspeed:
        config['deepspeed'] = args.deepspeed
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    print("=" * 60)
    print(f"🚀 PEFT 학습: {config['model_name']}")
    print(f"   데이터: {config['data_path']}")
    print(f"   모드: {'DeepSpeed' if config.get('deepspeed') else 'Default'}")
    print("=" * 60)
    
    # 1. 데이터셋 로드
    train_dataset = load_dataset_from_path(config['data_path'], config.get('max_samples'))
    print(f"✅ 학습 데이터: {len(train_dataset)}개")
    
    eval_dataset = None
    if config.get('eval_data_path') and os.path.exists(config['eval_data_path']):
        eval_dataset = load_dataset_from_path(config['eval_data_path'])
        print(f"✅ 평가 데이터: {len(eval_dataset)}개")
    elif config.get('validation_split', 0) > 0:
        split = train_dataset.train_test_split(test_size=config['validation_split'], seed=42)
        train_dataset, eval_dataset = split['train'], split['test']
        print(f"✅ 학습/검증 분할: {len(train_dataset)}/{len(eval_dataset)}")
    
    # 2. 모델 로딩
    print(f"🔄 모델 로딩 중: {config['model_name']}")
    dtype = torch.bfloat16 if torch.cuda.is_available() and is_bfloat16_supported() else torch.float16
    max_seq_length = config.get('max_length', 2048)
    
    if config.get('full_finetuning', False):
        print("✅ Full finetuning 모드")
    else:
        print("✅ PEFT 모드")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model_name'],
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=True,  # 4 bit quantization to reduce memory
        load_in_8bit=False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning=config.get('full_finetuning', False), # [NEW!] We have full finetuning now!
        # attn_implementation="eager",
        attn_implementation="flash_attention_3",
        # device_map = "balanced"
    )
    print("✅ 모델 로딩 완료")
    
    # 3. Chat template 설정
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-3")
    print(f"✅ Tokenizer: eos={tokenizer.eos_token}, pad={tokenizer.pad_token}")
    
    # 4. 데이터셋 전처리
    formatting_func = create_formatting_func(tokenizer, max_seq_length)
    train_dataset = train_dataset.map(formatting_func, batched=True)
    if eval_dataset:
        eval_dataset = eval_dataset.map(formatting_func, batched=True)
    print("✅ 데이터셋 전처리 완료")
    
    # 디버그 출력
    if local_rank <= 0 and len(train_dataset) > 0:
        print(f"\n🔍 첫 번째 샘플:\n{train_dataset[0]['text'][:]}\n")
    
    # 5. LoRA 설정
    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config.get('lora_dropout', 0.05),
        target_modules=config['target_modules'],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )
    if hasattr(model, 'print_trainable_parameters'):
        model.print_trainable_parameters()
    
    # 6. wandb
    if config.get('use_wandb') and local_rank <= 0:
        weave.init(config.get('wandb_project', 'full_finetuning'))
        wandb.init(project=config.get('wandb_project', 'full_finetuning'), name=f"{os.path.basename(config['model_name'])}_{timestamp}", config=config)
    
    # 7. Warmup 계산
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    global_batch_size = config['batch_size'] * config.get('gradient_accumulation_steps', 1) * num_gpus
    
    # Total_Steps = (N × Epochs) / (Batch_Size × Grad_Accumulation)
    N = len(train_dataset)
    Epochs = config['num_epochs']
    Batch_Size = config['batch_size']
    Grad_Accumulation = config.get('gradient_accumulation_steps', 1)
    
    total_steps = int((N * Epochs) / (Batch_Size * Grad_Accumulation))
    save_steps = int(total_steps * 0.1)
    eval_steps = int(total_steps * 0.1) if eval_dataset else None
    logging_steps = int(total_steps * 0.01)
    warmup_steps = int(total_steps * config.get('warmup_ratio', 0.05))
    
    if local_rank <= 0:
        print(f"📊 GPU: {num_gpus}, 배치: {global_batch_size}, 스텝: {total_steps}, Warmup: {warmup_steps}, Logging: {logging_steps}, Save/Eval: {save_steps}")
    
    # 8. SFTConfig
    # 모델 이름의 마지막 부분만 사용 (예: "Qwen/Qwen3-Coder-30B-A3B-Instruct" -> "Qwen3-Coder-30B-A3B-Instruct")
    model_name_safe = config['model_name'].split('/')[-1]
    
    if config.get('full_finetuning', False):
        output_dir = os.path.join(config['output_dir'], model_name_safe, f"full_{timestamp}")
    else:
        output_dir = os.path.join(config['output_dir'], model_name_safe, f"lora_{timestamp}")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        learning_rate=config['learning_rate'],
        warmup_steps=warmup_steps,
        logging_steps=logging_steps,
        save_steps=save_steps,
        eval_steps=eval_steps,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        bf16=True,
        optim=config.get('optim', 'adamw_torch'),
        lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),
        weight_decay=config.get('weight_decay', 0.01),
        max_grad_norm=config.get('max_grad_norm', 1.0),
        save_total_limit=3,
        dataloader_drop_last=True,
        packing=True,
        gradient_checkpointing=False,
        deepspeed=config.get('deepspeed'),
        local_rank=args.local_rank,
        report_to="wandb" if config.get('use_wandb') else None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
    )
    
    # 9. Trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
    )
    
    # 10. Response-only 학습
    trainer = train_on_responses_only(trainer, instruction_part = "<start_of_turn>user\n",response_part = "<start_of_turn>model\n")
    # trainer = train_on_responses_only(trainer, instruction_part="<|im_start|>user\n", response_part="<|im_start|>assistant\n")
    print("✅ Response-only 학습 모드")
    
    # 11. 학습
    # trainer.train()
    unsloth_train(trainer)
    
    # 12. 저장
    if config.get('merge_weights', False):
        final_dir = os.path.join(output_dir, "final_model")
        if local_rank <= 0:
            model.save_pretrained_merged(final_dir, tokenizer, save_method = "merged_16bit")
            print(f"\n✅ 완료! 모델: full_finetuning")
            print(f"\n✅ 경로: {final_dir}")
    else:
        final_dir = os.path.join(output_dir, "final_model_peft")
        if local_rank <= 0:
            model.save_pretrained(final_dir)
            # 저장 후 bfloat16 변환
            # adapter_path = os.path.join(final_dir, "adapter_model.safetensors")
            # if os.path.exists(adapter_path):
            #     tensors = load_file(adapter_path)
            #     tensors = {k: v.to(torch.bfloat16) for k, v in tensors.items()}
            #     save_file(tensors, adapter_path)
            tokenizer.save_pretrained(final_dir)
            print(f"\n✅ 완료! 모델: final_model_peft")
            print(f"\n✅ 경로: {final_dir}")
        
        
    
    
    if config.get('use_wandb') and local_rank <= 0:
        wandb.finish()


if __name__ == "__main__":
    main()
