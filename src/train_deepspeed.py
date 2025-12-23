#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PEFT (LoRA) 파인튜닝 - SFTTrainer + DeepSpeed ZeRO

deepspeed --num_gpus=4 train_deepspeed.py --config config.json
"""

import os
import json
import torch
import argparse
import warnings
from datetime import datetime
from typing import Dict, List, Any

warnings.filterwarnings('ignore')

from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, TaskType, get_peft_model

import wandb


# ============================================================================
# Data Utils
# ============================================================================

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


def create_formatting_func(tokenizer):
    """Gemma3 Chat 형식으로 변환하는 formatting_func 생성 (단일 문자열 반환)"""
    def formatting_func(example):
        question = example.get('question', '')
        answer = example.get('answer', '')
        
        # 빈 값 체크
        if not question or not answer or not question.strip() or not answer.strip():
            return ""
        
        # 시스템 프롬프트 분리
        system_content = """다음 질문에 대해 정확하고 상세하게 답변해주세요.

규칙:
1. 질문의 핵심을 파악하여 명확하게 답변하세요.
2. 기술적인 용어가 있다면 정확하게 설명하세요.
3. 논리적이고 체계적인 답변을 작성하세요."""
        
        # 유저 컨텐츠는 질문만 포함 (테스트 시와 동일하게)
        user_content = question
        
        # Chat message 구성 (system, user, assistant 순서)
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer}
        ]
        
        # tokenizer.apply_chat_template으로 단일 문자열 반환
        formatted_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return formatted_text
    
    return formatting_func


# ============================================================================
# Config & Main
# ============================================================================

def load_config(config_path: str) -> Dict[str, Any]:
    """설정 파일 로드"""
    if not config_path or not os.path.exists(config_path):
        raise FileNotFoundError(f"설정 파일이 필요합니다: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='PEFT 학습')
    parser.add_argument('--config', type=str, required=True, help='설정 파일 경로')
    parser.add_argument('--deepspeed', type=str, help='DeepSpeed config 경로')
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
    
    # 1. 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(
        config['model_name'],
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    
    # Gemma3Processor인 경우 내부 토크나이저에서 eos_token 가져오기
    inner_tokenizer = getattr(tokenizer, 'tokenizer', tokenizer)
    
    # eos_token 설정 (setattr로 강제 설정)
    if not hasattr(tokenizer, 'eos_token') or tokenizer.eos_token is None:
        if hasattr(inner_tokenizer, 'eos_token') and inner_tokenizer.eos_token is not None:
            setattr(tokenizer, 'eos_token', inner_tokenizer.eos_token)
        else:
            setattr(tokenizer, 'eos_token', "<eos>")
    
    # pad_token 설정 (setattr로 강제 설정)
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        if hasattr(inner_tokenizer, 'pad_token') and inner_tokenizer.pad_token is not None:
            setattr(tokenizer, 'pad_token', inner_tokenizer.pad_token)
        else:
            setattr(tokenizer, 'pad_token', tokenizer.eos_token)
    
    print(f"✅ Tokenizer type: {type(tokenizer).__name__}")
    print(f"   eos_token: {tokenizer.eos_token}, pad_token: {tokenizer.pad_token}")
    
    # 2. 데이터셋 (question, answer 필드 유지)
    train_dataset = load_dataset_from_path(
        config['data_path'],
        config.get('max_samples')
    )
    print(f"✅ 학습 데이터: {len(train_dataset)}개")
    
    eval_dataset = None
    if config.get('eval_data_path') and os.path.exists(config['eval_data_path']):
        eval_dataset = load_dataset_from_path(config['eval_data_path'])
        print(f"✅ 평가 데이터: {len(eval_dataset)}개")
    elif config.get('validation_split', 0) > 0:
        split = train_dataset.train_test_split(test_size=config['validation_split'], seed=42)
        train_dataset, eval_dataset = split['train'], split['test']
        print(f"✅ 학습/검증 분할: {len(train_dataset)}/{len(eval_dataset)}")
    
    # 3. 모델 + LoRA
    print(f"🔄 모델 로딩 중: {config['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        config['model_name'],
        attn_implementation="eager",
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=None,
        low_cpu_mem_usage=True,
        # is_training=True, 
    )
    print("✅ 모델 로딩 완료")
    model.config.use_cache = False
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config['lora_r'],
        lora_alpha=config['lora_alpha'],
        lora_dropout=config.get('lora_dropout', 0.05),
        target_modules=config['target_modules'],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()
    
    # 4. wandb
    if config.get('use_wandb') and local_rank <= 0:
        wandb.init(
            project=config.get('wandb_project', 'peft'),
            name=f"{os.path.basename(config['model_name'])}_{timestamp}",
            config=config
        )
    
    # ============================================================================
    # 5. Warmup Steps 자동 계산
    # ============================================================================
    WARMUP_RATIO = config.get('warmup_ratio', 0.05)  # config에서 가져오거나 기본값 5%
    
    # GPU 개수 (DeepSpeed 환경에서는 WORLD_SIZE 사용)
    num_gpus = int(os.environ.get("WORLD_SIZE", 1))
    micro_batch_size = config['batch_size']
    grad_accum_steps = config.get('gradient_accumulation_steps', 1)
    num_epochs = config['num_epochs']
    total_samples = len(train_dataset)
    
    # 글로벌 배치 크기 계산
    global_batch_size = micro_batch_size * grad_accum_steps * num_gpus
    
    # 총 학습 스텝 수 계산 (올림 처리)
    if global_batch_size == 0:
        total_steps = 0
    else:
        steps_per_epoch = (total_samples + global_batch_size - 1) // global_batch_size
        total_steps = steps_per_epoch * num_epochs
    
    # Warmup Steps 계산
    calculated_warmup_steps = int(total_steps * WARMUP_RATIO)
    
    if local_rank <= 0:
        print(f"--- Warmup Steps 자동 계산 결과 ---")
        print(f"  GPU 개수: {num_gpus}")
        print(f"  글로벌 배치 크기: {global_batch_size}")
        print(f"  총 학습 스텝 수: {total_steps} (Epoch: {num_epochs})")
        print(f"  Warmup Steps: {calculated_warmup_steps} ({WARMUP_RATIO*100}%)")
        print(f"------------------------------------")
    
    # 6. SFTConfig
    output_dir = os.path.join(config['output_dir'], f"peft_{timestamp}")
    
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config['num_epochs'],
        per_device_train_batch_size=config['batch_size'],
        per_device_eval_batch_size=config['batch_size'],
        gradient_accumulation_steps=config.get('gradient_accumulation_steps', 1),
        learning_rate=config['learning_rate'],
        warmup_steps=calculated_warmup_steps,
        logging_steps=config.get('logging_steps', 10),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        load_best_model_at_end=True,
        save_steps=config.get('save_steps', 100),
        eval_steps=config.get('eval_steps', 100) if eval_dataset else None,
        eval_strategy="steps" if eval_dataset else "no",
        save_strategy="steps",
        bf16=True,
        optim=config.get('optim', 'adamw_torch'),
        lr_scheduler_type=config.get('lr_scheduler_type', 'cosine'),
        loss_type=config.get('loss_type', 'dft'),
        weight_decay=0.01,
        max_grad_norm=config.get('max_grad_norm', 1.0),
        save_total_limit=3,
        dataloader_drop_last=True,
        group_by_length=False,
        packing=True,
        remove_unused_columns=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        deepspeed=config.get('deepspeed'),
        local_rank=args.local_rank,
        report_to="wandb" if config.get('use_wandb') else None,
        max_length=config.get('max_length', 2048),
        # use_liger_kernel=True
    )
    
    # 고정 길이 패딩 DataCollator (DeepSpeed ZeRO-3 호환성)
    # max_seq_len = config.get('max_length', 2048)
    # data_collator = DataCollatorForSeq2Seq(
    #     tokenizer,
    #     padding="max_length",      # 핵심: 무조건 max_length로 패딩
    #     max_length=max_seq_len,    # 고정 길이 지정
    #     pad_to_multiple_of=8,      # 성능 최적화
    #     return_tensors="pt"
    # )
    
    # 7. 학습 (formatting_func 사용 - 단일 문자열 반환)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # tokenizer=tokenizer,
        processing_class=tokenizer,
        # data_collator=data_collator,  # 고정 길이 collator 전달
        formatting_func=create_formatting_func(tokenizer),
        # model_init_kwargs={"_compute_loss": True},
    )
    
    trainer.train()
    
    # 8. 저장
    final_dir = os.path.join(output_dir, "final_model")
    trainer.save_model(final_dir)
    if local_rank <= 0:
        tokenizer.save_pretrained(final_dir)
        print(f"\n✅ 완료! 모델: {final_dir}")
    
    if config.get('use_wandb') and local_rank <= 0:
        wandb.finish()


if __name__ == "__main__":
    main()
