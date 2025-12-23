#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🎯 PEFT (LoRA) 기반 QA 모델 고효율 파인튜닝 - SFTTrainer + DeepSpeed ZeRO

【주요 개선사항】
- SFTTrainer 사용으로 대화형 데이터 처리 최적화
- Gemma3 공식 Chat Template 적용
- 🆕 규칙 기반 수동 레이블링 (response만 학습)
- 🚀 DeepSpeed ZeRO-2/3 지원으로 대규모 모델 학습
- 모듈화된 구조로 코드 가독성 및 재사용성 향상 (모델 로드 로직 분리)
- wandb 로깅 지원

【레이블링 방식】
- prompt 부분: labels = -100 (학습 안 함)
- response 부분: labels = token_ids (학습함)

【데이터 형식】
- JSONL 포맷: {"qas": [{"question": "...", "answer": "...", "question_type": [...], "difficulty": "..."}]}

【사용 예시】
# 단일 GPU (4bit 양자화)
python train_deepspeed.py --model /path/to/model --epochs 3 --batch_size 16

# DeepSpeed 멀티 GPU (BF16, ZeRO-2)
deepspeed --num_gpus=4 train_deepspeed.py \\
    --model /path/to/model --deepspeed ds_config_zero2.json \\
    --epochs 3 --batch_size 8

# DeepSpeed 멀티 GPU (BF16, ZeRO-3)
deepspeed --num_gpus=4 train_deepspeed.py \\
    --model /path/to/model --deepspeed ds_config_zero3.json \\
    --epochs 3 --batch_size 8

⚠️ 주의: DeepSpeed ZeRO-3 사용 시 4bit 양자화 및 device_map은 비활성화됩니다 (충돌 방지)
"""

import os
import json
import torch
import argparse
import random
import re
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Any, Optional

# 경고 무시 설정
warnings.filterwarnings('ignore')

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl
)
from trl import SFTTrainer
from peft import (
    LoraConfig,
    TaskType,
    PeftModel
)

# DeepSpeed ZeRO support 확인
try:
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    DEEPSPEED_AVAILABLE = True
except ImportError:
    DEEPSPEED_AVAILABLE = False

# wandb support 확인
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False



# ============================================================================
# DeepSpeed ZeRO Utility Functions
# ============================================================================

def maybe_zero_3(param: torch.Tensor) -> torch.Tensor:
    """DeepSpeed ZeRO-3에서 파라미터를 수집하는 유틸리티 함수"""
    if DEEPSPEED_AVAILABLE and hasattr(param, "ds_id"):
        assert param.ds_status == ZeroParamStatus.NOT_AVAILABLE
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params: List[tuple], bias: str) -> Dict[str, torch.Tensor]:
    """PEFT 모델 상태를 DeepSpeed ZeRO-3와 호환되게 저장"""
    # 원본 로직 유지 (DeepSpeed 유틸리티)
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if k in lora_bias_names:
                to_return[k] = t
    else:
        raise NotImplementedError(f"bias={bias} is not implemented")
    
    to_return = {k: maybe_zero_3(v) for k, v in to_return.items()}
    return to_return

# ============================================================================
# Data Collator and Metrics
# ============================================================================

class DataCollatorForCompletionOnly:
    """답변 부분만 학습하도록 하는 Custom Data Collator"""
    
    RESPONSE_TEMPLATE = "<start_of_turn>model\n"
    
    def __init__(self, tokenizer: AutoTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # response_template을 토큰화하여 ID 시퀀스 얻기
        self.response_token_ids = self.tokenizer.encode(
            self.RESPONSE_TEMPLATE, 
            add_special_tokens=False
        )
        print(f"📌 Response Template: '{self.RESPONSE_TEMPLATE}'")
        print(f"📌 Response Token IDs: {self.response_token_ids}")
        print(f"📌 Max Length: {self.max_length}")
        
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """배치 데이터를 collate하고 labels 마스킹"""
        
        # 텍스트 추출 (SFTTrainer의 formatting_func이 'text' 키에 결과를 저장)
        texts = [example['text'] for example in examples if 'text' in example]
        
        # 토큰화 (Gemma3 요구사항: token_type_ids 추가)
        batch = self.tokenizer(
            texts,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt"
        )
        
        # token_type_ids가 없으면 생성 (Gemma3 필수)
        if "token_type_ids" not in batch:
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
        
        # Labels 생성 (input_ids 복사)
        labels = batch["input_ids"].clone()
        
        response_len = len(self.response_token_ids)
        
        # 디버깅용 카운터
        found_count = 0
        not_found_count = 0
        
        # 각 샘플에 대해 response 시작 위치 찾기 및 마스킹
        for idx in range(len(labels)):
            input_ids = batch["input_ids"][idx].tolist()
            
            # response_template 토큰 시퀀스를 찾기
            try:
                # 템플릿의 시작 인덱스를 찾고, 템플릿 길이만큼 더해 응답 시작점 확보
                # Python list index()는 O(N)이지만, 응답 시작점은 보통 시퀀스 앞쪽에 위치하여 빠름
                template_start_idx = -1
                for i in range(len(input_ids) - response_len + 1):
                    if input_ids[i:i + response_len] == self.response_token_ids:
                        template_start_idx = i
                        break
                
                if template_start_idx != -1:
                    # response_template 이후부터 학습 (템플릿 자체는 제외)
                    response_start_idx = template_start_idx + response_len
                    labels[idx, :response_start_idx] = -100
                    found_count += 1
                else:
                    # template을 못 찾은 경우 전체 마스킹 (학습 안 함)
                    labels[idx, :] = -100
                    not_found_count += 1
                    # 처음 2개만 디버깅 출력
                    if not_found_count <= 2:
                        decoded_full = self.tokenizer.decode(input_ids)  # 전체 텍스트
                        print(f"⚠️  Response Template 못 찾음 (샘플 {idx})")
                        print(f"   ===== 전체 텍스트 =====")
                        print(decoded_full)
                        print(f"   ===== 끝 =====")
                        print(f"   찾는 템플릿: {self.RESPONSE_TEMPLATE}")
                        print(f"   템플릿 토큰 IDs: {self.response_token_ids}")
                        print(f"   텍스트에 템플릿 포함 여부: {self.RESPONSE_TEMPLATE in decoded_full}")
            
            except Exception as e:
                # 오류 발생 시 전체 마스킹 (안전 장치)
                labels[idx, :] = -100
                print(f"❌ 오류 발생 (샘플 {idx}): {e}")
            
            # padding 토큰도 -100으로 마스킹
            labels[idx][labels[idx] == self.tokenizer.pad_token_id] = -100
        
        # 배치 통계 출력 (가끔씩만)
        if not_found_count > 0:
            print(f"📊 배치 통계: 템플릿 찾음={found_count}, 못 찾음={not_found_count}")
        
        batch["labels"] = labels
        
        return batch


# ============================================================================
# Trainer Components
# ============================================================================

class PerplexityLoggingCallback(TrainerCallback):
    """Train/Eval loss를 perplexity로 변환하여 로깅하는 Callback"""
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float] = None, **kwargs):
        """로깅 시 perplexity 추가"""
        if logs is None:
            return
        
        # train_loss가 있으면 train_perplexity 계산
        if 'loss' in logs:
            import math
            logs['train_perplexity'] = math.exp(logs['loss'])
        
        # eval_loss가 있으면 eval_perplexity 계산
        if 'eval_loss' in logs:
            import math
            logs['eval_perplexity'] = math.exp(logs['eval_loss'])


class QAPEFTTrainer:
    """PEFT를 사용한 QA 파인튜닝 트레이너"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[Any] = None
        self.dataset: Optional[Dataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.eval_dataset: Optional[Dataset] = None
        self.peft_config: Optional[LoraConfig] = None
        
    def setup_model_and_tokenizer(self):
        """모델과 토크나이저 설정 (LoRA Config 포함)"""
        print(f"🤖 모델 및 토크나이저 설정 중: {self.config['model_name']}")
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['model_name'],
            trust_remote_code=True,
            padding_side="right",
            use_fast=False
        )
        
        # Gemma3 pad_token 설정 (EOS 토큰 사용 - ZeRO-3 호환)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        print(f"📌 PAD token ID: {self.tokenizer.pad_token_id}, EOS token ID: {self.tokenizer.eos_token_id}")
        
        # LoRA 설정
        self.peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.config['lora_r'],
            lora_alpha=self.config['lora_alpha'],
            lora_dropout=self.config['lora_dropout'],
            target_modules=self.config['target_modules']
        )
        
        print("✅ 토크나이저 및 PEFT 설정 완료")

    @staticmethod
    def _is_valid_example(item: Dict[str, Any]) -> bool:
        """필수 필드 존재 여부 확인: input.question, output.answer"""
        try:
            if not isinstance(item, dict):
                return False
            inp = item.get('input')
            out = item.get('output')
            if not isinstance(inp, dict) or not isinstance(out, dict):
                return False
            question = inp.get('question')
            answer = out.get('answer')
            if not isinstance(question, str) or not question.strip():
                return False
            if not isinstance(answer, str) or not answer.strip():
                return False
            return True
        except Exception:
            return False

    @staticmethod
    def _normalize_example(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """외부 스키마를 내부 표준(input/output)으로 변환
        
        지원 형식:
        1. {"qas": [{"question": "...", "answer": "...", ...}]}  (현재 ARMS QA 데이터)
        2. {"question": "...", "answer": "..."}  (단순 QA 형식)
        3. {"input": {"question": "..."}, "output": {"answer": "..."}}  (내부 표준)
        """
        if QAPEFTTrainer._is_valid_example(item):
            return item
        
        # 케이스1: {"qas": [{"question": "...", "answer": "...", ...}]} (ARMS QA 형식)
        if isinstance(item, dict) and 'qas' in item:
            qas_list = item.get('qas', [])
            if not isinstance(qas_list, list) or len(qas_list) == 0:
                return None
            
            # 첫 번째 QA 쌍 사용
            qa = qas_list[0]
            if not isinstance(qa, dict):
                return None
            
            question = qa.get('question', '')
            answer = qa.get('answer', '')
            
            if not question or not answer:
                return None
            
            return {
                'input': {
                    'question': question
                },
                'output': {
                    'answer': answer
                },
                'metadata': {
                    'question_type': qa.get('question_type', []),
                    'difficulty': qa.get('difficulty', '')
                }
            }
        
        # 케이스2: {"question": "...", "answer": "..."} (단순 QA 형식)
        if isinstance(item, dict) and 'question' in item and 'answer' in item:
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if not question or not answer:
                return None
            
            return {
                'input': {
                    'question': question
                },
                'output': {
                    'answer': answer
                },
                'metadata': {
                    'question_type': item.get('question_type', []),
                    'difficulty': item.get('difficulty', '')
                }
            }
        
        return None

    @staticmethod
    def _prepare_qa_data(item: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """QA 데이터에서 프롬프트에 필요한 요소를 준비"""
        question = item['input']['question']
        answer = item['output']['answer']
        
        # Gemma3 Chat Template User Content
        user_content = f"""다음 질문에 대해 정확하고 상세하게 답변해주세요.

질문:
{question}

---

규칙:
1. 질문의 핵심을 파악하여 명확하게 답변하세요.
2. 기술적인 용어가 있다면 정확하게 설명하세요.
3. 논리적이고 체계적인 답변을 작성하세요."""
        
        assistant_content = answer

        return {
            "user_content": user_content,
            "assistant_content": assistant_content,
            "reference_answer": answer  # 원본 답변 (메트릭 계산용)
        }

    def formatting_func(self, example: Dict[str, Any]) -> str:
        """SFTTrainer용 포맷팅 함수 - 전체 텍스트 반환 (동적 truncation 적용)"""
        try:
            formatted_data = self._prepare_qa_data(example, self.config)
            
            # Gemma3 포맷으로 조합
            # prompt에 model turn 시작까지 포함하여, assistant_content만 학습되도록 함
            user_content = formatted_data['user_content']
            assistant_content = formatted_data['assistant_content']
            
            # 고정 템플릿 (자르면 안 됨)
            prefix = "<bos><start_of_turn>user\n"
            middle = "<end_of_turn>\n<start_of_turn>model\n"
            suffix = "<end_of_turn>\n"
            
            response = assistant_content + suffix
            
            max_length = self.config.get('max_length', 512)
            
            # 각 부분의 토큰 길이 계산
            prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
            middle_tokens = self.tokenizer.encode(middle, add_special_tokens=False)
            response_tokens = self.tokenizer.encode(response, add_special_tokens=False)
            user_content_tokens = self.tokenizer.encode(user_content, add_special_tokens=False)
            
            # 고정 토큰 길이
            fixed_length = len(prefix_tokens) + len(middle_tokens) + len(response_tokens)
            available_for_user = max_length - fixed_length
            
            # user_content가 너무 길면 truncation
            if len(user_content_tokens) > available_for_user:
                if available_for_user > 50:  # 최소 길이 확보
                    # 앞부분만 유지 (뒤에서 자름)
                    truncated_tokens = user_content_tokens[:available_for_user]
                    user_content = self.tokenizer.decode(truncated_tokens, skip_special_tokens=False)
                    
                    # 디버깅 출력 (처음 몇 개만)
                    if not hasattr(self, '_truncation_count'):
                        self._truncation_count = 0
                    if self._truncation_count < 3:
                        print(f"✂️  User content truncated: {len(user_content_tokens)} → {available_for_user} tokens")
                        self._truncation_count += 1
                else:
                    # available 공간이 너무 작으면 경고
                    if not hasattr(self, '_warning_count'):
                        self._warning_count = 0
                    if self._warning_count < 3:
                        print(f"⚠️  Response가 너무 깁니다. max_length 증가 필요: response={len(response_tokens)}, max={max_length}")
                        self._warning_count += 1
            
            # 최종 텍스트 조합
            full_text = prefix + user_content + middle + response
            
            return full_text
            
        except KeyError as e:
            raise ValueError(f"필수 키가 누락되었습니다: {e}")
    
    def load_and_prepare_dataset(self):
        """데이터셋 로드 및 전처리 (스키마 정규화 포함)"""
        print(f"📊 데이터셋 로드 중: {self.config['data_path']}")
        
        data = []
        # JSONL 또는 JSON 파일 로드
        if self.config['data_path'].endswith('.jsonl'):
            with open(self.config['data_path'], 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        else:
            with open(self.config['data_path'], 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        # 데이터 샘플링 (옵션)
        if self.config.get('max_samples') and len(data) > self.config['max_samples']:
            data = data[:self.config['max_samples']]
            print(f"📋 데이터 샘플링: {len(data)}개 사용")
        
        # 스키마 정규화 및 필터링 (학습 데이터)
        normalized_data = []
        converted_cnt = 0
        for ex in data:
            norm = self._normalize_example(ex)
            if norm is None:
                continue
            if norm is not ex:
                converted_cnt += 1
            normalized_data.append(norm)
        dropped = len(data) - len(normalized_data)
        if converted_cnt > 0:
            print(f"🔧 학습 데이터 스키마 자동 변환: {converted_cnt}개")
        if dropped > 0:
            print(f"⚠️ 스키마 불일치로 제외된 학습 샘플: {dropped}개")
        data = normalized_data
        print(f"✅ 데이터 준비 완료: {len(data)}개")
        
        self.dataset = Dataset.from_list(data)
        
        # 별도 평가 데이터 경로가 제공되면 그 파일을 평가 데이터로 사용 (정규화 포함)
        eval_data_path = self.config.get('eval_data_path')
        if eval_data_path and os.path.exists(eval_data_path):
            print(f"📊 평가 데이터셋 로드 중: {eval_data_path}")
            eval_data = []
            if eval_data_path.endswith('.jsonl'):
                with open(eval_data_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            eval_data.append(json.loads(line))
            else:
                with open(eval_data_path, 'r', encoding='utf-8') as f:
                    eval_data = json.load(f)

            normalized_eval = []
            converted_eval_cnt = 0
            for ex in eval_data:
                norm = self._normalize_example(ex)
                if norm is None:
                    continue
                if norm is not ex:
                    converted_eval_cnt += 1
                normalized_eval.append(norm)
            dropped_eval = len(eval_data) - len(normalized_eval)
            if converted_eval_cnt > 0:
                print(f"🔧 평가 데이터 스키마 자동 변환: {converted_eval_cnt}개")
            if dropped_eval > 0:
                print(f"⚠️ 스키마 불일치로 제외된 평가 샘플: {dropped_eval}개")

            self.train_dataset = self.dataset
            self.eval_dataset = Dataset.from_list(normalized_eval)
            print(f"📊 학습 데이터: {len(self.train_dataset)}개, 평가 데이터: {len(self.eval_dataset)}개 (외부 파일)")
        else:
            # 학습/검증 분할
            if self.config['validation_split'] > 0:
                split_dataset = self.dataset.train_test_split(
                    test_size=self.config['validation_split'],
                    seed=42
                )
                self.train_dataset = split_dataset['train']
                self.eval_dataset = split_dataset['test']
                print(f"📊 학습 데이터: {len(self.train_dataset)}개, 검증 데이터: {len(self.eval_dataset)}개")
            else:
                self.train_dataset = self.dataset
                self.eval_dataset = None
                print(f"📊 학습 데이터: {len(self.train_dataset)}개 (검증 없음)")

    def setup_training_arguments(self, timestamp: str) -> TrainingArguments:
        """학습 인자 설정"""
        output_dir = os.path.join(
            self.config['output_dir'],
            f"qa_peft_{timestamp}"
        )
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.config['num_epochs'],
            per_device_train_batch_size=self.config['batch_size'],
            per_device_eval_batch_size=self.config['batch_size'],
            gradient_accumulation_steps=self.config['gradient_accumulation_steps'],
            learning_rate=self.config['learning_rate'],
            warmup_steps=self.config.get('warmup_steps', 0),
            warmup_ratio=self.config.get('warmup_ratio', None),
            logging_steps=self.config['logging_steps'],
            save_steps=self.config['save_steps'],
            eval_steps=self.config['eval_steps'] if self.eval_dataset else None,
            eval_strategy="steps" if self.eval_dataset else "no",
            save_strategy="steps",
            load_best_model_at_end=False,  # DeepSpeed ZeRO-3 호환성: checkpoint 로드 실패 방지
            metric_for_best_model="eval_loss" if self.eval_dataset else None,
            greater_is_better=False,
            save_only_model=True,  # LoRA adapter 저장을 위해 True로 설정
            fp16=False,
            bf16=True,  # BF16 사용 (Zero-3 최적화)
            optim="adamw_torch",
            lr_scheduler_type=self.config.get('lr_scheduler_type', 'cosine'),
            eval_accumulation_steps=4,
            weight_decay=0.01,
            max_grad_norm=1.0,
            save_total_limit=3,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            group_by_length=True,
            remove_unused_columns=False,
            push_to_hub=False,
            # hub_token=None,  # TRL 0.25.0.dev0 호환성 (push_to_hub_token → hub_token)
            gradient_checkpointing=True,
            deepspeed=self.config.get('deepspeed'),
            local_rank=self.config.get('local_rank', -1),
            report_to="wandb" if (self.config['use_wandb'] and WANDB_AVAILABLE) else None,
            run_name=self.config.get('wandb_run_name', f"rag_peft_{timestamp}") if (self.config['use_wandb'] and WANDB_AVAILABLE) else None
        )
        
        return training_args

    def _load_model(self) -> Any:
        """모델 로드 및 DeepSpeed 설정 처리 (deepspeed_ref.py 방식 적용)"""
        print("🤖 모델 로드 중...")
        
        # 💡 토크나이저 설정이 이미 완료되어야 함 (pad_token_id 확보)
        if self.tokenizer is None:
             raise ValueError("토크나이저가 설정되지 않았습니다. setup_model_and_tokenizer를 먼저 호출하세요.")

        # DeepSpeed config에 따라 precision 결정
        torch_dtype = torch.bfloat16  # BF16 사용 (Zero-3 최적화)
        print("  - BF16 precision (양자화 없음)")
        
        # 🔧 ZeRO-3 호환 모델 로드 (Gemma3에서 검증된 방법)
        print("  - 모델 로드 중 (ZeRO-3 호환)...")
        
        # ZeRO-3에서는 low_cpu_mem_usage와 device_map을 비활성화 (GPT 검증 방식)
        model = AutoModelForCausalLM.from_pretrained(
            self.config['model_name'],
            attn_implementation="eager",
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=False,  # ← ZeRO-3 필수! Accelerate 간섭 방지
            device_map=None,          # ← ZeRO-3 필수! DeepSpeed가 장치 관리
        )
        print("  - 모델 로드 성공!")
        
        # use_cache를 config에서 설정
        model.config.use_cache = False
        print("  - use_cache=False 설정 완료")
        
        # 🔧 LoRA 적용 (get_peft_model 사용)
        print("  - LoRA 적용 중...")
        from peft import get_peft_model
        model = get_peft_model(model, self.peft_config)
        print("  - LoRA 적용 완료!")
        
        # 🔧 gradient_checkpointing 활성화 시 enable_input_require_grads() 호출
        if self.config.get('gradient_checkpointing', True):
            model.enable_input_require_grads()
            print("  - gradient_checkpointing 활성화")
        
        print("✅ 모델 로드 완료")
        return model
        
    def train(self) -> Dict[str, Any]:
        """모델 학습 (SFTTrainer 사용)"""
        print("🚀 학습 시작 (SFTTrainer)")
        
        # 통일된 timestamp 생성 (wandb, output_dir에서 모두 사용)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # wandb 초기화 로직 (rank 0만 실행)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE and local_rank <= 0:
            model_name = os.path.basename(self.config['model_name'])
            run_name = self.config.get('wandb_run_name') or f"{model_name}_{timestamp}"
            
            wandb.init(
                project=self.config.get('wandb_project', 'arms_qa'),
                name=run_name,
                config={
                    'model': self.config['model_name'],
                    'lora_r': self.config['lora_r'],
                    'batch_size': self.config['batch_size'],
                    'learning_rate': self.config['learning_rate'],
                    'max_length': self.config['max_length']
                },
                tags=['QA', 'PEFT', 'LoRA', 'Gemma3']
            )
            print(f"📊 wandb 초기화 완료 (Rank {local_rank}): {self.config.get('wandb_project')}/{run_name}")
        
        # 💡 중요: DeepSpeed ZeRO-3에서는 TrainingArguments를 먼저 생성해야 함
        training_args = self.setup_training_arguments(timestamp=timestamp)
        
        # 모델 로드 (TrainingArguments 생성 후)
        model = self._load_model()
        
        # DataCollator 설정
        collator = DataCollatorForCompletionOnly(
            tokenizer=self.tokenizer,
            max_length=self.config['max_length'],
        )
        
        # Callbacks 설정 - 종합 평가 비활성화 (eval_loss만 사용)
        # Perplexity 로깅 콜백은 항상 추가
        callbacks = [PerplexityLoggingCallback()]
        metrics_callback = None
        
        # SFTTrainer 생성 (peft_config 제거 - 이미 모델에 LoRA 적용됨)
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            processing_class=self.tokenizer,
            formatting_func=self.formatting_func,
            data_collator=collator,
            callbacks=callbacks,
            # push_to_hub_token=None,  # TRL 일부 버전에서 무조건 pop 하므로 기본값 제공
        )
        
        # 학습 실행
        trainer.train()
        
        # 최종 모델 저장 및 결과 반환 로직 유지
        best_eval_loss = trainer.state.best_metric if hasattr(trainer.state, 'best_metric') and trainer.state.best_metric is not None else None
        model_folder_name = f"best_model_eval_loss_{best_eval_loss:.4f}" if best_eval_loss is not None else "best_model"
        
        final_output_dir = os.path.join(training_args.output_dir, model_folder_name)
        
        # DeepSpeed ZeRO-3: 모든 rank에서 save_model 호출 필요 (파라미터 수집)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        print(f"💾 모델 저장 중 (Rank {local_rank})...")
        trainer.save_model(final_output_dir)
        
        # tokenizer 저장은 rank 0에서만
        if local_rank <= 0:
            self.tokenizer.save_pretrained(final_output_dir)
            print(f"✅ 학습 완료! Eval Loss 기준 모델: {final_output_dir}")
        else:
            print(f"✅ Rank {local_rank}: 모델 저장 완료")
        
        result = {
            'eval_loss_model': final_output_dir,
            'best_model': final_output_dir,
            'best_eval_loss': best_eval_loss,
            'trained_model': trainer.model,
            'tokenizer': self.tokenizer,
            'training_args': training_args
        }
        
        # wandb 종료 (rank 0만 실행)
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if self.config.get('use_wandb', False) and WANDB_AVAILABLE and local_rank <= 0:
            wandb.finish()
            print("📊 wandb 로깅 완료")
            
        return result
    


def load_config(config_path: str = None) -> Dict[str, Any]:
    """설정 파일 로드 (로직 유지)"""
    # ... (기존 로직 유지) ...
    if config_path is None:
        config_path = "config/rag_peft_config.json"
    
    if not os.path.exists(config_path):
        print(f"⚠️ 설정 파일이 없습니다: {config_path}")
        print("기본 설정을 사용합니다.")
        return get_default_config()
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"✅ 설정 파일 로드: {config_path}")
        return config
    except Exception as e:
        print(f"❌ 설정 파일 로드 실패: {e}")
        print("기본 설정을 사용합니다.")
        return get_default_config()


def get_default_config():
    """기본 설정 반환"""
    return {
        # 모델 설정
        'model_name': '/home/rex/workspace/arms_qa/models/gemma-3-27b-it',
        
        # LoRA 설정
        'lora_r': 16,
        'lora_alpha': 32,
        'lora_dropout': 0.05,
        'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        
        # 데이터 설정
        'data_path': '/home/rex/workspace/arms_qa/data/train.jsonl',
        'eval_data_path': '/home/rex/workspace/arms_qa/data/test.jsonl',
        'max_length': 1024,  # QA 태스크는 더 긴 컨텍스트 필요
        'max_samples': None,
        'validation_split': 0.1,
        
        # 학습 설정
        'num_epochs': 3,
        'batch_size': 4,
        'gradient_accumulation_steps': 8,
        'learning_rate': 2e-4,
        'warmup_steps': 10,
        'logging_steps': 10,
        'save_steps': 100,
        'eval_steps': 100,
        
        # 출력 설정
        'output_dir': '/home/rex/workspace/arms_qa/output',
        'deepspeed': None,
        'local_rank': -1,
        'use_wandb': True,
        'wandb_project': 'arms_qa',
        'wandb_run_name': None,
    }

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='QA PEFT 학습')
    parser.add_argument('--config', type=str, help='설정 파일 경로 (JSON)')
    parser.add_argument('--model', type=str, help='모델 이름')
    parser.add_argument('--data', type=str, help='데이터 파일 경로')
    parser.add_argument('--epochs', type=int, help='학습 에포크 수')
    parser.add_argument('--eval_data', type=str, help='평가 데이터 파일 경로')
    parser.add_argument('--batch_size', type=int, help='배치 크기')
    parser.add_argument('--max_samples', type=int, help='최대 샘플 수')
    parser.add_argument('--deepspeed', type=str, default=None, help='DeepSpeed config 파일 경로 (멀티 GPU 학습 시)')
    parser.add_argument('--local_rank', type=int, default=-1, help='DeepSpeed local rank (자동 설정됨)')
    
    args = parser.parse_args()
    
    config = load_config(args.config)
    
    if args.deepspeed:
        config['deepspeed'] = args.deepspeed
    if args.local_rank != -1:
        config['local_rank'] = args.local_rank
    
    # 명령행 인자로 설정 오버라이드
    if args.model:
        config['model_name'] = args.model
    if args.data:
        config['data_path'] = args.data
    if args.eval_data:
        config['eval_data_path'] = args.eval_data
    if args.epochs:
        config['num_epochs'] = args.epochs
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.max_samples:
        config['max_samples'] = args.max_samples
    
    print("🚀 QA PEFT 학습 시작")
    print("=" * 70)
    print(f"모델: {config['model_name']}")
    print(f"데이터: {config['data_path']}")
    if config.get('eval_data_path'):
        print(f"평가 데이터: {config['eval_data_path']}")
    print(f"에포크: {config['num_epochs']}")
    print(f"배치 크기: {config['batch_size']}")
    print(f"LoRA r: {config['lora_r']}")
    print(f"Max Length: {config['max_length']}")
    print("=" * 70)
    
    try:
        # 전체 학습 실행
        trainer = QAPEFTTrainer(config)
        trainer.setup_model_and_tokenizer()
        trainer.load_and_prepare_dataset()
        train_result = trainer.train()
        
        # 결과 처리 로직 유지
        if isinstance(train_result, dict):
            eval_loss_model = train_result.get('eval_loss_model')
            best_model = train_result.get('best_model')
            trained_model = train_result.get('trained_model')
            tokenizer = train_result.get('tokenizer')
            
            if not best_model:
                training_args = train_result.get('training_args')
                best_model = os.path.join(training_args.output_dir, "best_model") if training_args else os.path.join(config['output_dir'], "best_model")
            
            print(f"\n📊 학습 결과:")
            if train_result.get('best_eval_loss'):
                print(f"  - Eval Loss: {train_result['best_eval_loss']:.4f} → {eval_loss_model}")
            print(f"  - 모델 저장: {best_model}")

            # 평가 메서드 제거됨 (ComprehensiveMetricsBestModelCallback 제거로 인해)
            eval_result = None
            
            result_file = best_model.replace('best_final', 'eval_result') + '.json'
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'config': {'epochs': config['num_epochs'], 'batch_size': config['batch_size'], 'lora_r': config['lora_r'], 'max_length': config['max_length']},
                    'eval_result': eval_result,
                    'train_result': {k:v for k,v in train_result.items() if k not in ['trained_model', 'tokenizer', 'training_args']} # 불필요한 객체 제외
                }, f, indent=2, ensure_ascii=False)
            print(f"\n📁 평가 결과 저장: {result_file}")
            
            # 추론 테스트 메서드 제거됨 (ComprehensiveMetricsBestModelCallback 제거로 인해)
            print("⚠️ 추론 테스트는 별도로 실행하세요.")
        else:
            print("⚠️ 학습 결과 객체를 찾을 수 없습니다.")
        
        print("\n✅ 모든 작업 완료!")
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
