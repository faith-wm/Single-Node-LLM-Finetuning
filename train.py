import datasets
import re
import time
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.backends import cuda
from torch import bfloat16, float16
import os
import deepspeed
from torch.utils.data import DataLoader

cuda.matmul.allow_tf32 = True

def main():
    model_id='Meta-Llama-3.1-70B-Instruct' #path to huggingface model
    tokenizer = AutoTokenizer.from_pretrained(model_id,use_fast=False)
    tokenizer.pad_token = tokenizer.eos_token

    timestamp = time.strftime("%Y%m%d_%H")
    filename=re.search(r'(?<=Llama-3\.1-)\d+B', model_id).group(0)
    out_dir = f'/scr1/users/mutindaf/finetuned_models/{filename}_{timestamp}'
    os.makedirs(out_dir, exist_ok=True)

    dataset=datasets.load_dataset('json', data_files="train_data.jsonl", split='train')  #tokenized training data with [input_ids, attention_mask, labels]
 
    training_args = TrainingArguments(
        gradient_checkpointing_kwargs={"use_reentrant": False},
        output_dir=out_dir,
        deepspeed='deepspeed_single_node.json',
        overwrite_output_dir=True,
        seed=42,
        do_eval=False,
        logging_strategy="steps",
        logging_steps=1000, 
        learning_rate=2e-5,
        warmup_steps=50,
        gradient_checkpointing=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        tf32=True,
        bf16=True,
        # fp16=True,
        weight_decay=0.1,
        push_to_hub=False,
        save_strategy="steps",
        num_train_epochs=20,
        save_steps=50,
        report_to="tensorboard",
        save_on_each_node=False,
        save_total_limit=5,
        optim="paged_adamw_32bit", # adamw_bnb_8bit (2 bytes), adafactor (4 bytes), paged_adamw_8bit can page out to CPU memory
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="flash_attention_2",
        torch_dtype=bfloat16
        )
    model.to('cuda')
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer)
    )

    trainer.train()
    trainer.save_model()
    print('Training DONE')

main()


