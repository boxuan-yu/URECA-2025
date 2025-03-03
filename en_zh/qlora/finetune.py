import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
import bitsandbytes as bnb
from trl import SFTTrainer
from tqdm import tqdm
from transformers import TrainerCallback
from accelerate import Accelerator
from functools import partial


load_dotenv(find_dotenv())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"running on {device}")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)


def get_completion(dialogue, model, tokenizer):
    # Create the prompt with a clear delimiter
    prompt = tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": f"Summarize the given code-switched dialogue. \nDialogue: {dialogue}"}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)
    generated_ids = model.generate(
        **encoded,
        max_new_tokens=128,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

    input_ids_len = encoded["input_ids"].shape[1]
    generated_tokens = generated_ids[:, input_ids_len:]
    decoded_output = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    
    return decoded_output

def format_prompt(example, tokenizer):
    system_prompt = f"Summarize the given code-switched dialogue.\nDialogue: {example['en_zh_dialogue']}"
    text = tokenizer.apply_chat_template(
        conversation=[
            {"role": "user", "content": system_prompt},
            {"role": "assistant", "content": f"Summary: {example['summary']}"}
        ],
        tokenize=False
    )

    tokenized = tokenizer(text, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = tokenized["input_ids"][0]

    labels = input_ids.clone()

    prompt_length = len(
        tokenizer(system_prompt)["input_ids"]
    )
    labels[:prompt_length] = -100
    labels[labels == tokenizer.pad_token_id] = -100 

    example["input_ids"] = input_ids
    example["labels"] = labels

    return example


def find_all_linear_names(model):
  cls = bnb.nn.Linear4bit #if args.bits == 4 else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
  lora_module_names = set()
  for name, module in model.named_modules():
    if isinstance(module, cls):
      names = name.split('.')
      lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
      lora_module_names.remove('lm_head')
  return list(lora_module_names)


def get_and_save_predictions(benchmark_path, model, tokenizer, output_csv_path):
    benchmark_df = pd.read_csv(benchmark_path)
    benchmark_ds = Dataset.from_pandas(benchmark_df)

    predictions, references = [], []
    for example in tqdm(benchmark_ds):
        dialogue = example["cs_dialogue"]
        reference_summary = example["summary"]

        prediction = get_completion(dialogue, model, tokenizer)

        predictions.append(prediction)
        references.append(reference_summary)
    
    results_df = pd.DataFrame({
        'predictions': predictions,
        'references': references
    })
    results_df.to_csv(output_csv_path, index=False)
    print(f"Saved predictions and references to {output_csv_path}")


class EarlyStoppingCallback(TrainerCallback):
   
    def __init__(self, patience=3):
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
    
    def on_evaluate(self, args, state, control, metrics, **kwargs):
        eval_loss = metrics.get("eval_loss", float('inf'))
        if eval_loss < self.best_loss:
            self.best_loss = eval_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Early stopping triggered!")
                control.should_training_stop = True


def finetune(csv_path, adapter_path, input_max_length, output_max_length, model_id):

    accelerator = Accelerator()

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        add_eos_token=True,
        use_auth_token=os.environ["HF_TOKEN"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        use_auth_token=os.environ["HF_TOKEN"],
        attn_implementation="eager"
    )

    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model).to(device)
    modules = find_all_linear_names(model)
    print(f"Linear modules: {modules}")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}")

    if ".csv" in csv_path: 
        df = pd.read_csv(csv_path)
    else:
        df = pd.read_json(csv_path, lines=True)
    df['dial_len'] = df['en_zh_dialogue'].apply(lambda x: len(tokenizer(x)['input_ids']))
    df['sum_len'] = df['summary'].apply(lambda x: len(tokenizer(x)['input_ids']))
    df = df.drop(df[(df['dial_len']>input_max_length) | (df['sum_len']>output_max_length)].index, axis=0) 

    ds = Dataset.from_pandas(df).train_test_split(test_size=0.1,shuffle=True,seed=2406)
    print(ds)
    ds = ds.map(partial(format_prompt, tokenizer=tokenizer))
    ds = ds.remove_columns(['en_zh_dialogue', 'summary'])

    torch.cuda.empty_cache()

    # args = TrainingArguments(
    #     output_dir="outputs",
    #     per_device_train_batch_size=16,
    #     per_device_eval_batch_size=16,
    #     gradient_accumulation_steps=4,
    #     eval_steps=max(len(ds['train']) // (16 * 4), 100),
    #     evaluation_strategy="steps",
    #     num_train_epochs=3,
    #     warmup_ratio=0.1,
    #     max_grad_norm=1.0,
    #     learning_rate=3e-4,
    #     lr_scheduler_type="cosine",
    #     logging_steps=100,
    #     optim="paged_adamw_8bit",
    #     save_strategy="steps",
    #     save_total_limit=1,
    #     report_to=["none"],
    #     fp16=True,
    # )
    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_steps=200,
        eval_steps=50,
        evaluation_strategy="steps",
        num_train_epochs=5,
        learning_rate=3e-4,
        logging_steps=50,
        optim="paged_adamw_8bit",
        save_strategy="steps",
        save_total_limit=1,
        report_to=["none"],
        bf16=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        dataset_text_field="input_ids",
        peft_config=lora_config,
        args = args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(patience=5)],
    )
    model.config.use_cache = False
    model, trainer = accelerator.prepare(model, trainer)
    trainer.train()

    trainer.model.save_pretrained(adapter_path)

    del model


if __name__ == "__main__":
   
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--benchmark_path", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--input_max_length", type=int, required=True)
    parser.add_argument("--output_max_length", type=int, required=True)
    parser.add_argument("--train", action="store_true")


    args = parser.parse_args()
    if args.train:
        finetune(csv_path=args.csv_path,
                adapter_path=args.adapter_path,
                input_max_length=args.input_max_length,
                output_max_length=args.output_max_length,
                model_id=args.model_id)
    
    torch.cuda.empty_cache()
   
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map="cuda"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        add_eos_token=True,
        use_auth_token=os.environ["HF_TOKEN"]
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    
    print("Evaluating the finetuned model directly on the test set...")
    merged_model = PeftModel.from_pretrained(
        base_model, args.adapter_path
    )
    merged_model = merged_model.merge_and_unload()

    get_and_save_predictions(
        benchmark_path=args.benchmark_path,
        model=merged_model,
        tokenizer=tokenizer,
        output_csv_path=f"en_zh/qlora/preds/{args.model_id}.csv"
    )