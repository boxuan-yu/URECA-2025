import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from transformers import DataCollatorForLanguageModeling
from dotenv import load_dotenv, find_dotenv
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTTrainer
from tqdm import tqdm
from transformers import TrainerCallback
from accelerate import Accelerator
from functools import partial

load_dotenv(find_dotenv())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"running on {device}")


def get_completion(dialogue, model, tokenizer):
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
    prompt_length = len(tokenizer(system_prompt)["input_ids"])
    labels[:prompt_length] = -100
    labels[labels == tokenizer.pad_token_id] = -100 
    example["input_ids"] = input_ids
    example["labels"] = labels
    return example


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


def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[-1])
    
    lora_module_names.discard('lm_head')
    
    return list(sorted(lora_module_names)) 


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

    tokenizer = AutoTokenizer.from_pretrained(model_id, add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.bfloat16,
        attn_implementation="eager").to(device)
    model.gradient_checkpointing_enable()
    
    target_modules = find_all_linear_names(model=model)
    print(f"lora modules: {target_modules}")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable: {trainable} | total: {total} | Percentage: {trainable/total*100:.4f}")

    df = pd.read_csv(csv_path) if ".csv" in csv_path else pd.read_json(csv_path, lines=True)
    df['dial_len'] = df['en_zh_dialogue'].apply(lambda x: len(tokenizer(x)['input_ids']))
    df['sum_len'] = df['summary'].apply(lambda x: len(tokenizer(x)['input_ids']))
    df = df.drop(df[(df['dial_len']>input_max_length) | (df['sum_len']>output_max_length)].index, axis=0)

    ds = Dataset.from_pandas(df).train_test_split(test_size=0.1, shuffle=True, seed=42)
    ds = ds.map(partial(format_prompt, tokenizer=tokenizer))
    ds = ds.remove_columns(['en_zh_dialogue', 'summary'])

    args = TrainingArguments(
        output_dir="outputs",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,
        warmup_steps=200,
        eval_steps=50,
        evaluation_strategy="steps",
        num_train_epochs=4,
        learning_rate=5e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="steps",
        save_total_limit=1,
        report_to=["none"],
        bf16=True,
        max_grad_norm=1.0
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds['train'],
        eval_dataset=ds['test'],
        dataset_text_field="input_ids",
        peft_config=lora_config,
        args=args,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[EarlyStoppingCallback(patience=5)],
    )

    model, trainer = accelerator.prepare(model, trainer)

    trainer.train()
    trainer.model.save_pretrained(adapter_path)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--adapter_path", type=str, required=True)
    parser.add_argument("--input_max_length", type=int, required=True)
    parser.add_argument("--output_max_length", type=int, required=True)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--benchmark_path", required=True, type=str)

    args = parser.parse_args()
    if args.train:
        finetune(csv_path=args.csv_path,
                adapter_path=args.adapter_path,
                input_max_length=args.input_max_length,
                output_max_length=args.output_max_length,
                model_id=args.model_id)
    
    torch.cuda.empty_cache()

    print("Loading model for evaluation...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model_id)
    model = PeftModel.from_pretrained(model, args.adapter_path)
    model.to(device)
    
    get_and_save_predictions(
        benchmark_path=args.benchmark_path, 
        model=model, 
        tokenizer=tokenizer, 
        output_csv_path=f"en_zh/lora/preds/{args.model_id}.csv"
    )