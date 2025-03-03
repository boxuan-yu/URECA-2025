import pandas as pd
from google import genai
from dotenv import load_dotenv, find_dotenv
import json
import os
import time
from tqdm import tqdm

load_dotenv(find_dotenv())
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY_4"))

model_id = "gemini-2.0-flash-exp"
base_prompt = """You are a Chinese person in your 20s.
You are recruited for translating English dialogues to English-Chinese code-switched dialogues.
The code-switched dialogues should follow the same structure as the English dialogue.
This is for educational purpose so DO NOT include swear words in your translation.
Return just the code-switched dialogue
"""

def generate_code_switched_dialogue(en_dialogue, base_prompt):
    full_prompt = f"{base_prompt}\nEnglish Dialogue:\n{en_dialogue}"
    retries = 0
    while retries<5:
        try:
            response = client.models.generate_content(model=model_id, contents=full_prompt)
            if not response or not hasattr(response, 'text'):
                raise ValueError(f"Invalid response for dialogue: {en_dialogue}")
            return response.text
        except Exception as e:
            error_message = str(e)
            if "503 UNAVAILABLE" in error_message:
                print("Model is overloaded. Waiting 30 seconds before retrying...")
                time.sleep(30)  # Wait before retrying
                retries += 1
                continue
            else:
                save_progress()
                raise  # Re-raise other exceptions

def save_progress():
    """Save the progress of translations"""
    temp_file = "en_zh/gemini_cs_temp.jsonl"
    with open(temp_file, "w") as f:
        for entry in translated:
            f.write(json.dumps(entry) + "\n")
    os.rename(temp_file, "en_zh/gemini_cs.jsonl")
    print("Progress saved!")

translated = []
if os.path.exists("en_zh/gemini_cs.jsonl"):
    with open("en_zh/gemini_cs.jsonl", "r") as f:
        for line in f:
            translated.append(json.loads(line))

idx = len(translated)
df = pd.read_csv("english_dialogues.csv")

try:
    for i, row in enumerate(tqdm(df.iloc[idx:].itertuples(), initial=idx, total=len(df))):
        try:
            cs = generate_code_switched_dialogue(row.en_dialogue, base_prompt)
            translated.append(
                {
                    "en_dialogue": row.en_dialogue,
                    "en_zh_dialogue": cs,
                    "summary": row.en_summary
                }
            )
            time.sleep(3) 

        except Exception as e:
            print(f"Error at index {i}: {e}")

finally:
    save_progress() 