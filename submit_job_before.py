import json, os, re
from openai import OpenAI
from typing import Iterator, Dict, Any, List

#export your OPENAI_API_KEY in your environment before running this script
# export OPENAI_API_KEY=""

RUN_BATCH = True
#for testing, set RUN_BATCH = False 
INPUT_JSON = "./data/librispeech_100_data.list"
BATCH_INPUT_FILE = "batch_input_v1.jsonl"
BATCH_SIZE = 16

SYSTEM_PROMPT = """You are a linguistics expert annotating speech transcripts according to the Switchboard (LDC) disfluency annotation standard.

Task
label two tags for every token in each sample:
- disfluency label l: 0 = fluent, 1 = disfluent(delete it for a fluent sentence), The final intended expression (repair) is always labeled 0.
- confidence label c: reflecting how sure you are that label l is correct in [50,100].

Input Constraints
- Input is JSON:{"samples":[{"id":"...","text":"..."}, ...]}
- Tokenization is STRICTLY by spaces. Do not modify tokens in any way.

Rules (apply in order; earlier rules override later)
1) Filled pauses: UH, UM, ER, AH → l=1

2) Repetitions:
Prefer detecting LOCAL/adjacent repetitions typical of speech.
- Single-token repetition: "I I I THINK" → earlier I’s are l=1; the last I is l=0.
- Block repetition: if a contiguous sequence of tokens is immediately repeated, mark ALL tokens in the earlier occurrence as l=1.
  Example: "DO YOU KNOW DO YOU KNOW WHERE IT IS"
  → first "DO YOU KNOW" = l=1 for each token; second "DO YOU KNOW" = l=0 for each token.
If repetition is obvious but not strictly adjacent, you may still apply this rule; otherwise do NOT over-mark.
Repetition has priority even if it contains discourse markers (e.g., YOU KNOW).

3) Repairs (reparandum + optional editing material + repair):
A repair replaces or restarts an earlier formulation.
- Reparandum (the replaced/abandoned content) → l=1
- Editing terms that introduce the correction (e.g., I MEAN, NO, WELL, OR, SORRY) → l=1
- Repair (the final corrected/intended content) → l=0

4) False starts / abandoned fragments and restarts:
If the speaker starts an expression and then abandons it and restarts, label the abandoned material as l=1 and the restart as l=0.
(Fragments, partial words, or cut-off pieces—if present as tokens—are typically l=1 when abandoned.)

5) Discourse markers (context-dependent):
LIKE, WELL, SO, ACTUALLY, YOU KNOW, BASICALLY
- If removable without changing propositional meaning and it functions as hesitation/filler → l=1
- If it contributes discourse/pragmatic meaning (contrast, emphasis, turn management) → l=0
Multi-token markers (e.g., YOU KNOW, I MEAN): decide for the phrase, but output labels per token.
Note: If I MEAN clearly introduces a repair, Rule 3 applies (editing term), not this rule.

Output Constraints
- Output JSON only.
- For each sample id return exactly one result.
- First split text by single spaces into a token list T (do not change tokens).
- Create a with EXACTLY len(T) items, one per token, in the same order.
- Each a[k].w MUST equal T[k] exactly.
- Before returning JSON, VERIFY len(a)==len(T). If not, fix it and re-check.
- Never omit any token; if unsure, output l=0 with low confidence rather than skipping.
"""

RESPONSE_SCHEMA = {
  "name": "disfluency_labeling_batch",
  "strict": True,
  "schema": {
    "type": "object",
    "additionalProperties": False,
    "properties": {
      "r": {
        "type": "array",
        "minItems": 1,
        "maxItems": 200,
        "items": {
          "type": "object",
          "additionalProperties": False,
          "properties": {
            "id": {"type": "string"},
            "a": {
              "type": "array",
              "minItems": 1,
              "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                  "w": {"type": "string"},
                  "l": {"type": "integer","enum": [0, 1]},
                  "c": {"type": "integer", "minimum": 50, "maximum": 100}
                },
                "required": ["w", "l", "c"]
              }
            }
          },
          "required": ["id", "a"]
        }
      }
    },
    "required": ["r"]
  }
}

def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error at line {line_no}: {e}\nLine: {line[:200]}")

def batch_samples(input_jsonl: str, batch_size: int) -> Iterator[Dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")

    buf: List[Dict[str, str]] = []

    for obj in iter_jsonl(input_jsonl):
        if "txt" not in obj:
            raise KeyError(f"Missing 'txt' in sample: {obj}")

        sample_id = obj.get("key")
        if sample_id is None:
            raise KeyError(f"Missing 'key' in sample: {obj}")

        raw = str(obj["txt"])
        text = re.sub(r"\s+", " ", raw.strip())  # 关键：单空格化

        buf.append({"id": str(sample_id), "text": text})

        if len(buf) >= batch_size:
            yield {"samples": buf}
            buf = []

    if buf:
        yield {"samples": buf}

def batch():
    with open(BATCH_INPUT_FILE, 'w', encoding='utf-8') as f:
        idx = 0
        for batch in batch_samples(INPUT_JSON, BATCH_SIZE):
            batch_json = json.dumps(batch, ensure_ascii=False)
            request = {
                "custom_id": f"id_{idx}", 
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": batch_json}
                    ],
                    "temperature": 0, # 必须为 0
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": RESPONSE_SCHEMA
                    }
                }
            }
            f.write(json.dumps(request, ensure_ascii=False) + "\n")
            idx += 1
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")
    client = OpenAI()
    
    print("Uploading file to OpenAI...")
    with open(BATCH_INPUT_FILE, "rb") as fp:
        batch_file = client.files.create(file=fp, purpose="batch")
    
    print(f"File uploaded. File ID: {batch_file.id}")
    print("Creating Batch Job...")
    
    batch_job = client.batches.create(
        input_file_id=batch_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )
    
    print("-" * 30)
    print(f"SUCCESS! Batch Job ID: {batch_job.id}")
    print("-" * 30)
    print("Please SAVE this Job ID. You will need it for the next step.")

def test():
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("Missing OPENAI_API_KEY in environment.")
    client = OpenAI()

    with open("sync_output.jsonl", "w", encoding="utf-8") as out:
        for batch in batch_samples(INPUT_JSON, BATCH_SIZE):
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
                ],
                temperature=0,
                response_format={"type": "json_schema", "json_schema": RESPONSE_SCHEMA},
            )
            content = resp.choices[0].message.content
            obj = json.loads(content)  # 失败就立刻暴露问题
            out.write(json.dumps(obj, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    if RUN_BATCH:
        batch()
    else:
        test()
