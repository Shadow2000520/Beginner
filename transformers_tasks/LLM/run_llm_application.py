"""Run the repository's zero-shot NLP demos and MBTI probe with a local chat LLM."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "Qwen2.5-1.5B-Instruct"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TYPES = ("人物", "书籍", "电视剧", "电影", "城市", "国家")
SCHEMA = {
    "人物": ("姓名", "性别", "出生日期", "出生地点", "职业", "获得奖项"),
    "书籍": ("书名", "作者", "类型", "发行时间", "定价"),
    "电视剧": ("电视剧名称", "导演", "演员", "题材", "出品方"),
}
MBTI_PAIRS = (("E", "I"), ("S", "N"), ("T", "F"), ("J", "P"))

CLASSIFICATION_CASES = [
    {"text": "加拿大首都渥太华，位于北美洲北部。", "label": "国家"},
    {"text": "《琅琊榜》由孔笙、李雪执导，胡歌、刘涛主演，是一部古装剧。", "label": "电视剧"},
    {"text": "《满江红》是张艺谋执导的悬疑喜剧电影。", "label": "电影"},
    {"text": "布宜诺斯艾利斯是阿根廷的首都和最大城市。", "label": "城市"},
    {"text": "张译是中国内地男演员。", "label": "人物"},
    {"text": "《三体》是刘慈欣创作的科幻小说。", "label": "书籍"},
]
MATCHING_CASES = [
    {"text": "如何修改头像", "text_b": "可以通过上传图片修改头像吗", "label": "相似"},
    {"text": "王者荣耀司马懿连招", "text_b": "王者荣耀司马懿有什么技巧", "label": "相似"},
    {"text": "王者荣耀司马懿连招", "text_b": "历史上司马懿真的被诸葛亮空城计骗了吗", "label": "不相似"},
    {"text": "晚餐可以做什么菜", "text_b": "明天北京的天气如何", "label": "不相似"},
]
EXTRACTION_CASES = [
    {
        "text": "张译，1978年2月17日出生于黑龙江省哈尔滨市，中国内地男演员。",
        "type": "人物",
        "fields": {"姓名": ["张译"], "性别": ["男"], "出生日期": ["1978年2月17日"], "出生地点": ["黑龙江省哈尔滨市"], "职业": ["演员"], "获得奖项": []},
    },
    {
        "text": "《三体》是刘慈欣创作的长篇科幻小说，第一部于2006年5月在《科幻世界》连载。",
        "type": "书籍",
        "fields": {"书名": ["《三体》"], "作者": ["刘慈欣"], "类型": ["长篇科幻小说"], "发行时间": ["2006年5月"], "定价": []},
    },
    {
        "text": "《琅琊榜》由孔笙、李雪执导，胡歌、刘涛主演，是一部古装电视剧。",
        "type": "电视剧",
        "fields": {"电视剧名称": ["《琅琊榜》"], "导演": ["孔笙", "李雪"], "演员": ["胡歌", "刘涛"], "题材": ["古装"], "出品方": []},
    },
]


def classification_messages(text: str, allowed=TYPES):
    return [
        {"role": "system", "content": "你是文本分类器。只根据输入文本判断类别，只输出 JSON 对象，不添加解释。"},
        {"role": "user", "content": f"从以下类别中选一个：{', '.join(allowed)}。\n文本：{text}\n输出格式：{{\"label\":\"类别\"}}"},
    ]


def matching_messages(text: str, text_b: str):
    return [
        {"role": "system", "content": "你是语义匹配分类器。两个句子讨论同一对象、同一领域且意图相关，就判为相似；若只共享少量词语，却讨论不同场景或目的，就判为不相似。只输出 JSON 对象。"},
        {"role": "user", "content": f"句子一：{text}\n句子二：{text_b}\n只选“相似”或“不相似”。输出格式：{{\"label\":\"相似\"}} 或 {{\"label\":\"不相似\"}}"},
    ]


def extraction_messages(text: str, entity_type: str):
    fields = SCHEMA[entity_type]
    skeleton = json.dumps({field: [] for field in fields}, ensure_ascii=False)
    return [
        {"role": "system", "content": "你是信息抽取器。只提取原文明确出现的信息，不推测。最终答案只能是一个以 { 开始、以 } 结束的 JSON 对象，不要外层数组或 Markdown。"},
        {"role": "user", "content": f"实体类型：{entity_type}\n原文：{text}\n严格按照这个 JSON 结构填值：{skeleton}\n每个键保留不变，每个值是字符串数组；原文未提及则保留 []。只输出填好的对象。"},
    ]


def parse_json_object(response: str):
    start, end = response.find("{"), response.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Model did not return a JSON object")
    result = json.loads(response[start : end + 1])
    if not isinstance(result, dict):
        raise ValueError("Model output must be a JSON object")
    return result


def parse_label(response: str, allowed):
    value = parse_json_object(response)
    if set(value) != {"label"} or value["label"] not in allowed:
        raise ValueError(f"Expected a label in {list(allowed)}")
    return value["label"]


def parse_fields(response: str, entity_type: str):
    value = parse_json_object(response)
    expected = set(SCHEMA[entity_type])
    if set(value) != expected:
        raise ValueError(f"Expected exactly these fields: {sorted(expected)}")
    for key, items in value.items():
        if not isinstance(items, list) or any(not isinstance(item, str) for item in items):
            raise ValueError(f"Field {key} must be a string array")
    return value


def load_model(model_name: str, device: str):
    local = Path(model_name).exists()
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local)
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, local_files_only=local).to(device).eval()
    if tokenizer.chat_template is None:
        raise ValueError("This runner requires an instruction model with a chat template")
    return model, tokenizer


@torch.inference_mode()
def generate(model, tokenizer, messages, max_new_tokens=128):
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0, inputs["input_ids"].shape[-1] :], skip_special_tokens=True).strip()


def infer(model, tokenizer, task, text, text_b=None, entity_type=None):
    if task == "classification":
        response = generate(model, tokenizer, classification_messages(text), 64)
        return {"response": response, "prediction": parse_label(response, TYPES)}
    if task == "matching":
        if not text_b:
            raise ValueError("Matching requires --text-b")
        response = generate(model, tokenizer, matching_messages(text, text_b), 64)
        return {"response": response, "prediction": parse_label(response, ("相似", "不相似"))}
    if entity_type is None:
        type_response = generate(model, tokenizer, classification_messages(text, SCHEMA), 64)
        entity_type = parse_label(type_response, SCHEMA)
    if entity_type not in SCHEMA:
        raise ValueError(f"Extraction type must be one of {list(SCHEMA)}")
    response = generate(model, tokenizer, extraction_messages(text, entity_type), 192)
    return {"response": response, "entity_type": entity_type, "prediction": parse_fields(response, entity_type)}


def normalized_values(fields):
    return {(key, re.sub(r"\s+", "", value)) for key, values in fields.items() for value in values if value and value != "原文中未提及"}


def evaluate_task(model, tokenizer, task):
    cases = {"classification": CLASSIFICATION_CASES, "matching": MATCHING_CASES, "extraction": EXTRACTION_CASES}[task]
    records, correct, type_correct, tp, pred_total, gold_total = [], 0, 0, 0, 0, 0
    for case in cases:
        record = dict(case)
        try:
            result = infer(model, tokenizer, task, case["text"], case.get("text_b"))
            record.update(result)
            if task == "extraction":
                type_correct += result["entity_type"] == case["type"]
                predicted = normalized_values(result["prediction"])
                expected = normalized_values(case["fields"])
                tp += len(predicted & expected)
                pred_total += len(predicted)
                gold_total += len(expected)
                correct += result["entity_type"] == case["type"] and predicted == expected
            else:
                correct += result["prediction"] == case["label"]
        except ValueError as exc:
            record["error"] = str(exc)
        records.append(record)
        print(json.dumps({"task": task, "index": len(records), "prediction": record.get("prediction"), "error": record.get("error")}, ensure_ascii=False), flush=True)
    metrics = {"count": len(cases), "exact_accuracy": correct / len(cases), "valid_rate": sum("error" not in x for x in records) / len(cases)}
    if task == "extraction":
        metrics["entity_type_accuracy"] = type_correct / len(cases)
        metrics["field_value_micro_f1"] = 2 * tp / (pred_total + gold_total) if pred_total + gold_total else 1.0
        metrics["field_value_counts"] = {"true_positive": tp, "predicted": pred_total, "reference": gold_total}
    return {"metrics": metrics, "records": records}


def valid_mbti_questions(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    accepted, excluded = [], []
    valid_pairs = {frozenset(pair) for pair in MBTI_PAIRS}
    for question_id, item in data.items():
        if not isinstance(item, dict) or not all(key in item for key in ("question", "A", "B")) or frozenset((item["A"], item["B"])) not in valid_pairs:
            excluded.append({"id": question_id, "reason": "invalid dimension mapping"})
            continue
        accepted.append((question_id, item))
    return accepted, excluded


@torch.inference_mode()
def mbti_choice(model, tokenizer, question: str):
    messages = [
        {"role": "system", "content": "请回答一个二选一问卷题。选择更符合你通常回答倾向的一项，只回答 A 或 B。"},
        {"role": "user", "content": question.replace("\\n", "\n") + "\n只输出 A 或 B。"},
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(model.device)
    logits = model(**inputs).logits[0, -1].float()
    choices = {label: tokenizer.encode(label, add_special_tokens=False) for label in ("A", "B")}
    if any(len(ids) != 1 for ids in choices.values()):
        raise ValueError("A and B must each encode as one token for next-token scoring")
    scores = {label: logits[ids[0]].item() for label, ids in choices.items()}
    answer = max(scores, key=scores.get)
    probability = torch.softmax(torch.tensor([scores["A"], scores["B"]]), dim=0)[0 if answer == "A" else 1].item()
    return answer, probability


def evaluate_mbti(model, tokenizer, max_questions=None):
    questions, excluded = valid_mbti_questions(ROOT / "llms_mbti" / "mbti_questions.json")
    full_count = len(questions)
    if max_questions:
        questions = questions[:max_questions]
    details = {letter: 0 for pair in MBTI_PAIRS for letter in pair}
    records = []
    for question_id, item in questions:
        choice, probability = mbti_choice(model, tokenizer, item["question"])
        trait = item[choice]
        details[trait] += 1
        records.append({"id": question_id, "choice": choice, "trait": trait, "choice_probability": probability})
        print(json.dumps({"question": question_id, "choice": choice, "trait": trait}, ensure_ascii=False), flush=True)
    complete = len(records) == full_count
    mbti_type = "".join(first if details[first] >= details[second] else second for first, second in MBTI_PAIRS) if complete else None
    ties = ["/".join(pair) for pair in MBTI_PAIRS if details[pair[0]] == details[pair[1]]] if complete else []
    return {"details": details, "res": mbti_type, "complete": complete, "ties": ties, "question_count": len(records), "excluded": excluded, "choices": records, "method": "single-token A/B next-token logits; ties choose the first letter"}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("zero-shot", "predict", "mbti"))
    parser.add_argument("--task", choices=("all", "classification", "matching", "extraction"), default="all")
    parser.add_argument("--model", default=str(DEFAULT_MODEL) if DEFAULT_MODEL.exists() else MODEL_ID)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--text")
    parser.add_argument("--text-b")
    parser.add_argument("--entity-type", choices=tuple(SCHEMA))
    parser.add_argument("--max-questions", type=int, help="MBTI smoke-test limit; omit for all valid questions")
    args = parser.parse_args()
    if args.mode == "predict" and (args.task == "all" or not args.text):
        parser.error("predict needs a specific --task and --text")
    if args.mode == "mbti" and args.max_questions is not None and args.max_questions < 1:
        parser.error("--max-questions must be positive")
    model, tokenizer = load_model(args.model, args.device)
    if args.mode == "predict":
        result = infer(model, tokenizer, args.task, args.text, args.text_b, args.entity_type)
        print(json.dumps({"task": args.task, "text": args.text, **result}, ensure_ascii=False))
        return
    if args.mode == "mbti":
        result = evaluate_mbti(model, tokenizer, args.max_questions)
        default_name = f"mbti_smoke_{args.max_questions}.json" if args.max_questions else "mbti.json"
        output = args.output or ROOT / "runs" / default_name
    else:
        tasks = ("classification", "matching", "extraction") if args.task == "all" else (args.task,)
        result = {task: evaluate_task(model, tokenizer, task) for task in tasks}
        default_name = "zero_shot.json" if args.task == "all" else f"zero_shot_{args.task}.json"
        output = args.output or ROOT / "runs" / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"model": args.model, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = ({key: result[key] for key in ("res", "complete", "details", "ties", "question_count", "excluded")}
               if args.mode == "mbti" else {task: data["metrics"] for task, data in result.items()})
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
