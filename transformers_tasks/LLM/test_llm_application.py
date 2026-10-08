"""Checks for structured NLP output and MBTI questionnaire integrity."""

import unittest

import torch

from run_llm_application import (
    ROOT,
    SCHEMA,
    mbti_choice,
    normalized_values,
    parse_fields,
    parse_label,
    valid_mbti_questions,
)


class FakeTokenizer:
    def apply_chat_template(self, *args, **kwargs):
        return FakeInputs(input_ids=torch.tensor([[1, 2]]))

    def encode(self, label, add_special_tokens=False):
        return {"A": [3], "B": [4]}[label]


class FakeModel:
    device = torch.device("cpu")

    def __call__(self, input_ids):
        logits = torch.zeros((1, input_ids.shape[-1], 5))
        logits[0, -1, 3] = 1.0
        logits[0, -1, 4] = 3.0
        return type("Output", (), {"logits": logits})()


class FakeInputs(dict):
    def to(self, device):
        return self


class LLMApplicationTests(unittest.TestCase):
    def test_structured_labels_and_fields(self):
        self.assertEqual(parse_label('```json\n{"label":"人物"}\n```', ("人物", "书籍")), "人物")
        for response in ('{"label":"城市"}', '人物', '{"label":"人物","extra":1}'):
            with self.assertRaises(ValueError):
                parse_label(response, ("人物", "书籍"))
        fields = {key: [] for key in SCHEMA["人物"]}
        fields["姓名"] = ["张译"]
        self.assertEqual(parse_fields(str(fields).replace("'", '"'), "人物"), fields)
        self.assertEqual(normalized_values(fields), {("姓名", "张译")})
        with self.assertRaises(ValueError):
            parse_fields('{"姓名":["张译"]}', "人物")

    def test_mbti_invalid_mapping_is_excluded(self):
        questions, excluded = valid_mbti_questions(ROOT / "llms_mbti" / "mbti_questions.json")
        self.assertEqual(len(questions), 92)
        self.assertEqual([item["id"] for item in excluded], ["75"])

    def test_mbti_uses_model_choice_logits(self):
        choice, probability = mbti_choice(FakeModel(), FakeTokenizer(), "测试题")
        self.assertEqual(choice, "B")
        self.assertGreater(probability, 0.8)


if __name__ == "__main__":
    unittest.main()
