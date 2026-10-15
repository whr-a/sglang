"""CPU checks for the limited Miles V4.1 encoder backport.

Parity checks require SGLANG_V41_REFERENCE_DIR to contain the unmodified
encoding_dsv41.py and chat_encoding.py from the pinned upstream commit. Their
hashes are checked before execution. No model, network, GPU, or import stubs
are used by these tests.
"""

import ast
import copy
import hashlib
import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace

from sglang.srt.entrypoints.openai import chat_encoding, encoding_dsv41
from sglang.srt.entrypoints.openai.protocol import Tool

BASE_COMMIT = "4c3f163fe54634f20ef846e1ec0c507b27ae2430"
UPSTREAM_COMMIT = "4076a68579b4a7ae3e9d2ce2f94dec9917237bab"
REFERENCE_HASHES = {
    "encoding_dsv41.py": "9c29c810c440589f2456df36ecf754a5083244f5d5887c64d6735f32b538ea4c",
    "chat_encoding.py": "940b41c09a7a604a513a6422007e6d364f6b14b3fc26aa7e076a20442783dfe3",
}
HELPERS = {
    "parse_dsv41_reasoning_effort",
    "dsv41_tool_payload",
    "default_dsv41_reasoning_effort_from_env",
}


def _definitions(source):
    return {
        node.name: ast.dump(node, include_attributes=False)
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef)
    }


def _reference_sources(test):
    directory = os.environ.get("SGLANG_V41_REFERENCE_DIR")
    if not directory:
        test.skipTest("SGLANG_V41_REFERENCE_DIR is required for upstream parity")
    sources = {}
    for name, expected in REFERENCE_HASHES.items():
        data = (Path(directory) / name).read_bytes()
        test.assertEqual(hashlib.sha256(data).hexdigest(), expected, name)
        sources[name] = data.decode("utf-8")
    return sources


def _execute_source(source, name):
    namespace = {"__name__": name}
    exec(compile(source, name, "exec"), namespace)
    return namespace


class MilesV41CompatibilityTests(unittest.TestCase):
    def test_encoder_matches_pinned_upstream_bytes(self):
        contents = Path(encoding_dsv41.__file__).read_bytes()
        self.assertEqual(
            hashlib.sha256(contents).hexdigest(),
            REFERENCE_HASHES["encoding_dsv41.py"],
        )

    def test_existing_functions_and_dispatch_are_unchanged(self):
        root = Path(chat_encoding.__file__).resolve().parents[5]
        base = subprocess.run(
            [
                "git",
                "-C",
                str(root),
                "show",
                f"{BASE_COMMIT}:python/sglang/srt/entrypoints/openai/chat_encoding.py",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        current = Path(chat_encoding.__file__).read_text(encoding="utf-8")
        before, after = _definitions(base), _definitions(current)
        self.assertEqual(set(after) - set(before), HELPERS)
        for name in before:
            self.assertEqual(after[name], before[name], name)

        baseline = _execute_source(base, "sglang_v41_compat_base_reference")
        cases = [
            ("LlamaForCausalLM", None, None),
            ("DeepseekV4ForCausalLM", None, "dsv4"),
            ("DeepseekV3ForCausalLM", None, "dsv32"),
            ("KimiK3ForCausalLM", None, "kimi_k3"),
            ("InklingForConditionalGeneration", None, "inkling"),
            ("LlamaForCausalLM", "deepseekv4", "dsv4"),
            ("LlamaForCausalLM", "deepseekv32", "dsv32"),
            ("LlamaForCausalLM", "kimi_k3", "kimi_k3"),
            # This backport does not enable V4.1 serving dispatch.
            ("DeepseekV41ForCausalLM", None, "dsv4"),
            ("LlamaForCausalLM", "deepseekv41", None),
        ]
        for architecture, parser, expected in cases:
            config = SimpleNamespace(
                architectures=[architecture], model_type="deepseek_v41"
            )
            kwargs = dict(hf_config=config, tokenizer=None, tool_call_parser=parser)
            with self.subTest(architecture=architecture, parser=parser):
                self.assertEqual(chat_encoding.resolve_chat_encoding_spec(**kwargs), expected)
                self.assertEqual(
                    chat_encoding.resolve_chat_encoding_spec(**kwargs),
                    baseline["resolve_chat_encoding_spec"](**kwargs),
                )

    def test_existing_text_encoders_with_real_tokenizer(self):
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast

        alphabet = pre_tokenizers.ByteLevel.alphabet()
        vocabulary = {token: index for index, token in enumerate(alphabet)}
        vocabulary["[UNK]"] = len(vocabulary)
        engine = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="[UNK]"))
        engine.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        engine.decoder = decoders.ByteLevel()
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=engine,
            unk_token="[UNK]",
            chat_template="{% for m in messages %}{{ m['role'] }}:{{ m['content'] }}\n{% endfor %}",
        )
        root = Path(chat_encoding.__file__).resolve().parents[5]
        source = subprocess.run(
            ["git", "-C", str(root), "show", f"{BASE_COMMIT}:python/sglang/srt/entrypoints/openai/chat_encoding.py"],
            check=True, capture_output=True, text=True,
        ).stdout
        baseline = _execute_source(source, "sglang_v41_compat_base_reference")
        messages = [{"role": "user", "content": "hello 世界"}]
        for spec in (None, "dsv4", "dsv32"):
            for mode in ("chat", "thinking"):
                kwargs = dict(tokenizer=tokenizer, spec=spec, messages=messages, thinking_mode=mode)
                with self.subTest(spec=spec, mode=mode):
                    self.assertEqual(
                        chat_encoding.encode_simple_chat(**kwargs),
                        baseline["encode_simple_chat"](**kwargs),
                    )

    def test_helpers_match_pinned_upstream(self):
        sources = _reference_sources(self)
        reference = _execute_source(sources["chat_encoding.py"], "sglang_v41_compat_upstream_chat")
        official = _definitions(sources["chat_encoding.py"])
        local = _definitions(Path(chat_encoding.__file__).read_text())
        for name in HELPERS:
            self.assertEqual(local[name], official[name], name)
        values = [None, True, False, -1, 0, 1, 50, 100, 101, -0.1, 0.0, 0.01, 0.5, 0.99, 1.0, "low", "high", "xhigh", "max", "invalid"]
        for value in values:
            with self.subTest(value=value):
                self.assertEqual(
                    chat_encoding.parse_dsv41_reasoning_effort(value),
                    reference["parse_dsv41_reasoning_effort"](value),
                )
        for raw in (None, "", "  ", "1", "50", "100", "low", " high ", "xhigh", "max"):
            self.assertEqual(
                chat_encoding.default_dsv41_reasoning_effort_from_env(raw),
                reference["default_dsv41_reasoning_effort_from_env"](raw),
            )
        for raw in ("0", "101", "bad", "0.5"):
            for implementation in (chat_encoding.default_dsv41_reasoning_effort_from_env, reference["default_dsv41_reasoning_effort_from_env"]):
                with self.assertRaises(ValueError):
                    implementation(raw)
        tools = [
            {"type": "function", "function": {"parameters": {"type": "object"}, "description": "查天气", "name": "weather"}},
            {"type": "function", "function": {"name": "ping"}},
            {"type": "function", "function": {"name": "strict", "strict": True, "parameters": {"type": "object"}}},
        ]
        for payload in tools:
            tool = Tool.model_validate(payload)
            self.assertEqual(chat_encoding.dsv41_tool_payload(tool), reference["dsv41_tool_payload"](tool))
            if "strict" not in payload["function"]:
                self.assertNotIn("strict", chat_encoding.dsv41_tool_payload(tool)["function"])
        ordered = chat_encoding.dsv41_tool_payload(Tool.model_validate(tools[0]))
        self.assertEqual(list(ordered["function"]), ["name", "description", "parameters"])

    def test_encoder_semantics_match_pinned_upstream(self):
        sources = _reference_sources(self)
        reference = _execute_source(sources["encoding_dsv41.py"], "sglang_v41_compat_upstream_encoder")
        conversation = [
            {"role": "system", "content": "Answer precisely."},
            {"role": "user", "content": "Question one"},
            {"role": "assistant", "content": "Answer one", "reasoning_content": "Prior reasoning"},
            {"role": "user", "content": "Question two 世界"},
        ]
        cases = [
            [{"role": "user", "content": "hello"}],
            conversation,
            conversation + [{"role": "system", "content": "New instruction"}],
            [{"role": "developer", "content": "Instruction"}, {"role": "user", "content": "Answer"}],
            [{"role": "system", "content": "", "response_format": {"type": "object"}}, {"role": "user", "content": "JSON"}],
            [{"role": "user", "content": [{"type": "text", "text": "Describe "}, {"type": "image_url", "image_url": {"url": "https://example.invalid/image.png"}}]}],
            [{"role": "system", "content": "Use a tool", "tools": [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object"}}}]}, {"role": "user", "content": "Ping"}, {"role": "assistant", "content": "", "reasoning_content": "Need ping", "tool_calls": [{"id": "call-1", "type": "function", "function": {"name": "ping", "arguments": '{"text":"世界","n":2}'}}]}, {"role": "tool", "tool_call_id": "call-1", "content": "pong"}],
        ]
        comparisons = 0
        for index, messages in enumerate(cases):
            for mode in ("chat", "thinking"):
                for effort in (None, "low", "high", "xhigh", "max", 1, 100):
                    for drop in (False, True):
                        kwargs = dict(thinking_mode=mode, reasoning_effort=effort, drop_thinking=drop, return_multi_modal_data=True)
                        with self.subTest(case=index, mode=mode, effort=effort, drop=drop):
                            before = copy.deepcopy(messages)
                            self.assertEqual(encoding_dsv41.encode_messages(messages, **kwargs), reference["encode_messages"](messages, **kwargs))
                            self.assertEqual(messages, before)
                            comparisons += 1
        self.assertEqual(comparisons, 196)
        rendered = encoding_dsv41.encode_messages(conversation + [{"role": "system", "content": "New instruction"}], thinking_mode="thinking")
        self.assertTrue(rendered.endswith("<｜Assistant｜><think>"))
        self.assertIn("Reasoning Effort: 75", rendered)
        for effort in (0, 101, "bad"):
            for implementation in (encoding_dsv41.encode_messages, reference["encode_messages"]):
                with self.assertRaises(ValueError):
                    implementation([{"role": "user", "content": "hello"}], thinking_mode="thinking", reasoning_effort=effort)

    def test_context_and_error_contract_match_pinned_upstream(self):
        sources = _reference_sources(self)
        reference = _execute_source(sources["encoding_dsv41.py"], "sglang_v41_compat_upstream_encoder")
        context = [
            {"role": "system", "content": "Be precise"},
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer", "reasoning_content": "Reasoning"},
        ]
        messages = [{"role": "user", "content": "Next question"}]
        for mode in ("chat", "thinking"):
            for drop in (True, False):
                for bos in (True, False):
                    kwargs = dict(context=context, thinking_mode=mode, drop_thinking=drop, add_default_bos_token=bos)
                    self.assertEqual(
                        encoding_dsv41.encode_messages(messages, **kwargs),
                        reference["encode_messages"](messages, **kwargs),
                    )
        invalid = [
            [{"role": "unknown", "content": "bad"}],
            [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "bad", "arguments": "[]"}}]}],
        ]
        for messages in invalid:
            errors = []
            for implementation in (encoding_dsv41.encode_messages, reference["encode_messages"]):
                try:
                    implementation(copy.deepcopy(messages), thinking_mode="chat")
                except Exception as error:
                    errors.append((type(error).__name__, str(error)))
            self.assertEqual(len(errors), 2)
            self.assertEqual(errors[0], errors[1])


if __name__ == "__main__":
    unittest.main()
