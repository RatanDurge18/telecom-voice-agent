"""LLM adapters. All return the same LLMResponse so the agent loop is provider-agnostic.

  AnthropicLLM : Claude with tool calling   (set ANTHROPIC_API_KEY)
  OpenAILLM    : OpenAI chat completions    (set OPENAI_API_KEY)
  MockLLM      : rule-based, offline; lets you run the UI, tools and evals with no key

Internally the conversation is stored in Anthropic message format. OpenAILLM converts it
to OpenAI format on every call, so nothing else in the project changes per provider.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)  # [{"id", "name", "input"}]

    def raw_content(self) -> list[dict]:
        """Assistant content blocks in the internal (Anthropic) message format."""
        blocks: list[dict] = []
        if self.text:
            blocks.append({"type": "text", "text": self.text})
        for c in self.tool_calls:
            blocks.append({"type": "tool_use", "id": c["id"], "name": c["name"], "input": c["input"]})
        return blocks


class AnthropicLLM:
    name = "anthropic"

    def __init__(self, model: str | None = None, client=None):
        if client is None:
            import anthropic  # lazy import: other modes don't need the package
            client = anthropic.Anthropic()
        self.client = client
        # Haiku is the sensible default for voice: latency matters more than depth here.
        self.model = model or os.getenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.label = f"Claude ({self.model})"

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        resp = self.client.messages.create(
            model=self.model, max_tokens=300, system=system, tools=tools, messages=messages)
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [{"id": b.id, "name": b.name, "input": dict(b.input)} for b in resp.content if b.type == "tool_use"]
        return LLMResponse(text=text.strip(), tool_calls=calls)


class OpenAILLM:
    name = "openai"

    def __init__(self, model: str | None = None, client=None):
        if client is None:
            import openai  # lazy import
            client = openai.OpenAI()  # reads OPENAI_API_KEY (and OPENAI_BASE_URL if set)
        self.client = client
        # Change with OPENAI_MODEL. Pick a small fast model for voice.
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.label = f"OpenAI ({self.model})"

    @staticmethod
    def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for m in messages:
            content = m["content"]
            if m["role"] == "user":
                if isinstance(content, str):
                    out.append({"role": "user", "content": content})
                else:  # tool results: one "tool" message each
                    for block in content:
                        out.append({"role": "tool", "tool_call_id": block["tool_use_id"],
                                    "content": block["content"]})
            else:  # assistant
                text = "".join(b["text"] for b in content if b["type"] == "text")
                calls = [{"id": b["id"], "type": "function",
                          "function": {"name": b["name"], "arguments": json.dumps(b["input"])}}
                         for b in content if b["type"] == "tool_use"]
                msg: dict = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                out.append(msg)
        return out

    @staticmethod
    def to_openai_tools(tools: list[dict]) -> list[dict]:
        return [{"type": "function",
                 "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}}
                for t in tools]

    def complete(self, system: str, messages: list[dict], tools: list[dict]) -> LLMResponse:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=self.to_openai_messages(system, messages),
            tools=self.to_openai_tools(tools))
        msg = resp.choices[0].message
        calls = []
        for tc in (msg.tool_calls or []):
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append({"id": tc.id, "name": tc.function.name, "input": args})
        return LLMResponse(text=(msg.content or "").strip(), tool_calls=calls)


def make_llm():
    """LLM_PROVIDER=mock|anthropic|openai forces a mode. Blank: Anthropic key, else OpenAI key, else mock."""
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if not provider:
        if os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        elif os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        else:
            provider = "mock"
    if provider == "openai":
        return OpenAILLM()
    if provider == "anthropic":
        return AnthropicLLM()
    from agent.mock_llm import MockLLM
    return MockLLM()
