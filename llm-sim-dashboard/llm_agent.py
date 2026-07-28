"""Natural language -> simulation parameters, via OpenAI or Ollama.

Both backends are given the same ``run_simulation`` tool (see
:func:`sim_spec.tool_schema`). The model either

* calls the tool -> we get a structured, validated parameter dict to run, or
* replies in plain text -> a clarification / answer shown to the user.

The two providers are selected by a flag (``settings.LLM_PROVIDER`` or the UI
toggle). OpenAI uses native function calling; Ollama uses its ``/api/chat``
tool-calling interface. Neither is imported until actually used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import settings
import sim_spec

SYSTEM_PROMPT = """\
You are the control agent for JaxMARL-BC, a JAX framework that trains
reinforcement-learning agents on macroeconomic models (Real Business Cycle,
Krusell-Smith, and a heterogeneous RBC grid).

Your job: translate the user's request into a single call to the
`run_simulation` tool with the correct parameters, OR, if the request is a
question or is too ambiguous to run, answer briefly in plain text and ask for
the missing detail.

Guidance:
- Always choose an `experiment`. Use 'rbc' for single-agent RBC, 'ks' for the
  heterogeneous-agent Krusell-Smith model, 'general' for the heterogeneous grid.
  Prefer smaller/quicker presets when the user asks for a quick/test/demo run.
- Only set parameters the user implies or that are needed. Leave the rest to
  framework defaults — do not invent values.
- Map informal language to parameters: "quick"/"fast"/"smoke test" -> small
  total_timesteps (10000-30000) and device 'cpu'; "on CPU" -> device 'cpu';
  "N agents" -> n_agents; "longer/thorough training" -> larger total_timesteps.
- Never fabricate a completed run or its results. You only configure the run;
  the framework executes it.
"""


@dataclass
class AgentReply:
    kind: str                       # "run" | "message" | "error"
    text: str = ""                  # assistant-facing prose
    params: Optional[Dict[str, Any]] = None   # normalized, when kind == "run"
    raw_args: Optional[Dict[str, Any]] = None
    dotlist: List[str] = field(default_factory=list)
    provider: str = ""
    model: str = ""


def translate(request: str, provider: Optional[str] = None,
              history: Optional[List[Dict[str, str]]] = None) -> AgentReply:
    provider = (provider or settings.LLM_PROVIDER).lower()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": request})

    if provider == "openai":
        return _via_openai(messages)
    if provider == "ollama":
        return _via_ollama(messages)
    return AgentReply(kind="error", text=f"Unknown provider: {provider}",
                      provider=provider)


def _finalize(raw_args: Dict[str, Any], provider: str, model: str) -> AgentReply:
    """Validate tool args into a runnable AgentReply (or an error)."""
    try:
        params = sim_spec.normalize(raw_args)
        dotlist = sim_spec.to_dotlist(params)
    except ValueError as e:
        return AgentReply(kind="error", provider=provider, model=model,
                          raw_args=raw_args,
                          text=f"The model produced an invalid configuration: {e}")
    return AgentReply(kind="run", params=params, raw_args=raw_args,
                      dotlist=dotlist, provider=provider, model=model,
                      text=_describe(dotlist))


def _describe(dotlist: List[str]) -> str:
    body = "\n".join(f"  {tok}" for tok in dotlist)
    return "Proposed run:\n" + body


# ── OpenAI ───────────────────────────────────────────────────────────────────
def _via_openai(messages: List[Dict[str, str]]) -> AgentReply:
    if not settings.OPENAI_API_KEY:
        return AgentReply(kind="error", provider="openai",
                          text="OPENAI_API_KEY is not set. Add it to .env or "
                               "switch the provider to Ollama.")
    try:
        from openai import OpenAI
    except ImportError:
        return AgentReply(kind="error", provider="openai",
                          text="The 'openai' package is not installed.")

    client = OpenAI(api_key=settings.OPENAI_API_KEY,
                    base_url=settings.OPENAI_BASE_URL)
    model = settings.OPENAI_MODEL
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=[sim_spec.tool_schema()],
            tool_choice="auto",
            temperature=0,
        )
    except Exception as e:  # network / auth / model errors
        return AgentReply(kind="error", provider="openai", model=model,
                          text=f"OpenAI request failed: {e}")

    msg = resp.choices[0].message
    if msg.tool_calls:
        call = msg.tool_calls[0]
        try:
            raw_args = json.loads(call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            return AgentReply(kind="error", provider="openai", model=model,
                              text=f"Could not parse tool arguments: {e}")
        return _finalize(raw_args, "openai", model)

    return AgentReply(kind="message", provider="openai", model=model,
                      text=msg.content or "(no response)")


# ── Ollama ───────────────────────────────────────────────────────────────────
def _via_ollama(messages: List[Dict[str, str]]) -> AgentReply:
    import requests

    model = settings.OLLAMA_MODEL
    url = settings.OLLAMA_HOST.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "tools": [sim_spec.tool_schema()],
        "stream": False,
        "options": {"temperature": 0},
    }
    try:
        r = requests.post(url, json=payload, timeout=180)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        return AgentReply(kind="error", provider="ollama", model=model,
                          text=f"Ollama request to {url} failed: {e}")

    msg = data.get("message", {}) or {}
    tool_calls = msg.get("tool_calls") or []
    if tool_calls:
        args = tool_calls[0].get("function", {}).get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError as e:
                return AgentReply(kind="error", provider="ollama", model=model,
                                  text=f"Could not parse tool arguments: {e}")
        return _finalize(args, "ollama", model)

    content = (msg.get("content") or "").strip()
    return AgentReply(kind="message", provider="ollama", model=model,
                      text=content or "(no response — does this model support "
                                      "tool calling?)")
