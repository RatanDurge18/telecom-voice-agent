"""Text-only agent in the terminal. Build and debug the logic here before adding voice.

    python cli.py           # uses Claude if ANTHROPIC_API_KEY is set, else the offline mock
    python cli.py --trace   # also print every tool call
"""
import sys

from agent.agent import GREETING, VoiceAgent

try:  # optional: load .env if python-dotenv is installed
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main() -> None:
    show_trace = "--trace" in sys.argv
    agent = VoiceAgent()
    print(f"[LLM: {agent.llm.name}]  Demo login: 9876543210 / 1234   (type 'quit' to exit)\n")
    print(f"Asha: {GREETING}")
    while True:
        try:
            text = input("You:  ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        result = agent.respond(text)
        if show_trace:
            for step in result["trace"]:
                print(f"      [tool] {step['tool']}({step['input']}) -> {step['output']}")
        print(f"Asha: {result['reply']}   ({result['latency_ms']} ms)")
        if result["transfer"]:
            print(f"      [transferred to human] {result['transfer']}")
            break


if __name__ == "__main__":
    main()
