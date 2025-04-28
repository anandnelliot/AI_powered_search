from __future__ import annotations
import os
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama

# import your retrieval tool (must be decorated with @tool)
from hybrid.products.product4 import run_retrieval


def main() -> None:
    # 1. Define the tools your agent can call
    tools = [run_retrieval]

    # 2. Instantiate your LLM with streaming enabled
    llm = ChatOllama(
        model="llama3.1:8b",
        stream=True,
        temperature=0.2,
        top_k=40,
        repeat_penalty=1.2,
        num_ctx=4096,
        seed=42,
    )

    # 3. Create the agent with tool support
    agent_executor = create_react_agent(llm, tools)

    print("Chat Agent ready! Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # Streaming Messages
        for step, metadata in agent_executor.stream(
            {"messages": [HumanMessage(content=user_input)]},
            stream_mode="messages",
        ):
            if metadata["langgraph_node"] == "agent" and (text := step.text()):
                print(text, end="|")
        print()  # newline after the full response


if __name__ == "__main__":
    main()
