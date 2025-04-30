from __future__ import annotations
import os
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent

# your retrieval tool must be decorated with @tool
from agentic_search.retriever_tool import run_retrieval


def main() -> None:
    # 0) Hard-code or set your Ollama API key

    # 1) Define available tools
    tools = [run_retrieval]

    # 2) Initialize ChatOllama with streaming enabled
    llm = ChatOllama(
        model="llama3.1:8b",
        stream=True,
        temperature=0.2,
        top_k=40,
        repeat_penalty=1.2,
        num_ctx=4096,
        seed=42,
    )

    # 3) Provide a system prompt string to guide the agent
    system_prompt = (
        "You are a friendly product search assistant and chat companion. "
        "When the user asks for products (e.g., 'I need laptops' or 'show me engine oil'), "
        "invoke the run_retrieval tool and clearly summarize the top results. "
        "Otherwise, engage in natural conversation—answer greetings and keep the tone warm."
    )

    # 4) Create the React-style agent with the system prompt
    agent = create_react_agent(
        llm,
        tools,
        prompt=system_prompt,
    )

    print("Chat Agent ready! Type 'exit' or 'quit' to stop.")
    while True:
        user_input = input("\n> ").strip()
        if user_input.lower() in {"exit", "quit"}:
            break

        # 5) Stream the assistant's response using "values" mode
        for event in agent.stream(
            {"messages": [HumanMessage(content=user_input)]},
            stream_mode="values",
        ):
            # event is a dict; last message is the AI's
            msg = event["messages"][-1]
            print(msg.content, end="", flush=True)
        print()  # newline after the full response


if __name__ == "__main__":
    main()
