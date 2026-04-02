"""
client.py
Interactive chat client for the vLLM OpenAI-compatible server.
Run serve.sh first, then: python client.py
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="placeholder",
)

def chat(prompt: str, stream: bool = True) -> str:
    response = client.chat.completions.create(
        model="qwen2.5-1.5b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=512,
        stream=stream,
    )
    full_response = ""
    if stream:
        print("Assistant: ", end="", flush=True)
        for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                full_response += delta
        print()
    else:
        full_response = response.choices[0].message.content
        print(f"Assistant: {full_response}")

    return full_response


if __name__ == "__main__":
    print("vLLM Chat Client — type 'quit' to exit\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        chat(user_input)
        print()