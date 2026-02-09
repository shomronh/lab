import os
import time
import json
from typing import Generator, List, Dict
import requests


class LLMClient:
    """Simple client for interacting with LocalAI's API with streaming support"""

    def __init__(self, api_base: str = "http://localhost:8080/v1"):
        self.api_base = api_base
        self.headers = {"Content-Type": "application/json"}

    def list_models(self) -> List[Dict]:
        """List available models"""
        response = requests.get(f"{self.api_base}/models")
        return response.json()

    # The formatting is important to ensure the model understands the conversation structure
    # in our case its about the huggingface phi models, which expect a specific format for system, 
    # user, and assistant messages
    def _format_phi_messages(self, messages: List[Dict]) -> str:
        """Format messages for Phi model's expected input format"""
        formatted = ""

        system_msg = next((msg for msg in messages if msg["role"] == "system"), None)
        if system_msg:
            formatted += f"<|system|>{system_msg['content']}<|end|>"

        user_msgs = [msg for msg in messages if msg["role"] == "user"]
        if user_msgs:
            formatted += f"<|user|>{user_msgs[-1]['content']}<|end|><|assistant|>"

        return formatted

    def chat_stream(
        self,
        messages: List[Dict],
        model: str = "phi-3.5-mini-instruct",
    ) -> Generator[str, None, None]:
        """Send a stream chat completion request"""
        formatted_prompt = self._format_phi_messages(messages)

        data = {
            "prompt": formatted_prompt,
            "model": model,
            "stream": True,
            "top_p": 0.1,
            # control the randomness of the output, lower values make 
            # it more deterministic
            "temperature": 0.3,
            "stop": ["<|endoftext|>", "<|end|>"],
            "max_tokens": 1024,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }

        with requests.post(f"{self.api_base}/completions", headers=self.headers, 
                           json=data,
                           stream=True) as response:
            if response.status_code != 200:
                raise Exception(f"Error: {response.text}")
            
            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    
                    if line.startswith("data: "):
                        line = line[6:]  # Remove "data: " prefix
                    
                    if line == "[DONE]":
                        continue
                    
                    try:
                        chunk = json.loads(line)
                        if chunk.get("choices") and chunk["choices"][0].get("text"):
                            content = chunk["choices"][0]["text"]
                            # when using Generator we can yield each chunk of content 
                            # as it arrives, allowing for real-time streaming responses
                            yield content
                    except json.JSONDecodeError:
                        continue


def demonstrate_capabilities():
    """Show basic capabilities of the LLM with streaming responses"""
    llm = LLMClient(os.getenv("LLM_API_BASE", "http://localhost:8080/v1"))

    print("\nAvailable Models:")
    try:
        models = llm.list_models()
        print(json.dumps(models, indent=2))
    except Exception as e:
        print(f"Error listing models: {e}")

    examples = [
        {
            "title": "DevOps Explanation",
            "messages": [
                {
                    "role": "user",
                    "content": "What is DevOps in one sentence?",
                },
            ],
        },
    ]

    for example in examples:
        print(f"\nExample: {example['title']}")
        print("Response:")

        for token in llm.chat_stream(example["messages"]):
            end_char = "\n" if token == " " else ""
            print(token, end=end_char, flush=True)


if __name__ == "__main__":
    print("🤖 LocalAI Streaming Client Demo")
    print("Testing connection to LocalAI and demonstrating basic capabilities...")
    demonstrate_capabilities()
