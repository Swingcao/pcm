"""
LLM Client Utilities
====================
Wrapper for OpenAI API calls with retry logic and structured output support.
"""

import json
import asyncio
from typing import Optional, Dict, Any, Type, TypeVar
from pydantic import BaseModel
from openai import AsyncOpenAI

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
import config


T = TypeVar('T', bound=BaseModel)


class LLMClient:
    """
    Async OpenAI client wrapper with retry logic and structured output support.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.base_url = base_url or config.OPENAI_BASE_URL
        self.model = model or config.MAIN_LLM_MODEL
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.use_mock = config.USE_MOCK_LLM

        if not self.use_mock:
            self.client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.base_url
            )
        else:
            self.client = None

    async def chat_completion(
        self,
        messages: list[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1500,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a chat completion request with retry logic.

        Returns:
            Dict containing 'content', 'usage', and 'model' keys.
        """
        if self.use_mock:
            return self._mock_response(messages)

        model = model or self.model

        for attempt in range(self.max_retries):
            try:
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                return {
                    "content": response.choices[0].message.content,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    } if response.usage else None,
                    "model": response.model
                }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise e
                await asyncio.sleep(self.retry_delay * (attempt + 1))

    async def structured_output(
        self,
        messages: list[Dict[str, str]],
        output_schema: Type[T],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1500
    ) -> T:
        """
        Make a chat completion request and parse the response into a Pydantic model.

        The prompt should instruct the model to output JSON matching the schema.
        """
        # Add JSON instruction to the last message if not present
        messages = messages.copy()

        schema_description = json.dumps(output_schema.model_json_schema(), indent=2)
        json_instruction = f"\n\nRespond with valid JSON matching this schema:\n```json\n{schema_description}\n```"

        if messages and messages[-1]["role"] == "user":
            messages[-1] = {
                "role": "user",
                "content": messages[-1]["content"] + json_instruction
            }

        response = await self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Parse the JSON response
        content = response["content"]

        # Extract JSON from markdown code blocks if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()

        try:
            data = json.loads(content)
            return output_schema.model_validate(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse LLM response as JSON: {e}\nContent: {content}")

    def _mock_response(self, messages: list[Dict[str, str]]) -> Dict[str, Any]:
        """Generate a mock response for testing."""
        last_message = messages[-1]["content"] if messages else ""

        # Simple mock logic based on content
        if "intent" in last_message.lower():
            mock_content = '{"label": "General", "confidence": 0.8, "distribution": {"General": 0.8, "Coding": 0.2}}'
        elif "hypothesis" in last_message.lower():
            mock_content = '{"hypothesis_content": "User is interested in AI", "reasoning": "Based on recent queries"}'
        elif "conflict" in last_message.lower():
            mock_content = '{"conflicting_nodes": [], "diagnosis": "No conflict detected"}'
        else:
            mock_content = '{"result": "Mock response", "status": "success"}'

        return {
            "content": mock_content,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "model": "mock-model"
        }


# Singleton instance for convenience
_default_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get the default LLM client instance."""
    global _default_client
    if _default_client is None:
        _default_client = LLMClient()
    return _default_client


async def chat(
    messages: list[Dict[str, str]],
    model: Optional[str] = None,
    **kwargs
) -> str:
    """Convenience function for simple chat completions."""
    client = get_llm_client()
    response = await client.chat_completion(messages, model=model, **kwargs)
    return response["content"]
