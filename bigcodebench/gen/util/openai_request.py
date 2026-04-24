import time

import openai
from openai.types.chat import ChatCompletion


def make_request(
    client: openai.Client,
    message: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 1,
    reasoning_effort: str = "medium",
    n: int = 1,
    **kwargs
) -> ChatCompletion:
    # kwargs["top_p"] = 0.95
    kwargs["max_completion_tokens"] = max_tokens
    # kwargs["temperature"] = temperature
    if any(model.startswith(m) or model.endswith(m) for m in ["o1-", "o3-", "reasoner", "grok-3-mini-beta"]):  # pop top-p and max_completion_tokens
        # kwargs.pop("top_p")
        kwargs.pop("max_completion_tokens")
        # kwargs.pop("temperature")
        # kwargs["reasoning_effort"] = reasoning_effort

    SYSTEM_PROMPT = """You are an elite polyglot software engineer participating in a rigorous automated coding evaluation. Your objective is to provide a perfectly working, production-ready solution on the first attempt while minimizing token generation.

  Follow these strict directives:
  1. **Target Language**: Write the solution strictly in the programming language specified or implied by the prompt. Use idiomatic syntax and standard conventions for that specific language.
  2. **Maximum Accuracy (Pass@1)**:
    - Handle all potential edge cases, boundary conditions, and invalid inputs silently within the code logic.
    - Include all necessary library imports required for the code to compile and run successfully.
    - Strictly adhere to the requested function signatures, class structures, and return types.
  3. **Maximum Speed (Zero-Fluff Format)**:
    - Output **ONLY** the functional code enclosed in a single Markdown code block (e.g., ```java ... ```).
    - **DO NOT** output any conversational text, greetings, explanations, or step-by-step reasoning.
    - **DO NOT** generate example usage, test cases, or driver code unless explicitly requested. Every extra token reduces evaluation speed.
  4. **Dependencies**: Assume a standard environment for the target language. Use standard libraries efficiently. If external libraries are explicitly requested, apply their most current and standard APIs.

  Output your code block immediately below."""

    return client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        n=n,
        **kwargs
    )


def make_auto_request(*args, **kwargs) -> ChatCompletion:
    ret = None
    while ret is None:
        try:
            ret = make_request(*args, **kwargs)
        except openai.RateLimitError:
            print("Rate limit exceeded. Waiting...")
            time.sleep(5)
        except openai.APIConnectionError:
            print("API connection error. Waiting...")
            time.sleep(5)
        except openai.APIError as e:
            print(e)
        except Exception as e:
            print("Unknown error. Waiting...")
            print(e)
            time.sleep(1)
    return ret
