# Quick Reference for connecting DSPy to different LLMs

```python
gemini_1_5_pro = dspy.Google(model="gemini-1.5-pro-latest", api_key=google_api_key)
gpt4 = dspy.OpenAI(model="gpt-4o", max_tokens=4_000, model_type="chat")
claude_opus = dspy.Claude(model="claude-3-opus-20240229", api_key=CLAUDE_API_KEY)
command_r_plus = dspy.Cohere(model="command-r-plus", max_input_tokens=32_000, max_tokens=4_000, api_key=cohere_api_key)
ollama_llama3 = dspy.OllamaLocal(model="llama3.1")
```

Say hello test script:

```python
lms = [
    {"name": "Gemini-1.5-Pro", "lm": gemini_1_5_pro
    {"name": "GPT-4", "lm": gpt4},
    {"name": "Claude Opus", "lm": claude_opus},
    {"name": "Command R+", "lm": command_r_plus}
    {"name": "Llama 3.1", "lm": ollama_llama3}
]

connection_prompt = "Please say something interesting about Database Systems intended to impress me with your intelligence."

print(f"\033[91mTesting the prompt:\n{connection_prompt}\n")

for lm_dict in lms:
    lm, name = lm_dict["lm"], lm_dict["name"]
    with dspy.context(lm=lm):
        print(f"\033[92mResult for {name}\n")
        print(f"\033[0m{lm(connection_prompt)[0]}\n")
```
