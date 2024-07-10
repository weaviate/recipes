# Gemini Python SDK

Learn more about the Gemini Python client [here!](https://ai.google.dev/gemini-api/docs/get-started/tutorial?lang=python)

### Connect

```python
import google.generativeai as genai

genai.configure(api_key=MY_GOOGLE_API_KEY)

model = genai.GenerativeModel("gemini-1.5-pro")
```

### Generate Text

```python
response = model.generate_content(prompt)

print(response.text)
```
