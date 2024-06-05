# LLM Frameworks

Large Language Models have seen breakthrough successes with models such as OpenAI's ChatGPT or Google's Gemini, and many others! 
These frameworks facilitate building applications around Large Language Models. Please see the following list for more information about each of the partners featured in this repository:

- DSPy
- LangChain
- LlamaIndex
- Semantic Kernel

  This repository contains a mix of Jupyter Notebook cookbooks as well as helpful recipes such as the following dspy Signature,

```python
class GenerateAnswer(dspy.Signature):
  """Assses the context and answer the question"""

  context = dspy.InputField()
  question = dspy.InputField()
  answer = dspy.OutputField()
```

Or the following LangChain LCEL Output Parser:

```python
class UpdatedPropertyValue(BaseModel):
    property_value: str

parser = PydanticOutputParser(pydantic_object=UpdatedPropertyValue)

prompt = PromptTemplate(
    template="Given the references: {references}. Update the property {property_name} by following the instruction, {instruction}. Respond with JSON with the key `property_value`.",
    input_variables=["property_name", "references", "instruction"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

prompt_and_model = prompt | model
output = prompt_and_model.invoke({
    "instruction": "Please write a compelling description for this AirBnB Listing",
    "references": "price-$300,neighborhood-Brooklyn,bedrooms=2,bathrooms=2",
    "property_name": "description"
})
parser.invoke(output)
```

## Interested in Contributing? Please see our [PR template!](https://github.com/weaviate/recipes/blob/main/.github/PULL_REQUEST_TEMPLATE.md)
