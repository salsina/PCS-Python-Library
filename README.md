# PCS Annotator: A Python Library for Classification Tasks Using Multiple LLMs

## How to Install

```bash
pip install pcs-annotator
```

## How to Use

```python
from pcs_annotator import PCS
```

### Create a Prompt for Classification Tasks

```python
prompt = """Analyze the news article and determine whether it is 'Fake' or 'Real.'
Label the article using the tag format: <label>Fake</label> or <label>Real</label>.
Respond only with the label.
"""
```

### Training Dataset Path (`dataset_path`)

You will need a CSV file with two columns: `label` and `text` to ensure better annotation, as the data is used to train the LLMs.




## 🔧 Customizing Hyperparameters

The `PCS` class allows you to customize various hyperparameters, including the choice of annotator models, text mutator models, and API keys for different LLM providers.

### 📌 Example Usage
```python
pcs = PCS(
    prompt="Your classification prompt here",
    dataset_path="path/to/dataset.csv",
    annotators=["llama3-8b-8192", "mistralai/Mistral-7B-Instruct-v0.3", "gemma-2-9b-it"],
    textmutator="llama-3.1-8b-instant",
    GROQ_API_KEY=None,
    OPENAI_API_KEY=None,
    ANTHROPIC_API_KEY=None,
    HUGGINGFACE_API_KEY=None,
    generate_annotations=True,
    Optimizer="LR"  # or "GA"
)

print(pcs.annotate("America is a country"))
```
### 🖨️ Output

The `.annotate()` function returns a dictionary of confidence scores for each label:

```python
{'Real': 0.87, 'Fake': 0.13}
```

# Configuration

## 🔹 Annotators (`annotators`)
Defines the list of LLMs used for annotation.
* **Default Models**:
   * `"llama3-8b-8192"`
   * `"mistralai/Mistral-7B-Instruct-v0.3"`
   * `"google/gemma-2-9b-it"`
* **Customization**:
   * You can add, remove, or modify the annotators by passing a list of model names.

## 🔹 Text Mutator (`textmutator`)
Determines the LLM used for generating text mutations.
* **Default Model**: `"llama-3.1-8b-instant"`
* **Customization**:
   * You can replace it by passing a different model name as a string.

## 🔹 API Keys
To access certain LLM models, you need to provide API keys. These can be passed as arguments during initialization or set as environment variables in a `.env` file.

| API Key | Purpose |
|---------|---------|
| `GROQ_API_KEY` | Required for **Llama and Google Gemma models**. Set via argument or `.env` file. |
| `OPENAI_API_KEY` | Required for **OpenAI models** (e.g., `"gpt-4"`). |
| `ANTHROPIC_API_KEY` | Required for **Anthropic models** (e.g., `"claude-3-5-sonnet-20241022"`). |
| `HUGGINGFACE_API_KEY` | Required for **Mistral models**. |


## 🔹 Continue the Training (`generate_annotations`)
Determines whether to continue generating more annotations in the dataset or not
* **Default Value**: `True`
* **Customization**:
   * You can change to `False`



## ⚙️ Optimizer (`Optimizer`)

The `PCS` class supports weight optimization strategies for combining LLM predictions and Metamorphic Relation (MR) outputs. You can choose the optimizer during initialization using the `Optimizer` parameter.

### 🔧 Options

| Value   | Description                                                                 |
|---------|-----------------------------------------------------------------------------|
| `"LR"`  | **Linear Regression** – Optimizes annotator and MR weights via regression. *(default)* |
| `"GA"`  | **Genetic Algorithm** – Uses evolutionary search to find optimal weights.   |



## ➕ Adding a Custom Metamorphic Relation (MR)

You can define and register your own MR by passing a custom prompt-generation function to the `TextMutator` class:

```python
from pcs_annotator.TextMutator import TextMutator

# Define a new MR prompt
def custom_negation_prompt(text):
    return f"""Transform all affirmative statements in the following text into their negated forms.
Start the response with 'new_text:' and include only the revised text.

Original Text:
{text}
"""
```

# Initialize a mutator and register the new MR
mutator = TextMutator(model_name="llama-3.1-8b-instant", token="your_groq_api_key")
mutator.register_mr("custom_negation", custom_negation_prompt)

# Apply the new MR
mutated_text = mutator.MutateText("The cat is on the table.", "custom_negation")
print(mutated_text)
