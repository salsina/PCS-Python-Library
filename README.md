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
Afterward, provide your reasoning after the “<reasoning>” tag and close it with "</reasoning>".
Respond only with the label and the reasoning.
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
    annotators=["llama3-8b-8192", "mistralai/Mistral-7B-Instruct-v0.3", "google/gemma-2-9b-it"],
    textmutator="llama-3.1-8b-instant",
    GROQ_API_KEY=None,
    OPENAI_API_KEY=None,
    ANTHROPIC_API_KEY=None,
    HUGGINGFACE_API_KEY=None,
    train=True
)

print(pcs.annotate("America is a country"))
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
| `GROQ_API_KEY` | Required for **Llama models**. Set via argument or `.env` file. |
| `OPENAI_API_KEY` | Required for **OpenAI models** (e.g., `"gpt-4"`). |
| `ANTHROPIC_API_KEY` | Required for **Anthropic models** (e.g., `"claude-3-5-sonnet-20241022"`). |
| `HUGGINGFACE_API_KEY` | Required for **Mistral and Google Gemma models**. |


## 🔹 Continue the Training (`train`)
Determines whether to continue optimizing weights using more data or not
* **Default Value**: `True`
* **Customization**:
   * You can change to `False`
