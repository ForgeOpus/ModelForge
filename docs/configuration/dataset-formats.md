# Dataset Formats

Learn how to prepare your data for training with ModelForge.

## Overview

ModelForge uses **JSONL** (JSON Lines) format for training datasets. Each line is a valid JSON object representing one training example.

## General Format

```jsonl
{"field1": "value1", "field2": "value2"}
{"field1": "value1", "field2": "value2"}
```

- One JSON object per line
- No commas between lines
- UTF-8 encoding
- File extension: `.jsonl`

## Task-Specific Formats

### Text Generation

**Use Case**: Chatbots, instruction following, code generation, creative writing

**Format**:
```jsonl
{"input": "What is machine learning?", "output": "Machine learning is a subset of artificial intelligence..."}
{"input": "Explain neural networks", "output": "Neural networks are computing systems inspired by biological neural networks..."}
{"input": "Write a Python function to sort a list", "output": "Here's a Python function:\n\ndef sort_list(lst):\n    return sorted(lst)"}
```

**Required Fields**:
- `input` (string): The prompt or instruction
- `output` (string): The expected response

**Example Dataset** (customer support):
```jsonl
{"input": "How do I reset my password?", "output": "To reset your password:\n1. Click 'Forgot Password' on the login page\n2. Enter your email address\n3. Check your email for reset link\n4. Follow the link and create new password"}
{"input": "Where is my order?", "output": "To track your order:\n1. Log into your account\n2. Go to 'My Orders'\n3. Click on the order number\n4. View tracking information"}
```

### Summarization

**Use Case**: Document summarization, article condensing, meeting notes

**Format**:
```jsonl
{"input": "Long article or document text here...", "output": "Concise summary here."}
```

**Required Fields**:
- `input` (string): The long text to summarize
- `output` (string): The summary

**Example Dataset** (news summarization):
```jsonl
{"input": "The Federal Reserve announced today that it will maintain interest rates at their current level of 5.25-5.50%, citing ongoing concerns about inflation despite recent economic slowdowns. Federal Reserve Chair Jerome Powell stated in a press conference that the central bank remains data-dependent and will adjust policy as needed. Markets reacted positively to the news, with the S&P 500 gaining 1.2% in afternoon trading.", "output": "The Federal Reserve kept interest rates unchanged at 5.25-5.50% due to inflation concerns. Chair Powell emphasized data-dependent approach. Markets rose 1.2%."}
{"input": "Scientists at MIT have developed a new battery technology that could potentially triple the range of electric vehicles. The breakthrough involves using solid-state electrolytes instead of traditional liquid electrolytes, which allows for higher energy density and improved safety. The research team, led by Professor Jane Smith, published their findings in Nature Energy this week. Commercial applications are expected within 5-10 years.", "output": "MIT researchers developed solid-state battery technology that could triple EV range. The innovation improves energy density and safety. Commercial use expected in 5-10 years."}
```

### Extractive Question Answering

**Use Case**: RAG systems, document search, FAQ systems

**Format**:
```jsonl
{"context": "Background information and document text", "question": "Question about the context", "answer": "Exact answer from context"}
```

**Required Fields**:
- `context` (string): The paragraph or document containing the answer
- `question` (string): The question being asked
- `answer` (string): The answer extracted from context

**Example Dataset** (FAQ):
```jsonl
{"context": "ModelForge is a no-code toolkit for fine-tuning Large Language Models on your local GPU. It supports text generation, summarization, and question answering tasks. The tool uses LoRA (Low-Rank Adaptation) for efficient fine-tuning and supports both HuggingFace and Unsloth providers.", "question": "What tasks does ModelForge support?", "answer": "text generation, summarization, and question answering"}
{"context": "To install ModelForge, you need Python 3.11, an NVIDIA GPU with at least 4GB VRAM, CUDA installed, and a HuggingFace account. The installation process involves running 'pip install modelforge-finetuning' and then setting up your HuggingFace token.", "question": "What are the prerequisites for ModelForge?", "answer": "Python 3.11, NVIDIA GPU with 4GB+ VRAM, CUDA, HuggingFace account"}
```

## Preference Data (DPO/RLHF)

**Use Case**: Aligning model with human preferences (DPO or RLHF strategy)

**Format**:
```jsonl
{"prompt": "What is the capital of France?", "chosen": "The capital of France is Paris.", "rejected": "I don't know."}
{"prompt": "Explain quantum computing", "chosen": "Quantum computing uses quantum bits...", "rejected": "It's complicated."}
```

**Required Fields**:
- `prompt` (string): The input prompt or question
- `chosen` (string): The preferred/better response
- `rejected` (string): The non-preferred/worse response

> **Note**: When using DPO or RLHF strategy, this format is required instead of the task-specific format above. Both strategies also require `"task": "text-generation"`.

## Dataset Size Recommendations

| Model Size | Min Examples | Recommended | Optimal |
|------------|--------------|-------------|---------|
| < 1B params | 100 | 500 | 1,000+ |
| 1B-3B params | 200 | 1,000 | 5,000+ |
| 3B-7B params | 500 | 2,000 | 10,000+ |
| 7B+ params | 1,000 | 5,000 | 20,000+ |

**Quality > Quantity**: 100 high-quality examples are better than 10,000 low-quality ones.

## Data Quality Guidelines

### DO:

✅ Use clean, grammatically correct text  
✅ Ensure input-output pairs are logically related  
✅ Include diverse examples covering different scenarios  
✅ Use consistent formatting across examples  
✅ Remove personal information (PII)  
✅ Verify all data is relevant to your use case  

### DON'T:

❌ Include malformed JSON  
❌ Use inconsistent field names  
❌ Include duplicate or near-duplicate examples  
❌ Mix different tasks in one dataset  
❌ Use copyrighted content without permission  
❌ Include biased or harmful content  

## Creating Your Dataset

### Method 1: Manual Creation

Create a file `dataset.jsonl`:

```jsonl
{"input": "Example 1 input", "output": "Example 1 output"}
{"input": "Example 2 input", "output": "Example 2 output"}
```

### Method 2: Python Script

```python
import json

data = [
    {"input": "What is AI?", "output": "AI stands for Artificial Intelligence..."},
    {"input": "Explain ML", "output": "Machine Learning is..."},
]

with open('dataset.jsonl', 'w', encoding='utf-8') as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
```

### Method 3: Convert from CSV

```python
import pandas as pd
import json

# Read CSV
df = pd.read_csv('data.csv')

# Convert to JSONL
with open('dataset.jsonl', 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        item = {"input": row['question'], "output": row['answer']}
        f.write(json.dumps(item, ensure_ascii=False) + '\n')
```

### Method 4: From HuggingFace Dataset

```python
from datasets import load_dataset
import json

# Load dataset
dataset = load_dataset("squad", split="train[:1000]")

# Convert to JSONL
with open('dataset.jsonl', 'w', encoding='utf-8') as f:
    for item in dataset:
        formatted = {
            "context": item['context'],
            "question": item['question'],
            "answer": item['answers']['text'][0]
        }
        f.write(json.dumps(formatted, ensure_ascii=False) + '\n')
```

## Validation

ModelForge automatically validates datasets before training:

✅ JSON syntax validation  
✅ Required fields check  
✅ Minimum size check (at least 10 examples)  
✅ Field type validation  

### Common Validation Errors

**Error**: `Missing required field 'output'`
- **Fix**: Ensure all examples have required fields

**Error**: `Invalid JSON on line 42`
- **Fix**: Check line 42 for syntax errors (missing quotes, commas, etc.)

**Error**: `Dataset too small (5 examples, minimum 10)`
- **Fix**: Add more examples to your dataset

**Error**: `Field 'input' must be a string`
- **Fix**: Ensure all field values are strings, not numbers or objects

## Advanced Features

### Multi-line Text

Use `\n` for line breaks:

```jsonl
{"input": "Write a haiku", "output": "Code flows like water\nBugs hide in shadows unseen\nDebug, test, deploy"}
```

### Special Characters

Escape special characters:

```jsonl
{"input": "What is JSON?", "output": "JSON uses \"quotes\" for strings and {\"key\": \"value\"} for objects"}
```

### Unicode Support

Full UTF-8 support:

```jsonl
{"input": "Translate: Hello", "output": "你好 (Chinese), Hola (Spanish), Bonjour (French)"}
```

### Long Context

No hard limit on length, but consider model's max sequence length:

```jsonl
{"input": "Summarize this article", "output": "Summary here"}
```

## Sample Datasets

ModelForge includes sample datasets for testing:

```bash
# Located in: ModelForge/test_datasets/
low_text_generation.jsonl          # Text generation examples
low_summarization_train_set.jsonl  # Summarization examples
low_qa_train_set.jsonl             # QA examples
```

Download from repository:
```bash
curl -O https://raw.githubusercontent.com/forgeopus/modelforge/main/ModelForge/test_datasets/low_text_generation.jsonl
```

## Best Practices

### 1. Data Splitting

ModelForge automatically splits data into train/validation sets:

```json
{
  "eval_split": 0.2  // 20% for validation, 80% for training
}
```

### 2. Data Balancing

Ensure balanced representation:
- Equal distribution of topics
- Diverse input lengths
- Varied complexity levels

### 3. Data Cleaning

Before creating JSONL:
1. Remove duplicates
2. Fix typos and grammar
3. Normalize formatting
4. Remove irrelevant examples

### 4. Iterative Improvement

1. Start with small dataset (100-500 examples)
2. Train and evaluate
3. Identify weak areas
4. Add targeted examples
5. Repeat

## Troubleshooting

### Dataset Won't Upload

**Problem**: Upload fails

**Checks**:
1. File is valid JSONL (one JSON object per line)
2. File size < 500MB
3. Proper UTF-8 encoding
4. No special characters in filename

### Training Fails with Dataset Error

**Problem**: Training starts but fails immediately

**Checks**:
1. All required fields present
2. Fields are correct type (strings)
3. No empty values
4. No extremely long examples (> max_seq_length)

### Poor Training Results

**Problem**: Model doesn't learn effectively

**Solutions**:
1. Add more examples (aim for 1,000+)
2. Improve data quality
3. Ensure examples are representative
4. Check for data leakage or duplicates

## Next Steps

- **[Configuration Guide](configuration-guide.md)** - Learn about training parameters
- **[Training Tasks](training-tasks.md)** - Understand different task types
- **[Quick Start](../getting-started/quickstart.md)** - Train your first model
- **[Troubleshooting](../troubleshooting/common-issues.md)** - Common issues

---

**Good data is the foundation of good models!** 📊
