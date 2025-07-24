---
language: en
license: apache-2.0
library_name: transformers
tags:
- llama3
- finetuned
- credit-analysis
- lora
base_model:
- meta-llama/Meta-Llama-3-8B-Instruct
pipeline_tag: text-generation
---

# Llama-3-8B Fine-Tuned for Credit Analysis

## Model Description
This model is a fine-tuned version of `meta-llama/Meta-Llama-3-8B-Instruct`, specifically adapted to function as an expert credit analyst. It is trained to analyze structured credit report data (representing a "before" and "after" snapshot, two months of consecutive credit report) and generate a concise, balanced summary of the most significant positive and negative changes.
This model excels at translating complex, multi-account credit data into an easy-to-understand narrative, making it ideal for quickly assessing a user's credit profile updates.

**Developed by:** Dushyant4342@gmail.com
**Model type:** Causal Language Model
**Language(s):** English
**Finetuned from model:** `meta-llama/Meta-Llama-3-8B-Instruct`

---

## Training Details
Training Data: 
The model was fine-tuned on a private, proprietary dataset of approximately 10,000 anonymized customer credit profiles. Each profile contained a structured text 'before' and 'after' snapshot of their credit information, and the target completion was a narrative summary of the changes.

Training Procedure: 
The model was trained using the trl.SFTTrainer from the TRL library. The training data was formatted into the Llama-3-Instruct chat template, where the customer_info was the user turn and the customer_credit_update was the assistant turn.

Low-Rank Adaptation (LoRA) was used to efficiently fine-tune the model, adding only a small number of trainable parameters (~0.4% of the total) to the base model. LoRA adapters (rank r=64, alpha=16 (128) ) were injected into the attention layers' query, key, value, and output projection matrices (q_proj, k_proj, v_proj, o_proj) throughout the model.

Training Hyperparameters: 

Learning Rate: 2e-4 or 5e-5

Epochs: 1

Effective Batch Size: 8 (Batch Size: 2 per device, Gradient Accumulation: 4)

Precision: bfloat16

Max Sequence Length: 4096

Environmental Impact

Hardware Type: A100 GPU (via Google Colab)

Cloud Provider: Google Cloud

---

## How to Get Started with the Model

The model requires a specific prompt structure, including a system prompt to define its expert role and a user prompt containing both a command and the structured data.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = "Dushyant4342/ft-llama3-8b-credit-analyst"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# 1. Define the expert system prompt
system_prompt = """You are an expert credit analyst. Your role is to analyze a customer's credit data and generate a highly concise summary of the most important positive and negative changes. You must adhere to the length and format constraints given by the user."""

# 2. Provide the customer data (as a structured string)
customer_data = """--- Credit Profile Report for Customer: 852923... ---

## Key Metric Summary
-  Risk Score : 715.0 (was 760.0)
-  Overall Utilization : 65.30% (was 45.10%)
-  Credit Card Utilization : 88.50% (was 52.00%)
-  Total Active Accounts : 3.0 (was 4.0)

## Current Credit Mix
-  Total Accounts : 12 (3.0 active)
-  Active Secured Products : 2
-  Active Unsecured Products : 1
-  Active Lender Distribution : Public(1), Private(2), NBFC(1), Corporate(0), Foreign(0)

## Payment Behavior History
-  Past Delinquencies : User has been delinquent on 1 accounts in their history.
-  Highest Past Delinquency : 35.0 days on a  Credit Card with SBI.

## Account Details Breakdown

-  Account : HDFC BANK -  Housing Loan (#4512)
  -  Status : Active
  -  DPD : 0.0 days (was 0.0 days)
  -  Utilization : 34.51% (was 35.02%)
... (and so on) ...
"""

# 3. Give a specific command (Examples)
user_command = "Start your summary with the credit score change, then list the main reasons for it."
#user_command = "Summarize the key takeaways in two bullet points one Good & other Bad."


# 4. Format the final prompt for the model
user_content = f"{user_command}\n\n--- DATA ---\n{customer_data}"

msgs = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_content},
]

prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# 5. Generate the response
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    )

response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
print(response)

The credit score dropped by 45 points, primarily due to a new 35-day delinquency on a credit card. On a positive note, a personal loan was recently closed.
```

### Direct Use
The primary use of this model is for automated credit report summarization. It requires a structured input detailing a customer's credit profile and can generate narrative summaries based on user commands.

### Out-of-Scope Use
This model should **NOT** be used for:
- Making final, automated credit approval or rejection decisions without human oversight.
- Providing financial advice to consumers.
- Any application that violates data privacy or financial regulations.

---

## Bias, Risks, and Limitations
This model's knowledge is strictly limited to the patterns observed in its private training data. It does not possess real-time financial information. The summaries it generates are based on the provided data and may reflect biases inherent in the training dataset. It should be used as an analytical assistant, not as a definitive decision-making tool.
