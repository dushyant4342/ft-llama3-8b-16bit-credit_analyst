
### **Project Retrospective: Building a Specialized AI Credit Analyst with Llama 3**

Excited to share a summary of our recent project where we transformed a powerful generalist LLM into a specialized financial expert! Our goal was to fine-tune `meta-llama/Meta-Llama-3-8B-Instruct` to analyze complex credit reports and generate simple, actionable summaries.

Here's a look at our journey from start to finish:

**1. Data Engineering:** The foundation of any great model is great data. We started by processing raw credit reports into a structured, fine-tuning dataset. Each example contained:
* **`customer_info`:** A detailed "before and after" fact sheet.
* **`customer_credit_update`:** A human-written narrative summary of the key changes.

**2. Efficient Fine-Tuning:** We used PEFT/LoRA (`r=64`, `alpha=128`) to efficiently adapt the base model. This allowed us to teach it the new skill of credit analysis by training only a tiny fraction (~0.4%) of its total parameters, saving significant time and compute.

**3. Mastering the Prompt:** This was our biggest challenge and learning.
* **Problem:** The model initially defaulted to its fine-tuned `Good:-/Bad:-` format, ignoring conversational instructions.
* **Solution:** We implemented a powerful **two-step pipeline**. First, we use our fine-tuned model for what it does best: accurately extracting structured facts. Then, we feed those facts to the original base Llama 3 model to synthesize them into a polished, conversational paragraph. This proved far more reliable than a single, complex prompt.

**4. Rigorous Evaluation:** We didn't just guess if it was working. We validated performance on a holdout test set, achieving excellent results that confirmed the model's accuracy and coherence:
* **ROUGE-L Score:** > 0.67 (High structural similarity to human summaries)
* **BERTScore F1:** > 0.90 (High semantic understanding)

**5. High-Speed Inference with vLLM:** To make the model ready for real-world applications, we deployed it using vLLM. This dramatically increased throughput and reduced latency, making inference blazing-fast. vLLM runs batches in parallel, so 20 generations still take ~7s (same as 1) if GPU allows. Its smart batching (PagedAttention + continuous filling) boosts throughput by 20–30x vs standard transformers.

**6. Sharing with the Community:** The final, merged model was uploaded to the Hugging Face Hub, complete with a detailed model card explaining its capabilities and how to use it.

### **What Could Be Done Better?**

To make the model even better at listening to user commands (like "How can I improve my score?"), the next step would be to create a **secondary fine-tuning dataset**. This dataset would contain examples of direct user questions and the ideal, actionable answers based on the credit data. This would explicitly train the model on the conversational Q&A behavior we prompted it for.

This project was a fantastic journey through the entire lifecycle of creating a specialized AI model!

#AI #LLM #FineTuning #Llama3 #DataScience #MachineLearning #Fintech #HuggingFace #vLLM #NLP













If one generation takes ~7 seconds in vLLM, and you're using batching, then:
vLLM processes multiple generations in parallel — not sequentially.
So if you send 20 summaries in a single batch, total time will still be approximately 7 seconds:
✅ vLLM parallelizes the 20 requests → processes all at once → same time as 1 request.
Batch size ≤ GPU memory capacity → All requests run in parallel




**vLLM boosts batch inference dramatically**
A single request is ~3x faster with vLLM (7.4s vs 25s).
But the real power is in throughput(requests processed per second):
100 requests = ~20 mins without vLLM
With vLLM = < 1 min — a 20–30x speedup
Why so fast?
✅ PagedAttention = smarter GPU memory use
✅ Continuous batching = fills GPU instantly, no idle time

Throughput = number of requests processed per second (not per batch).
If batch size = 20, and those 20 are processed in 2 seconds, then
→ Throughput = 10 requests/sec
If batch size = 1, and it takes 2 seconds per request, then
→ Throughput = 0.5 requests/sec

🔑 With vLLM, throughput is higher because: It processes many requests in parallel (via continuous batching). The GPU stays fully utilized.


Without vLLM, running them one by one would take:
100 requests * ~12 seconds/request = ~20 minutes

With vLLM, it would handle these requests in a continuous batch:
100 requests(in one batch) processed concurrently = Likely under 1 minute
This isn't a 2x speedup; it's a 20-30x (or more) increase in throughput.

Why is vLLM So Much Faster in Batches?
PagedAttention: vLLM uses a technology called PagedAttention, which is a more efficient way to manage the GPU's memory. It eliminates wasted space and prevents the bottlenecks that slow down standard transformers.

Continuous Batching: Instead of waiting for a whole batch of requests to finish, vLLM continuously adds new requests to the GPU as soon as there is space. This keeps the GPU running at nearly 100% utilization, maximizing its processing power.


