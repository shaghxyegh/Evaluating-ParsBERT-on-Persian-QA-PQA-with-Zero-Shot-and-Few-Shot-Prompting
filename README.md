# Evaluating Llama-3.1-8B-4bit on Persian QA (PQA) with Zero-Shot and Few-Shot Prompting
First eval of Llama-3.1-8B-4bit via API (access rejected) on SQuAD using Colab T4. Fixed dataset errors with JSON upload. Few-shot boosted EM >3x, F1 ~2x vs zero-shot. Challenges: T4 limits, Transformers issues. 
# Project Report: Evaluating Llama-3.1-8B-4bit via API on SQuAD with Zero-Shot and Few-Shot Prompting

## Project Overview
This was my first attempt to evaluate a model with prompting techniques, preceding my fine-tuning and Persian RAG projects. Using Google Colab’s free T4 GPU, I evaluated the `Llama-3.1-8B-4bit` model via an API (due to rejected access on Hugging Face) on the SQuAD dataset for zero-shot and few-shot prompting, measuring Exact Match (EM) and F1 scores. The report is in English, focusing on English question-answering performance.

**Objectives**:
- Load and preprocess SQuAD dataset for QA.
- Evaluate `Llama-3.1-8B-4bit` via API with zero-shot and few-shot prompting.
- Compare prompting methods using SQuAD metrics.
- Address computational constraints and API integration.

**Tech Stack**:
- Python libraries: Transformers (v4.55.3), Datasets, Torch, Evaluate, BitsAndBytes.
- Model: Llama-3.1-8B-4bit (accessed via API, e.g., Hugging Face Inference API).
- Environment: Google Colab (free T4 GPU, ~15GB VRAM).

## Methodology
1. **Data Preparation**:
   - Loaded SQuAD train (87,599 examples) and dev (10,570 examples) from Hugging Face (`squad` dataset).
   - Formatted into SQuAD structure (`id`, `context`, `question`, `answers`).
   - Prepared prompts for zero-shot (question/context only) and few-shot (3-5 train examples).

2. **Model Setup**:
   - Accessed `Llama-3.1-8B-4bit` via API due to rejected direct model access.
   - Used quantized model for T4 compatibility; configured tokenizer for inference.

3. **Inference**:
   - Zero-shot: Prompted model via API with question/context.
   - Few-shot: Included 3-5 example QA pairs from train set in prompts for better context.
   - Extracted predicted answers from API responses.

4. **Evaluation**:
   - Computed EM and F1 scores on dev set using SQuAD metric.

## Results
- **Zero-Shot**: Low performance (e.g., EM ~10%, F1 ~20%) due to lack of context.
- **Few-Shot**: Improved >3x in EM (~35%) and nearly doubled F1 (~40%), proving few-shot effectiveness for QA.
- Inference completed successfully on T4 via API; no training involved.

## Challenges and Solutions
1. **Model Access Rejection**:
   - **Challenge**: Request to access `Llama-3.1-8B-4bit` on Hugging Face was rejected, likely due to Meta AI restrictions.
   - **Solution**: Used API (e.g., Hugging Face Inference API) for model access, enabling inference without direct model download.

2. **Dataset Loading Error**:
   - **Challenge**: Failed to load SQuAD directly due to connectivity or configuration issues; encountered errors with dataset scripts.
   - **Solution**: Manually downloaded SQuAD JSON files (`train-v1.1.json`, `dev-v1.1.json`) and uploaded to Colab for loading via `load_dataset("json", ...)`.

3. **Overfitting Risk**:
   - **Challenge**: Few-shot prompting risked bias from non-diverse example selection, mimicking overfitting.
   - **Solution**: Selected 3-5 diverse train examples; limited shots to avoid bias.

4. **Colab T4 Limitations**:
   - **Challenge**: T4’s limited VRAM (~15GB) slowed inference for 8B model via API.
   - **Solution**: Relied on API’s optimized inference; processed small batches to manage memory.

5. **Transformers Library Complexity**:
   - **Challenge**: Latest Transformers version (v4.55.3) had complexities (e.g., renamed `eval_strategy`); compatibility issues with API integration.
   - **Solution**: Adapted to `eval_strategy`; used pipeline for API-based prompting to simplify integration.

## Future Work
- **Advanced Prompting**: Test chain-of-thought (CoT) or self-consistency for improved QA performance via API.
- **Fine-Tuning**: Fine-tune `Llama-3.1-8B` on SQuAD or PersianQA (as in Task3(2).ipynb) if access granted.
- **Metrics Expansion**: Include BLEU/ROUGE; analyze errors on complex questions.
- **RAG Integration**: Extend to Persian RAG pipeline (as in RAG_task4.ipynb).
- **Resource Upgrade**: Use Colab Pro or A100 GPU for faster API inference.

## References
- SQuAD Dataset: https://huggingface.co/datasets/rajpurkar/squad
- Hugging Face Docs: https://huggingface.co/docs/transformers
- Model: [Llama-3.1-8B-4bit, e.g., via Hugging Face Inference API]
- BitsAndBytes: https://huggingface.co/docs/bitsandbytes
- Hugging Face API: https://huggingface.co/docs/api-inference

**Author**: Shaghayegh Shafiee  
**Date**: August 24, 2025

