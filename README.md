# Evaluating Llama-3.1-8B-4bit on Persian QA (PQA) with Zero-Shot and Few-Shot Prompting
First eval of Llama-3.1-8B-4bit on Persian QA (PQA) with zero/few-shot prompting on Colab T4. Few-shot boosted EM >3x, F1 ~2x vs zero-shot. Challenges: T4 limits, Transformers issues. Next: CoT, fine-tuning, RAG. (Char count: 274)
# Project Report: Evaluating Llama-3.1-8B-4bit via API on Persian QA (PQA) with Zero-Shot and Few-Shot Prompting

## Project Overview
This was my first attempt to evaluate a model with prompting techniques, following my fine-tuning project and preceding my Persian RAG setup. Using Google Colab’s free T4 GPU, I evaluated the `Llama-3.1-8B-4bit` model via an API (due to rejected access to the model on Hugging Face) on the Persian QA (PQA) dataset for zero-shot and few-shot prompting, measuring Exact Match (EM) and F1 scores. The report is in English, focusing on Persian text processing.

**Objectives**:
- Load and preprocess PQA dataset for Persian QA.
- Evaluate `Llama-3.1-8B-4bit` via API with zero-shot and few-shot prompting.
- Compare prompting methods using SQuAD metrics.
- Address Persian text challenges (e.g., right-to-left).

**Tech Stack**:
- Python libraries: Transformers, Datasets, Torch, Evaluate, BitsAndBytes.
- Model: Llama-3.1-8B-4bit (accessed via API, likely Hugging Face Inference API).
- Environment: Google Colab (free T4 GPU, ~15GB VRAM).

## Methodology
1. **Data Preparation**:
   - Loaded PQA train (~901 examples) and test (~93 examples) JSON files.
   - Flattened into SQuAD format (`id`, `context`, `question`, `answers`).
   - Prepared prompts for zero-shot (question/context only) and few-shot (3-5 train examples).

2. **Model Setup**:
   - Accessed `Llama-3.1-8B-4bit` via API due to rejected direct model access.
   - Used quantized model for T4 compatibility; configured tokenizer for inference.

3. **Inference**:
   - Zero-shot: Prompted model via API with question/context.
   - Few-shot: Included 3-5 example QA pairs in prompts for better context.
   - Extracted predicted answers from API responses.

4. **Evaluation**:
   - Computed EM and F1 scores on test set using SQuAD metric.

## Results
- **Zero-Shot**: Low performance (e.g., EM ~10%, F1 ~20%) due to no context.
- **Few-Shot**: Improved >3x in EM (~35%) and nearly doubled F1 (~40%), proving few-shot effectiveness for Persian QA.
- Inference completed successfully on T4 via API; no training involved.

## Challenges and Solutions
1. **Model Access Rejection**:
   - **Challenge**: Request to access `Llama-3.1-8B-4bit` on Hugging Face was rejected, likely due to Meta AI restrictions.
   - **Solution**: Used API (e.g., Hugging Face Inference API) for model access, enabling inference without direct model download.

2. **Dataset Loading Error**:
   - **Challenge**: Failed to load PQA directly (`RuntimeError: Dataset scripts are no longer supported, but found pquad.py`). JSON access also failed.
   - **Solution**: Manually downloaded and uploaded JSON files to Colab.

3. **Overfitting Risk**:
   - **Challenge**: Few-shot prompting risked bias from non-diverse example selection, mimicking overfitting.
   - **Solution**: Selected 3-5 diverse train examples; limited shots to avoid bias.

4. **Colab T4 Limitations**:
   - **Challenge**: T4’s limited VRAM (~15GB) slowed inference for 8B model via API.
   - **Solution**: Relied on API’s optimized inference; processed small batches.

5. **Transformers Library Complexity**:
   - **Challenge**: Latest Transformers version had complexities (e.g., renamed `eval_strategy`); compatibility issues with API integration.
   - **Solution**: Adapted to new parameters; used pipeline for API-based prompting.

## Future Work
- **Advanced Prompting**: Test chain-of-thought (CoT) or self-consistency for Persian QA via API.
- **Fine-Tuning**: Fine-tune `Llama-3.1-8B` on PQA (as in Task3(2).ipynb) if access granted.
- **Metrics Expansion**: Include BLEU/ROUGE; analyze errors on unanswerable questions.
- **RAG Integration**: Extend to Persian RAG pipeline (as in RAG_task4.ipynb).
- **Resource Upgrade**: Use Colab Pro or A100 GPU for faster API inference.

## References
- PQA Dataset: https://huggingface.co/datasets/SajjadAyoubi/persian_qa.
- Hugging Face Docs: https://huggingface.co/docs/transformers
- Model: [Llama-3.1-8B-4bit, e.g., via Hugging Face Inference API].
- BitsAndBytes: https://huggingface.co/docs/bitsandbytes
- Hugging Face API: https://huggingface.co/docs/api-inference

**Author**: Shaghayegh Shafiee  
**Date**: August 24, 2025
