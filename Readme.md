# Project: Prompt Engineering with LLM Whisperers

## Introduction

Prompt engineering is the process of strategically designing inputs (prompts) to guide pre-trained language models (LMs) toward performing specific tasks without retraining the entire model. This project focuses on **prefix tuning**, a technique that optimizes task-specific continuous vectors (prefixes) added to inputs, without modifying the LM's internal parameters. The aim is to explore the efficacy of this lightweight tuning method, particularly for **language translation tasks** like Hindi-English and French-English.

This project was developed as part of a **Generative AI course**, where I worked alongside the LLM Whisperers team to investigate ways to reduce the computational overhead of adapting large language models to new tasks.

## Objectives

- Evaluate the performance of **prefix tuning** in steering language models for specific translation tasks.
- Analyze translation quality between **Hindi-English** and **French-English** datasets using various models like GPT, BERT, and BARD.

## Background

Prefix tuning is an alternative to traditional **fine-tuning**, which requires retraining all parameters of a model for each new task, resulting in high computational costs. **Prefix tuning** freezes the LM's parameters and optimizes only a small set of task-specific parameters, or prefixes, that influence the output. This allows the original model to be reused across various tasks, making it highly modular and space-efficient.

Inspired by the concept of prompting, where task instructions are prepended to the input, prefix tuning utilizes continuous vectors (virtual tokens) that guide the LM through the task without altering its architecture. 

The project uses **prefix tuning** to improve translation accuracy between **Hindi-English** and **French-English**. The Hindi-English dataset consists of 1.56 million sentence pairs from the IIT Bombay dataset, while the French-English dataset includes over 22.5 million sentence pairs. By analyzing the improvements in **BLEU** and **METEOR** scores, the project aims to measure the quality and efficiency of prefix tuning for translation tasks.

## Experimental Setup

- **Prefix Tuning**: Task-specific vectors are appended as prefixes to the input, which the LM uses to adjust its output. This method keeps the underlying architecture of the model frozen.
  
- **Datasets**:
  - **Hindi-English**: The IIT Bombay English-Hindi dataset (1,561,840 instances).
  - **French-English**: A larger dataset with 22.5 million sentence pairs.
  
- **Models**: Various LMs were tested, including **GPT**, **BERT**, and **BARD**, to evaluate their adaptability with prefix tuning.

## Methodology

The core idea behind prefix tuning is to prepend task-specific continuous vectors to model inputs. These vectors serve as a "soft prompt," influencing the LM’s internal states and guiding its responses.

Key steps include:
1. **Prefix Matrix Creation**: A task-specific prefix matrix is developed for each language pair, which is attached to the input sequence before the tokens are processed by the LM.
2. **Optimization**: Various prompt lengths and strategies are tested to balance translation quality and computational efficiency.
3. **Comparison with Baseline Models**: The impact of prefix tuning is evaluated in comparison to baseline models, both in terms of translation accuracy and computational cost.

## Metrics of Evaluation

- **BLEU Score**: Measures the similarity of machine-generated translations to human translations.
- **METEOR Score**: Focuses on semantic accuracy, accounting for synonyms and paraphrases.

## Challenges

- Managing large computational requirements for tuning large language models.
- Balancing the prefix length to optimize for both accuracy and efficiency in translation tasks.

## Results and Observations

- Prefix tuning outperformed **k-shot models** with notable improvements in BLEU and METEOR scores.
- **Longer prefixes** and more **iterations** resulted in higher-quality translations, especially for the **French-English** pair, which benefited from its larger dataset size.
- The method showed greater improvements for **French-English** translation compared to **Hindi-English**, likely due to the dataset size discrepancy.

## Conclusion

The project successfully demonstrated that **prefix tuning** is an effective and efficient alternative to full model retraining, particularly for translation tasks. With minimal task-specific parameters, prefix tuning can significantly improve model performance without requiring substantial computational resources. Future work will focus on further optimizations, expanding the method to additional language pairs, and exploring other tasks.

## Acknowledgments

Special thanks to all team members and contributors, as well as the data providers and computational resource suppliers. This project was developed as part of a **Generative AI course**.

---

### Detailed Explanation of Prefix Tuning (For Interviews)

**Prefix Tuning** is a method that allows the adaptation of large language models (LLMs) for specific tasks without modifying their parameters. Unlike fine-tuning, where the entire model is updated for each task, prefix tuning introduces a small task-specific set of continuous vectors (called prefixes) before the actual input. These prefixes act as "virtual tokens" that guide the model's output while preserving the original model's structure.

**Why is it Useful?**
Prefix tuning is highly efficient as it only modifies a tiny fraction of the model’s parameters, making it computationally cheaper and faster. Additionally, it is **modular**—a single model can be used for many tasks simply by adding new prefixes, rather than training multiple models.

**Key Observations in Our Project**:
- Prefix tuning outperformed models using a few-shot approach, showing higher BLEU and METEOR scores.
- **Longer prefixes** improved translation accuracy, particularly for **French-English**, which had a larger dataset.
- **Hindi-English** translations saw less improvement, suggesting that dataset size plays a significant role in the success of prefix tuning.
- By keeping the model's parameters frozen, the method is more **space-efficient** and adaptable to various tasks.

**Challenges** included balancing the prefix length to avoid overfitting or underfitting the model to specific tasks and managing computational demands for larger datasets.

This approach proves to be a promising avenue for adapting LLMs to new tasks with minimal resources, making it highly practical for industry applications, where retraining models from scratch can be prohibitively expensive.
