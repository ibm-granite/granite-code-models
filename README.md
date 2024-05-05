<p align="center">
  <img src="figures/granite-code-models-banner_1x.png" />
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330"> Models Download</a>&nbsp | <a href="http://"> Blog</a>&nbsp | <a href="https://">Paper Link 👁️</a>&nbsp
<br>

---
## Introduction to Granite Code Models
Granite Code Models are a family of code models ready for enterprise commercial use for three important reasons: 1) all models are released under [Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0), 2) our models were built by following transparent data collection and filtering practices, and 3) our models perform better or the same than competitive models that are twice their size, requiring less computational resources without an impact on performance.

Our family of code-specialized enterprise-ready models comes in two main variants:

1. Granite Code Base: Base foundational models designed for general code generative tasks (e.g., code repair, code explanation, code synthesis). 
2. Granite Code Instruct: Instruction following models finetuned using Git commits paired with human instructions.

All variants are available in sizes of 3B, 8B, 20B, and 34B parameters. 

<!-- ## :sparkles: Model highlights

* `Granite-34b-code-base` model outperforms the recently released `Code-llama-70b` model on [HumanEval](https://paperswithcode.com/sota/code-generation-on-humaneval) benchmark (Pass1: 54.2% vs 53.0%) despite having less thant half of the model parameters.
* `Granite-8b-code-base` model considerably outperforms`code-llama-13b` by 8.2%. -->

## Models Download
You can download our models from 🤗 Hugging Face, please follow these steps:

* Visit one of the repos, for example ibm-granite/granite-3b-code-base. To download the models, click on *Files and versions* tab and download the content of the main branch. You can also download our models from the command line if you pip install huggingface-hub, or by cloning the hugging face repository.

## 🤗 Model Cards
The model cards for each model variant are available in their respective repository. Please visit our collection [here](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330).

## :point_up: Intended use
Prominent enterprise use cases of LLMs in software engineering productivity include code generation, code explanation, code fixing, generating unit tests, generating documentation, addressing technical debt issues, vulnerability detection, code translation, and more. Our Granite Code foundational models perform well on these tasks.

## :page_with_curl: Ethical considerations
### Data
All our models are trained on license-permissible code data collected by following IBM’s AI Ethics principles and guided by IBM’s Corporate Legal team. IBM has developed a robust data governance process that evaluates datasets for governance, risk, and compliance (GRC) criteria, including but not limited to IBM’s standard data clearance process, quality checks, malware scanning, masking of personally identifiable information (PII), as well as removal of hate, abuse, and profanity (HAP).

### Risks and harms
LLMs are often prone to generating incorrect information, typically referred to as hallucinations. Our models are not the exception in this regard. However, thanks to our careful selection and processing of training data, we hope that our models generate less harmful, offensive, and biased content than others.

## :sparkles: Training data highligths

* The pretraining code data was sourced from a combination of publicly available datasets (e.g., [Github Code Clean](https://huggingface.co/datasets/codeparrot/github-code-clean), [Starcoder data](https://huggingface.co/datasets/bigcode/starcoderdata)), and additional public code repositories and issues from GitHub. 
* We produced 251 million files of code datasets and 0.5 million files of GitHub issues that sum up 549 billion unique code tokens.
* Each model in the series has been trained from scratch on 4 trillion tokens sourced from 116 programming languages, ensuring a comprehensive understanding of programming languages and syntax.
* Besides training base code LLMs, we finetune on a filtered variant of [CommitPack](https://huggingface.co/datasets/bigcode/commitpackft) and [OASST](https://huggingface.co/datasets/bigcode/oasst-octopack) for improving instruction following capabilities. 
* We tokenize data via byte pair encoding  ([Li et al., 2023](https://arxiv.org/pdf/2305.06161)), with the same tokenizer used for StarCoder.

Please check our [paper]() for further details about data collection and processing.

## :pushpin: Models Architecture
We train the series of code models of varying sizes based on the transformer decoder architecture ([Vaswani et al](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)., 2017). The table below shows the model hyperparameters that we used per model size.

| Model                 | 3B       | 8B       | 20B       | 34B       |
| --------              | -------- | -------- | --------  | --------  |
| Batch size            | 2048     | 1024     | 576       | 532       |
| Context length        | 2048     | 4096     | 8192      | 8192      |
| Hidden size           | 2560     | 4096     | 4096      | 6144      |
| FFN hidden size       | 10240    | 14336    | 24576     | 24576     |
| Attention heads       | 32       | 32       | 48        | 48        |
| Key-Value heads       | 32(MHA)  | 8(GQA)   | 1(MQA)    | 1(MQA)    |
| Layers                | 32       | 36       | 52        | 88        |
| Normalization         | RMSNorm  | RMSNorm  | LayerNorm | LayerNorm |
| Activation            | swiglu   | swiglu   | gelu      | gelu      |
| Vocab size            | 49252    | 49252    | 49252     | 49252     |

Please check our [paper]() for further discussion on architectural decisions per model size (3B and 8B, 20B, 34B).

## Pretraining

We trained Granite Code models on 4.5T tokens of code data and natural language datasets related to code. To do so, we followed a two-phase training process:
* Phase 1 (code only training):in this phase, we trained 3B and 8B models on 4 trillion tokens of code data comprising 116 languages. We trained the 20B parameter model on 3 trillion tokens of code. Finally, we trained the 34B model on 1.4T tokens after depth upscaling, which we did on the 1.6T checkpoint of 20B model.
* Phase 2 (code + language training): in this phase, we included additional high-quality publicly available data from various domains (e.g., technical, mathematics, web documents and instruction following data) to improve further the models' performance. Specifically, we trained all models for 500B tokens on a 80% code-20% natural language mixture.

## :chart_with_upwards_trend: Evaluation
### Overview
We evaluate Granite Code models on a wide variety of tasks, including code generation, code explanation, code fixing, code editing, math reasoning, and more. The table below summarizes the evaluations that we performed.

~~TO DO: TABLE 2 FROM PAPER~~

We compare our series of models with recent state-of-the-art open code LLMs, such as: StableCode, Code Llama, StarCoder, StarCoder2, and CodeGemma, and recent high-performing general purpose LLMs like Mistral, Mixtral and LLama3.

<!-- OPTION A -->
<!-- TO DO: MOVE TO METADATA: Generation in Python, .. give all the numbers -->
### Code Generation in Python 
| Model                             | HumanEval   | HumanEval+  | MBPP           | MBPP+          |
| --------                          | --------    | --------    | --------       | --------       |
| StarCoderBase-3B                  | 22.6        | 17.7        | 29.4           | 37.8           |
| StableCode-3B                     | 28.6        | 25.6        | 34.8           | 43.6           |
| StarCoder2-3B                     | 32.3        | 28.0        | 42.4           | 48.6           |
| CodeGemma-2B (fp32/fp16/bf16)     | 7.9/7.9/7.3 | 6.7/6.7/5.5 | 30.4/29.6/22.2 | 30.8/28.8/19.5 |
| Granite-3B-Code-Base              | 34.1        | 29.9        | 36.0           | 45.1           |

###  Multilingual Code Generation (HumanEvalSynthesize)
| Model                             | Python   | Javascript  | Java     | Go       |
| --------                          | -------- | --------    | -------- | -------- |
| StarCoderBase-3B                  |          |             |          |          |
| StableCode-3B                     |          |             |          |          |
| StarCoder2-3B                     |          |             |          |          |
| CodeGemma-2B (fp32/fp16/bf16)     |          |             |          |          |
| Granite-3B-Code-Base              |          |             |          |          |


### Results
Our base Granite Code models beat CodeLlama models in three out of four sizes compared. The figure below shows our evaluation results. Please note that `Granite-34b-code-base` model outperforms `Code-llama-70b` on HumanEval by achieving a Pass1 score of 54.2%, despite being half the size of `Code-llama-70b`. Moreoever, our 8b model shows a considerable improvement over`code-Llama-13b`. It is also worth mentioning that even our smaller model, `Granite-3b-code-base`, proves to be competitive with respect to `Code-llama-7b`.

![image](https://hackmd.io/_uploads/SJBxrJvbA.png)

We share detailed evaluation results in our paper and each model HF card.

## :page_with_curl: Research Paper
* Would you like to read our paper? Please click [here](https://www.overleaf.com/project/6520094b0a31c2dc6445597e).
* Would you like to cite our paper? Please do it this way:
```
@misc{granite-models,
  author = {author 1, author2, ...},
  title = {Granite Code Large Language Models: IBM Foundation Models for Code},
  journal = {},
  volume = {},
  year = {2024},
  url = {https://arxiv.org/abs/0000.00000},
}
```

## :star: License 
All Granite Code Models are distributed under [Apache 2.0](./LICENSE) license.

## :raising_hand: Would you like to provide feedback?
Please let use know your comments about our family of code models by visiting our :hugging_face: [collection](https://huggingface.co/collections/ibm-granite/granite-code-models-6624c5cec322e4c148c8b330). Select the repository of the model you would like to provide feedback about. Then, go to *Community* tab, and click on *New discussion*.