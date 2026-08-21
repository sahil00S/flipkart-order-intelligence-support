# Flipkart Order Intelligence & Support Assistant

An end-to-end AI/ML capstone that combines classical machine learning, transfer-learning image classification, retrieval-augmented generation, and a LangGraph support assistant into one connected system.

## Project Overview

The system is designed as a single connected support workflow with three capabilities:

1. Predict whether an order is likely to be returned.
2. Classify a product image into one of the Fashion-MNIST catalog categories.
3. Answer policy questions from a grounded knowledge base through a LangGraph support assistant.

Part 3 connects the saved artifacts from Parts 1 and 2 as real callable tools.

## Architecture

```text
Customer Request
      |
      v
+----------------------+
|   LangGraph Agent    |
+----------------------+
      |
      v
   Intent Node
      |
      +--------------------+----------------------+
      |                    |                      |
      v                    v                      v
   Policy              Return Risk          Image Category
      |                    |                      |
      v                    v                      v
  FAISS RAG          RF Model Tool         ResNet-18 Tool
      |                    |                      |
      +--------------------+----------------------+
                           |
                           v
                  Response Generation
                           |
                           v
                 Structured JSON Output
```

The default agent mode is deterministic `MOCK_LLM` mode and requires no API key.

## Repository Structure

```text
flipkart-order-intelligence-support/
│
├── data/
│   └── sample_images/
│       ├── test_0000_true_9_Ankle_boot.png
│       ├── test_0001_true_2_Pullover.png
│       ├── test_0002_true_1_Trouser.png
│       ├── test_0003_true_1_Trouser.png
│       └── test_0004_true_6_Shirt.png
│
├── knowledge_base/
│   ├── cod_policy.md
│   ├── cod_refund.md
│   ├── customer_support.md
│   ├── damaged_product.md
│   ├── delivery_sla.md
│   ├── missing_product.md
│   ├── payment_policy.md
│   ├── prepaid_policy.md
│   ├── replacement_policy.md
│   ├── return_windows.md
│   ├── reverse_pickup.md
│   └── wrong_product.md
│
├── models/
│   ├── product_classifier.pt
│   ├── return_risk_model.pkl
│   └── return_risk_threshold.json
│
├── part1/
│   └── missing_data_analysis.py
│
├── part2/
│   ├── confusion_matrix.npy
│   ├── confusion_matrix.png
│   ├── evaluate_classifier.py
│   ├── per_class_metrics.csv
│   └── train_classifier.py
│
├── part3/
│   ├── build_vector_index.py
│   ├── generate_transcripts.py
│   ├── langgraph_assistant.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   ├── retrieval_metrics.json
│   ├── test_state_and_guardrails.py
│   ├── tools/
│   │   ├── image_classifier_tool.py
│   │   └── order_risk_tool.py
│   └── vector_index/
│       ├── chunks.json
│       ├── documents.json
│       └── policy.index
│
├── transcripts/
│   ├── 01_policy_query.json
│   ├── 02_second_policy_query.json
│   ├── 03_return_risk.json
│   ├── 04_image_classification.json
│   ├── 05_multi_turn_state.json
│   ├── 06_fresh_conversation.json
│   ├── 07_prompt_injection.json
│   └── 08_unsupported_policy.json
│
├── generate_orders.py
├── orders_dataset.csv
├── requirements.txt
└── README.md
```

## Environment Setup

The project was implemented using Python 3.14.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Main libraries:

* NumPy
* Pandas
* scikit-learn
* Matplotlib
* PyTorch
* torchvision
* Pillow
* sentence-transformers
* FAISS CPU
* LangGraph

## Part 1 — Return-Risk Prediction

### Dataset

The dataset is generated deterministically by:

```text
generate_orders.py
```

The generator uses a fixed NumPy random seed and produces exactly 6,000 synthetic orders.

To regenerate the dataset:

```bash
python generate_orders.py
```

This creates:

```text
orders_dataset.csv
```

### Missing-Data Analysis

The `rating_given` column contains missing values whose missingness depends on the observed `payment_method` column.

Measured missingness:

```text
Overall missing rating: 13.05%
COD missing rating:     22.83%
Non-COD missing rating:  6.06%
```

The missingness is therefore classified as **MAR (Missing At Random)** conditional on the observed payment method.

### Preprocessing

The Part 1 pipeline uses:

* stratified 80/20 train/test split
* median imputation for numeric features
* most-frequent imputation for categorical features
* one-hot encoding
* standard scaling
* leakage-safe preprocessing through `Pipeline` and `ColumnTransformer`

### Dummy Baseline

The most-frequent DummyClassifier produced:

```text
Accuracy:  0.7725
Precision: 0.0000
Recall:    0.0000
F1:        0.0000
```

The high accuracy is misleading because the model predicts the majority class and detects none of the returned orders.

### Logistic Regression

Configuration:

```text
class_weight="balanced"
```

Default threshold results:

```text
Accuracy:  0.5917
Precision: 0.2964
Recall:    0.5788
F1:        0.3921
ROC-AUC:   0.6253
```

Threshold sweep:

```text
Range: 0.10 to 0.90
Step: 0.01
```

F1-maximising threshold:

```text
Threshold: 0.44
Precision: 0.2801
Recall:    0.7582
F1:        0.4091
```

Lowering the threshold increases recall, allowing the system to catch more potentially returned orders while accepting more false positive return-risk flags and therefore lower precision.

### Random Forest

The Random Forest was tuned using `GridSearchCV` with:

```text
n_estimators: [100, 200]
max_depth: [6, 10, None]
class_weight: balanced
random_state: 42
5-fold StratifiedKFold
scoring: roc_auc
```

Winning configuration:

```text
n_estimators = 200
max_depth    = 6
```

Measured performance:

```text
Best CV ROC-AUC: 0.6192
Test ROC-AUC:    0.6203
```

The small CV/test difference provides evidence against severe overfitting.

### Feature Importance

Top impurity-based features included:

```text
payment_method_COD
price_inr
delivery_distance_km
customer_tenure_days
delivery_days
```

Permutation importance on the held-out test split showed:

```text
payment_method
price_inr
num_previous_returns
product_category
delivery_days
```

The comparison demonstrates why impurity importance and held-out permutation importance can produce different rankings. In particular, a continuous feature can receive high impurity importance because tree-splitting criteria can favor many possible split points even when the feature carries comparatively little independent predictive signal.

### Subgroup Analysis

The weakest payment-method subgroup was:

```text
Prepaid_Card
Precision: 0.2000
Recall:    0.0204
```

A concrete next step is to introduce a payment-method-specific decision threshold for weaker prepaid subgroups and evaluate the resulting recall/precision trade-off.

### Final Model

The final saved artifact is:

```text
models/return_risk_model.pkl
```

It is a fitted scikit-learn `Pipeline` containing the leakage-safe preprocessing and the tuned Random Forest.

The Random Forest probability threshold used by Part 3 is stored in:

```text
models/return_risk_threshold.json
```

Measured:

```text
t*_rf = 0.50
```

For the threshold-anchored risk bucket implementation:

```text
Low    : probability < 0.50
Medium : 0.50 <= probability < 0.65
High   : probability >= 0.65
```

## Part 2 — Fashion-MNIST Product Image Classifier

### Dataset

The project uses the required Fashion-MNIST dataset:

```text
Training:   60,000
Validation:  5,000
Training subset: 55,000
Test:       10,000
```

The validation split is stratified and the test split remains untouched until final evaluation.

### Transfer Learning

The classifier uses:

* pretrained ResNet-18
* grayscale Fashion-MNIST images replicated to 3 channels
* resize to 224×224
* ImageNet normalization
* frozen pretrained backbone during feature extraction
* a new 10-class classifier head
* Adam optimization

Feature-extraction validation accuracy reached:

```text
89.64%
```

Because validation accuracy exceeded 80%, later-layer fine-tuning was not required.

### Final Test Result

```text
Test accuracy: 88.49%
```

### Confusion Matrix

The generated 10×10 confusion matrix is stored in:

```text
part2/confusion_matrix.png
part2/confusion_matrix.npy
```

The strongest real confusion pairs were:

```text
Shirt -> T-shirt_top : 120
Coat  -> Shirt       : 105
```

The confusion between these classes is plausible because clothing silhouettes can share similar upper-body shapes, sleeves, collars, and torso outlines in low-resolution grayscale images.

### Per-Class Metrics

Per-class precision and recall are stored in:

```text
part2/per_class_metrics.csv
```

The weakest class by F1 in the measured test results was:

```text
Shirt
Precision: 0.6693
Recall:    0.6880
F1:        0.6785
```

### Saved Model

```text
models/product_classifier.pt
```

The Part 3 image-classification tool loads this saved ResNet-18 state dictionary and performs prediction against committed PNG files.

## Part 3 — RAG Support Assistant

### Knowledge Base

The policy knowledge base contains 12 short policy documents covering:

* return windows by category
* COD refunds
* delivery SLAs
* reverse-pickup eligibility
* replacement policy
* payment policy
* prepaid payments
* damaged products
* wrong products
* missing products
* customer support guidance

The documents are sentence-chunked and each chunk retains its parent document ID and source file.

### Embeddings and Vector Index

Embeddings are generated locally with:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The resulting vectors are indexed with FAISS.

Artifacts:

```text
part3/vector_index/policy.index
part3/vector_index/chunks.json
part3/vector_index/documents.json
```

Measured index:

```text
Documents: 12
Sentence chunks: 37
Embedding dimension: 384
FAISS vectors: 37
```

### Retrieval Evaluation

Six evaluation queries were used with document-level relevance labels.

Measured averages:

```text
Mean Precision@3: 0.7222
Mean Recall@3:    1.0000
```

Full per-query results are stored in:

```text
part3/retrieval_metrics.json
```

### Real Tools

#### Return-risk tool

```python
check_return_risk(order_features: dict) -> dict
```

The tool:

1. loads `models/return_risk_model.pkl`
2. calls the real `predict_proba()`
3. loads `t*_rf`
4. returns the predicted probability
5. assigns a threshold-anchored risk bucket

#### Image classification tool

```python
classify_product_image(image_path: str) -> dict
```

The tool:

1. loads `models/product_classifier.pt`
2. reads a real committed PNG
3. performs the same preprocessing used by the classifier
4. returns category and confidence

### LangGraph

The agent contains four required nodes:

```text
intent
rag_retrieval
tool_calling
response_generation
```

A conditional edge routes policy questions to RAG and return-risk/image questions to the tool-calling path.

### MOCK_LLM

The default response generator is deterministic and rule/template based.

It requires:

```text
No API key
No paid service
No live LLM
```

The implementation includes:

* role prompting
* Specific
* Short
* Surround
* Single
* two few-shot intent examples
* fixed JSON response schema

Response schema:

```json
{
  "answer": "string",
  "source": "policy_kb | return_risk_tool | image_classifier_tool",
  "confidence": 0.0
}
```

### Conversation State

The agent demonstrates both:

```text
Multi-turn conversation:
state/messages carried across turns
```

and:

```text
Fresh conversation:
state starts empty
```

The state demonstrations are saved in:

```text
transcripts/05_multi_turn_state.json
transcripts/06_fresh_conversation.json
```

### Guardrails

Input-side prompt-injection detection blocks patterns such as:

```text
ignore previous instructions
ignore all rules
reveal the system prompt
pretend you are...
```

Output-side groundedness checking refuses unsupported policy answers when the top retrieved similarity is below the configured threshold.

Example refusal:

```text
similarity = 0.4449
threshold  = 0.4500
```

The refusal is saved in:

```text
transcripts/08_unsupported_policy.json
```

### Test Transcripts

The repository contains eight required transcripts:

```text
01_policy_query.json
02_second_policy_query.json
03_return_risk.json
04_image_classification.json
05_multi_turn_state.json
06_fresh_conversation.json
07_prompt_injection.json
08_unsupported_policy.json
```

## Running the Project

### Part 1

Regenerate the dataset:

```bash
python generate_orders.py
```

Run Part 1 analysis/model training:

```bash
python part1/missing_data_analysis.py
```

### Part 2

Train the classifier:

```bash
python part2/train_classifier.py
```

Run final evaluation:

```bash
python part2/evaluate_classifier.py
```

### Part 3

Build the FAISS vector index:

```bash
python part3/build_vector_index.py
```

Evaluate retrieval:

```bash
python part3/rag_pipeline.py
```

Test the return-risk tool:

```bash
python part3/tools/order_risk_tool.py
```

Test the image-classification tool:

```bash
python part3/tools/image_classifier_tool.py
```

Run the LangGraph assistant in deterministic MOCK_LLM mode:

```bash
python -m part3.langgraph_assistant
```

Run state and guardrail tests:

```bash
python -m part3.test_state_and_guardrails
```

Regenerate the required transcripts:

```bash
python -m part3.generate_transcripts
```

## Example Agent Transcript

### Policy

**User**

```text
How long does a COD refund take?
```

**Assistant**

```json
{
  "answer": "COD refunds are processed within 5 to 7 business days after refund initiation.",
  "source": "policy_kb",
  "confidence": 0.9078
}
```

### Return Risk

**User**

```text
What is the return risk for this order?
```

**Assistant**

```json
{
  "answer": "Return probability is 0.6276. Risk bucket: Medium. Decision threshold: 0.5000. High-risk cutoff: 0.6500.",
  "source": "return_risk_tool",
  "confidence": 0.6276
}
```

### Image Classification

**User**

```text
Classify the product image.
```

**Assistant**

```json
{
  "answer": "The product image is classified as Ankle_boot with confidence 0.9462.",
  "source": "image_classifier_tool",
  "confidence": 0.9462
}
```

### Unsupported Policy

**User**

```text
What is the maximum compensation for a delayed order?
```

**Assistant**

```json
{
  "answer": "I cannot confirm that policy from the available knowledge base.",
  "source": "policy_kb",
  "confidence": 0.4449
}
```

The groundedness check prints the comparison:

```text
similarity=0.4449
threshold=0.4500
```

## Git Workflow

The repository demonstrates the required feature-branch workflow.

Completed branches include:

```text
part1-part2-complete
part3-complete
```

The final `main` branch contains visible merge commits for the completed work.

## Limitations

The current support assistant uses deterministic MOCK_LLM behavior rather than a live language model.

The order-risk demo tool uses a deterministic example order when invoked through the sample agent transcript.

The policy knowledge base is a project-controlled demonstration knowledge base and should not be interpreted as an official live Flipkart policy source.

## Future Improvements

Possible future improvements include:

* connecting the risk tool to real order IDs and dynamically supplied order features
* adding more policy documents and more retrieval evaluation queries
* adding a calibrated confidence strategy
* supporting additional image classes and higher-resolution product images
* adding more sophisticated conversational state
* optionally integrating a live LLM behind a separate configuration flag

## Final Status

```text
Part 1: Complete
Part 2: Complete
Part 3: Complete
GitHub repository: Public
MOCK_LLM mode: Supported
Saved model artifacts: Present
Required transcripts: Present
```
