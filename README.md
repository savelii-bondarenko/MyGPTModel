# MyGPTModel

MyGPTModel is a custom, lightweight PyTorch implementation of a GPT-style Transformer language model. It is designed to train on a simple text dataset and generate text using autoregressive sampling.

## Features

* **Custom Transformer Architecture**: Implements a standard decoder-only Transformer featuring Multi-Head Attention, FeedForward networks, GELU activations, and Layer Normalization.
* **Embeddings**: Utilizes PyTorch `nn.Embedding` for token embeddings alongside sinusoidal positional embeddings.
* **Tokenization**: Uses OpenAI's `tiktoken` library to encode text using the `gpt2` encoding standard.
* **Training Pipeline**: Includes a complete training loop with PyTorch's `DataLoader`, the `Adam` optimizer, and `CrossEntropyLoss`.
* **Text Generation**: Provides an inference script that supports text generation using temperature scaling and top-k sampling (k=3).

## Model Configuration

The project is currently set up with a configuration modeled after a 124M parameter architecture:

* **Vocabulary Size**: 50,257
* **Embedding Dimension**: 768
* **Attention Heads**: 12
* **Layers**: 12
* **Context Length**: 4
* **Dropout Rate**: 0.1

## Repository Structure

* `config/gpt_config.py`: Contains the core `GPT_CONFIG_124M` dictionary that defines the model's hyperparameters.
* `data/dataset.py`: Defines the text tokenization pipeline and a `TokenPair` PyTorch Dataset that yields input/target sequences chunked by the context length.
* `model/transformer.py`: Contains the neural network layer implementations including `Transformer`, `MultiHeadAttention`, `LayerNorm`, `FeedForward`, and `GELU`.
* `model/embedding.py`: Handles the construction of the token and positional embeddings.
* `train/train.py`: The main training script that processes the dataset, performs forward and backward passes, and saves model checkpoints at the end of each epoch.
* `train/utils.py`: Utility functions for saving and loading model states (`save_model`, `load_model`).
* `generate/generate.py`: The inference script that loads a saved checkpoint (defaulting to epoch 9) and generates novel text token-by-token.
* `the-verdict.txt`: The sample dataset used to train the model, which is a short story text.

## Getting Started

### Prerequisites

Make sure you have Python installed, then install the required dependencies from the requirements file:

```bash
pip install -r requirements.txt

```

The dependencies include packages like `torch`, `tiktoken`, and `tqdm`.

### Training the Model

To train the model from scratch on `the-verdict.txt`, run the training script. The script is configured to train for 100 epochs using an Adam optimizer with a learning rate of 1e-4:

```bash
python -m train.train

```

Model checkpoints will be automatically saved to a `checkpoints/` directory as `gpt_epoch_{epoch}.pt`.

### Generating Text

Once you have trained the model and have a checkpoint available, you can generate text. Ensure the checkpoint path in `generate/generate.py` matches your saved weights (it looks for `checkpoints/gpt_epoch_9.pt` by default):

```bash
python -m generate.generate

```

By default, the script generates 32 new tokens starting from the input prompt `"Hello"` with a temperature of 0.8.