# MyGPTModel

MyGPTModel is a custom, lightweight PyTorch implementation of a GPT-style Transformer language model. It is designed to train on a simple text dataset and generate text using autoregressive sampling.

## Features

- **Custom Transformer Architecture**: Implements a decoder-only Transformer with Multi-Head Attention, FeedForward networks, GELU activations, and Layer Normalization.
- **Embeddings**: Utilizes PyTorch `nn.Embedding` for token embeddings alongside sinusoidal positional encodings.
- **Tokenization**: Uses OpenAI's `tiktoken` library to encode text using the `gpt2` encoding.
- **Training Pipeline**: Includes a training loop with PyTorch's `DataLoader`, the `Adam` optimizer, and `CrossEntropyLoss`.
- **Text Generation**: Provides an inference script that supports autoregressive text generation using temperature scaling and top-k sampling (k=3).
- **Dependency Management**: Uses `uv` for managing dependencies and running the project.

## Model Configuration

The project uses a configuration inspired by the GPT-2 124M architecture:

- **Vocabulary Size**: 50,257
- **Embedding Dimension**: 768
- **Attention Heads**: 12
- **Configured Layers**: 12 (currently one Transformer block is implemented)
- **Context Length**: 4
- **Dropout Rate**: 0.1

## Repository Structure

- `src/mygptmodel/config`: Contains the core `GPT_CONFIG_124M` dictionary that defines the model's hyperparameters.
- `src/mygptmodel/data`: Defines the text tokenization pipeline and a `TokenPair` PyTorch Dataset that yields input/target sequences chunked by the context length.
- `src/mygptmodel/model`: Contains the neural network implementations, including `Transformer`, `MultiHeadAttention`, `LayerNorm`, `FeedForward`, `GELU`, and the token and positional embeddings.
- `src/mygptmodel/train`: Contains the training script that processes the dataset, performs forward and backward passes, and saves model checkpoints after each epoch.
- `src/mygptmodel/train`: Includes utility functions for saving and loading model states (`save_model`, `load_model`).
- `src/mygptmodel/generate`: Contains the inference script that loads a saved checkpoint (defaulting to epoch 9) and generates text token-by-token.
- `data/the-verdict.txt`: The sample text dataset used to train the model.
- `pyproject.toml`: Defines project metadata and dependencies managed by `uv`.

## Getting Started

### Prerequisites

Make sure you have Python and [uv](https://docs.astral.sh/uv/) installed.

Install and synchronize the project dependencies:

```bash
uv sync
```

This creates or updates the project's virtual environment and installs the required dependencies, including `torch`, `tiktoken`, and `tqdm`.

### Training the Model

To train the model from scratch on `data/the-verdict.txt`, run:

```bash
uv run python -m mygptmodel.train.train
```

The script is configured to train for 100 epochs using the Adam optimizer with a learning rate of `1e-4`.

Model checkpoints are saved after each epoch to the `checkpoints/` directory as `gpt_epoch_{epoch}.pt`.

### Generating Text

Once the model has been trained and a checkpoint is available, generate text using:

```bash
uv run python -m mygptmodel.generate.generate
```

Ensure the checkpoint path in the generation script matches your saved weights. By default, the script loads `checkpoints/gpt_epoch_9.pt`.

The script prompts the user for an initial text input and generates 32 new tokens using a temperature of `0.8` and top-k sampling with `k=3`.