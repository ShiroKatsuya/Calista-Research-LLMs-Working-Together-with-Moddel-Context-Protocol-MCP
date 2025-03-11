![Minions Logo](assets/Ollama_minionS_background.png)

# Where On-Device and Cloud LLMs Meet

[![Discord](https://img.shields.io/badge/Discord-7289DA?logo=discord&logoColor=white)](https://discord.gg/jfJyxXwFVa)

_What is this?_ Minions is a communication protocol that enables small on-device models to collaborate with frontier models in the cloud. By only reading long contexts locally, we can reduce cloud costs with minimal or no quality degradation. This repository provides a demonstration of the protocol. Get started below or see our paper and blogpost below for more information.

Paper: [Minions: Cost-efficient Collaboration Between On-device and Cloud
Language Models](https://arxiv.org/pdf/2502.15964)

Blogpost: https://hazyresearch.stanford.edu/blog/2025-02-24-minions

## Setup

_We have tested the following setup on Mac and Ubuntu with Python 3.10-3.11_ (Note: Python 3.13 is not supported)

<details>
  <summary>Optional: Create a virtual environment with your favorite package manager (e.g. conda, venv, uv)</summary>
        
  ```python
  conda create -n minions python=3.11
  ```
  
</details>

**Step 1:** Clone the repository and install the Python package.

```bash
git clone https://github.com/HazyResearch/minions.git
cd minions
pip install -e .  # installs the minions package in editable mode
```

**Step 2:** Install a server for running the local model.

We support two servers for running local models: `ollama` and `tokasaurus`. You need to install at least one of these.

- You should use `ollama` if you do not have access to NVIDIA GPUs. Install `ollama` following the instructions [here](https://ollama.com/download). To enable Flash Attention, run
  `launchctl setenv OLLAMA_FLASH_ATTENTION 1` and, if on a mac, restart the ollama app.
- You should use `tokasaurus` if you have access to NVIDIA GPUs and you are running the Minions protocol, which benefits from the high-throughput of `tokasaurus`. Install `tokasaurus` with the following command:

```
uv pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tokasaurus==0.0.1.post1
```

**Step 3:** Set your API key for at least one of the following cloud LLM providers.

_If needed, create an [OpenAI API Key](https://platform.openai.com/docs/overview) or [TogetherAI API key](https://docs.together.ai/docs/quickstart) for the cloud model._

```bash
export OPENAI_API_KEY=<your-openai-api-key>
export TOGETHER_API_KEY=<your-together-api-key>
```

## Minions Demo Application

[![Watch the video](https://img.youtube.com/vi/70Kot0_DFNs/0.jpg)](https://www.youtube.com/watch?v=70Kot0_DFNs)

To try the Minion or Minions protocol, run the following command:

```bash
streamlit run app.py
```

If you are seeing an error about the `ollama` client,

```
An error occurred: Failed to connect to Ollama. Please check that Ollama is downloaded, running and accessible. https://ollama.com/download
```

try running the following command:

```bash
OLLAMA_FLASH_ATTENTION=1 ollama serve
```

## Example code: Minion (singular)

The following example is for an `ollama` local client and an `openai` remote client.
The protocol is `minion`.

```python
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minion import Minion

local_client = OllamaClient(
        model_name="llama3.2",
    )

remote_client = OpenAIClient(
        model_name="gpt-4o",
    )

# Instantiate the Minion object with both clients
minion = Minion(local_client, remote_client)


context = """
Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.
Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable.
"""

task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

# Execute the minion protocol for up to two communication rounds
output = minion(
    task=task,
    context=[context],
    max_rounds=2
)
```

## Streaming Output

To enable real-time streaming of response output as it's being generated, you can use the `stream_output` parameter:

```python
from minions.clients.openai import OpenAIClient
from minions.minion import Minion

# Enable streaming in the client
openai_client = OpenAIClient(
    model_name="gpt-4o",
    stream=True  # Enable streaming in the client
)

# Create the minion with streaming enabled
minion = Minion(
    local_client=openai_client,
    remote_client=openai_client,
    stream_output=True  # Enable streaming in the Minion
)

# Now responses will print incrementally in real-time
result = minion(
    task="Generate a short poem about artificial intelligence",
    context=["Make it thoughtful but accessible to general audiences"]
)
```

You can also provide a custom callback function for more control over how streaming content is displayed:

```python
def custom_callback(agent_type, chunk, is_streaming=False, is_final=False):
    """Custom callback to handle streaming output."""
    if is_streaming:
        # Process streaming chunks in real-time
        # For example, you could format or colorize the output
        pass
    elif chunk:
        # Handle complete responses
        print(f"\n[{agent_type}] COMPLETE RESPONSE: {chunk}\n")

# Create the minion with custom callback
minion = Minion(
    local_client=openai_client,
    remote_client=openai_client,
    callback=custom_callback,
    stream_output=True
)
```

See the `examples/stream_example.py` file for a complete example.

## Example Code: Minions (plural)

The following example is for an `ollama` local client and an `openai` remote client.
The protocol is `minions`.

```python
from minions.clients.ollama import OllamaClient
from minions.clients.openai import OpenAIClient
from minions.minions import Minions
from pydantic import BaseModel

class StructuredLocalOutput(BaseModel):
    explanation: str
    citation: str | None
    answer: str | None

local_client = OllamaClient(
        model_name="llama3.2",
        temperature=0.0,
        structured_output_schema=StructuredLocalOutput
)

remote_client = OpenAIClient(
        model_name="gpt-4o",
)


# Instantiate the Minion object with both clients
minion = Minions(local_client, remote_client)


context = """
Patient John Doe is a 60-year-old male with a history of hypertension. In his latest checkup, his blood pressure was recorded at 160/100 mmHg, and he reported occasional chest discomfort during physical activity.
Recent laboratory results show that his LDL cholesterol level is elevated at 170 mg/dL, while his HDL remains within the normal range at 45 mg/dL. Other metabolic indicators, including fasting glucose and renal function, are unremarkable.
"""

task = "Based on the patient's blood pressure and LDL cholesterol readings in the context, evaluate whether these factors together suggest an increased risk for cardiovascular complications."

# Execute the minion protocol for up to two communication rounds
output = minion(
    task=task,
    doc_metadata="Medical Report",
    context=[context],
    max_rounds=2
)
```

## Python Notebook

To run Minion/Minions in a notebook, checkout `minions.ipynb`.

## CLI

To run Minion/Minions in a CLI, checkout `minions_cli.py`.

```bash
minions --help
```

```bash
minions --context <path_to_context> --protocol <minion|minions>
```

## Maintainers

- Avanika Narayan (contact: avanika@cs.stanford.edu)
- Dan Biderman (contact: biderman@stanford.edu)
- Sabri Eyuboglu (contact: eyuboglu@cs.stanford.edu)
