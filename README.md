# Evaluating structured extraction with LangSmith

This repository is an example of the necessary scripts and modules for evaluation with [LangSmith](https://smith.langchain.com/).

It allows you to evaluate different frontier and OSS models on structured extraction over real-world emails.

![](/public/images/public_dataset.png)

The public dataset and evaluations run against it are available [in LangSmith here](https://smith.langchain.com/public/36bdfe7d-3cd1-4b36-b957-d12d95810a2b/d).

It includes:

1. A dataset of 42 real emails I pulled and deduped from my spam folder, with semantic HTML tags removed, as well as a script for initial extraction and formatting of other emails from an arbitrary `.mbox` file like the one exported by Gmail.
    - Some additional cleanup of the data was done by hand after the initial pass.
2. A script for bootstrapping a dataset by calling OpenAI's GPT-4, then logging those runs to a [LangSmith](https://smith.langchain.com/) project.
3. A script for running evaluations against the created dataset by swapping in different OpenAI, Anthropic, and various OSS models run through [Ollama](https://ollama.ai).

## Results

As expected, GPT-4 led the pack comfortably, but there were some surprises. Claude 2 with an XML-based prompt did better than GPT-3.5-turbo,
and though I was limited by hardware to small 7B parameter OSS models, [mistral-openorca](https://ollama.ai/library/mistral-openorca) with Ollama's JSON mode
performed almost on par with GPT-3.5-turbo despite not having a native functions API!

You can see the [results here](https://smith.langchain.com/public/36bdfe7d-3cd1-4b36-b957-d12d95810a2b/d).

More powerful and specialized OSS models would likely exceed some of these results.

Note that invalid/unparseable outputs are currently omitted in the final LangSmith tally rather than e.g. scored as 0, 
but with the exception of base Llama 2, which had many output errors, there were only a few cases for the non-functions models.
So on average displayed results should be close to the true values.

## Setup

Run `poetry install` to install the required dependencies.

## Creating the LangSmith dataset

Set up a LangSmith project and the following environment variables:

```
LANGCHAIN_API_KEY=
LANGCHAIN_SESSION=
LANGCHAIN_TRACING_V2="true"

OPENAI_API_KEY=""
```

Then, run `poetry run python bootstrap_dataset.py` to run GPT-4 against the raw emails and log the results to your LangSmith project.

The extraction chain will attempt to gather information according to the following Pydantic schema from the original emails:

```python
class ToneEnum(str, Enum):
    positive = "positive"
    negative = "negative"


class Email(BaseModel):
    """Relevant information about an email."""

    sender: Optional[str] = Field(None, description="The sender's name, if available")
    sender_phone_number: Optional[str] = Field(None, description="The sender's phone number, if available")
    sender_address: Optional[str] = Field(None, description="The sender's address, if available")
    action_items: List[str] = Field(..., description="A list of action items requested by the email")
    topic: str = Field(..., description="High level description of what the email is about")
    tone: ToneEnum = Field(..., description="The tone of the email.")
```

Once the script has finished, go into your LangSmith project. From there, you can add all the runs directly to a dataset, but to guarantee accuracy (GPT-4 isn't perfect!),
you can first add them to an annotation queue:

![](/public/images/logged_runs.png)

From there, you can examine the runs and make edits if they look incorrect, then add the validated run to a final curated dataset.

![](/public/images/annotation_queue.png)

## Evaluating other models

Run other models against an existing LangSmith key-value dataset with `poetry run python run_evals.py`.

The script currently requires you to manually change the model in the code. In addition to OpenAI's native function calling, you can easily swap in
the experimental LangChain [AnthropicFunctions](https://api.python.langchain.com/en/latest/llms/langchain_experimental.llms.anthropic_functions.AnthropicFunctions.html)
wrapper around Anthropic's Claude family of models and 
[OllamaFunctions](https://api.python.langchain.com/en/latest/llms/langchain_experimental.llms.ollama_functions.OllamaFunctions.html) for open source models 
running locally. These wrappers are designed to present the same functions interface.

If you want to evaluate Anthropic, you will need to retrieve your [Anthropic API key](https://console.anthropic.com/).

If you want to evaluate OSS models through Ollama, you will need to set up and install the [Ollama desktop app](https://ollama.ai/) and then pull a model.
Note that you will need version 0.1.9 or later for JSON mode support

You can also swap in other models or prompts if you'd like to test those.

The script will use `run_on_datset` to automatically run a chain against the entire dataset input, then run a custom evaluator powered by GPT-4 
that compares the output against the curated dataset output and grades it out of 10. You can see the actual criteria in the script.

Finally, go to your LangSmith dataset to see the evaluation results!

## Thank you!

I hope you found this informative! For more, follow me on X (formerly Twitter) [@Hacubu](https://x.com/hacubu/) and LangChain [@LangChainAI](https://x.com/langchainai/).
