# Local LLM Provider

This is use for [for VSCode Continue Plugin](https://docs.continue.dev), to provider a local LLM to provide code completion and chat, agent(edit,apply) ability.

## Usage

Download the openai standard model from [Hugging Face](https://huggingface.co) and put it into the `model` directory.

Change the `model_name` in `model_provider.py` to the downloaded model name.

Run the `main.py` file to start the server.

Install Vscode Continue Plugin and connect to the server.