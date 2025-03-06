# Setup Local DSPy Development Environment

Here are the steps needed to set up and manage a local build of DSPy project with Poetry, build a local version of the library, create a Jupyter notebook kernel, and update the environment when the local DSPy package changes.

# Setting Up and Managing a Python Project with Poetry and Jupyter

This guide walks you through setting up a Python project with Poetry, building a local version of a library, creating a Jupyter notebook kernel using this environment, and updating the environment when the local package changes.

## Prerequisites

- Ensure you have Python installed (version 3.9 or later).
- Install Poetry by running:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

# Step 1: Initialize a Poetry Project
Navigate to your project directory:

```bash
cd path/to/dspy-dev
```

Initialize a new Poetry project:

```bash
poetry init
```

Follow the prompts to set up your project, including naming it, specifying the version, description, etc.

Edit the `pyproject.toml` file to include your local package:

```toml
[tool.poetry]
name = "dspy-dev"
version = "0.1.0"
description = ""
authors = ["Your Name <you@example.com>"]
readme = "README.md"

[tool.poetry.dependencies]
python = ">=3.9,<3.13"
dspy = { path = "./dspy" }

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"
```

# Step 2: Install Dependencies
Install the dependencies and create a virtual environment:

```bash
poetry install
```

# Step 3: Create a Jupyter Kernel

```bash
poetry add jupyter ipykernel
```

Create a new Jupyter kernel:

```bash
poetry run python -m ipykernel install --user --name dspy-dev --display-name "Python (dspy-dev)"
```

# Step 4: Launch Jupyter Notebook

```bash
poetry run jupyter notebook
```

# Step 5: Select the New Kernel

Open a new or existing notebook.
Go to the kernel selection menu (Kernel -> Change kernel).
Select "Python (dspy-env)".

# Step 6: Update the Local Package

When you make changes to the local package:

Update the code in the `./dspy` directory as needed.

Reinstall the package to ensure Poetry recognizes the changes:

```bash
poetry install
```

In your Jupyter Notebook, click on Kernel -> Restart Kernel.

By following these steps, you can efficiently manage your local DSPy build with Poetry, create a Jupyter Notebook kernel using the environment, and update the environment when the local package changes.
