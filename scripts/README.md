# Jupyter Notebook to Markdown Converter

This tool converts Jupyter notebooks to Markdown format optimized for Docusaurus documentation.

## Overview

The `generate_markdown.py` script processes Jupyter notebooks defined in the `/index.toml` configuration file, converting them to Markdown files with appropriate frontmatter, formatting, and enhancements for display in a Docusaurus documentation site.

## Usage

```bash
python generate_markdown.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
```

### Arguments

- `--config`: Path to the TOML configuration file (default: `/index.toml`)
- `--output`: Base directory for markdown output (default: `/markdowns`)

### Configuration File Structure

The items in the TOML configuration file should have the following structure:

```toml
[[recipe]]
title = "My Notebook Title"
notebook = "path/to/notebook.ipynb"
featured = false
integration = false
agent = false
tags = ["tag1", "tag2"]
```

## What the script does

1. Reads the TOML configuration file that specifies notebooks to convert
2. For each notebook:
   - Generates frontmatter with metadata (title, tags, etc.)
   - Adds a Colab badge for easy opening in Google Colab
   - Converts the notebook content to Markdown
   - Applies various transformations to make it Docusaurus-compatible
   - Outputs the transformed Markdown to the specified directory

### Transformation process

The `notebook_converter.py` script performs the following steps in the conversion process:

1. **Generate Frontmatter**: Creates YAML frontmatter with metadata from the configuration
2. **Add Colab Badge**: Inserts an HTML badge after frontmatter for easy opening in Google Colab
3. **Load & Clean Notebook**: Processes the notebook and cleans Colab-specific dataframe outputs
4. **Convert to Markdown**: Uses nbconvert to transform the notebook to Markdown
5. **Format Indented Output**: Properly formats code output blocks with ```text markers
   - Preserves Python code within output blocks
   - Escapes backticks within output blocks
   - Strips ANSI color codes from outputs
6. **Clean HTML Tags**: Removes span tags from pre blocks and transforms style attributes to React format
7. **Apply Docusaurus Fixes**: Converts inline styles to React style format
8. **Escape Special Characters**: Handles special characters for MDX compatibility
9. **Fix Image Paths**: Updates image paths to reference the GitHub repository
10. **Remove WhatNext Component**: Removes WhatNext import and component references
11. **Remove First H1 Heading**: Removes the first H1 heading if it exists
12. **Write Output**: Saves the processed Markdown to the output directory

## Output Structure

The output will be organized into subdirectories based on notebook type:

- `agents/` - For agent-related notebooks
- `integrations/` - For integration-related notebooks
- `weaviate/` - For general Weaviate notebooks

## Example

```bash
python generate_markdown.py --config my_index.toml --output docs/recipes
```

This will process all notebooks defined in `my_index.toml` and output the resulting Markdown files to the `docs/recipes` directory.
