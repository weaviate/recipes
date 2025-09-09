import argparse
from pathlib import Path
import traceback

# Use tomllib for Python 3.11+, otherwise fallback to tomli
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore
    except ModuleNotFoundError:
        print(
            "Error: 'tomli' package not found. Please install it (`pip install tomli`) for Python < 3.11."
        )
        exit(1)

# Import the conversion function from our new module
from notebook_converter import convert_notebook_to_markdown


def load_config(config_path):
    """Loads and parses the TOML configuration file."""
    print(f"Looking for index file at: {config_path}")
    if not config_path.exists():
        print(f"Error: Configuration file not found at {config_path}")
        return None

    try:
        index_toml_content = config_path.read_text(encoding="utf-8")
        index_data = tomllib.loads(index_toml_content)
        print("Successfully loaded configuration.")

        if "recipe" not in index_data:
            print("Warning: No recipes found in 'recipe' section of configuration.")
            # Allow continuing if recipes might be empty
            index_data["recipe"] = []
        return index_data
    except Exception as e:
        print(f"Error parsing configuration file {config_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert Jupyter Notebooks to Docusaurus Markdown"
    )
    parser.add_argument(
        "--config",
        default="index.toml",
        help="Path to the TOML configuration file relative to .docs directory (default: index.toml)",
    )
    parser.add_argument(
        "--output",
        dest="output",
        default="markdown",
        help="Base directory for markdown output relative to .docs directory (default: markdown)",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent  # .docs/scripts/
    docs_dir = script_dir.parent  # .docs/
    root_path = docs_dir.parent  # project root

    # Handle config path
    config_path_arg = Path(args.config)
    if config_path_arg.is_absolute():
        config_file_path = config_path_arg
        print(f"Using absolute config path: {config_file_path}")
    else:
        # Config path is relative to .docs directory
        config_file_path = (docs_dir / config_path_arg).resolve()

    print(f"Project root path: {root_path}")
    print(f"Docs directory path: {docs_dir}")

    # Load configuration
    index_data = load_config(config_file_path)
    if index_data is None:
        exit(1)

    # Prepare output directory - relative to .docs directory
    output_path_arg = Path(args.output)
    if output_path_arg.is_absolute():
        base_output_path = output_path_arg
    else:
        base_output_path = docs_dir / output_path_arg

    base_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_output_path.resolve()}")

    # Process recipes
    recipes = index_data.get("recipe", [])  # Use .get for safety

    if not recipes:
        print("No recipes found in configuration to process.")
        exit(0)

    processed_count = 0
    error_count = 0

    for i, recipe_data in enumerate(recipes):
        print(f"\n--- Processing Recipe {i+1} / {len(recipes)} ---")
        title = recipe_data.get(
            "title", f"Recipe {i+1}"
        )  # Use default title if missing

        if "notebook" not in recipe_data:
            print(f"Warning: Skipping recipe '{title}' due to missing 'notebook' path.")
            error_count += 1
            continue

        notebook_rel_path_str = recipe_data["notebook"]
        # Resolve notebook path relative to the project root
        notebook_abs_path = (root_path / notebook_rel_path_str).resolve()

        if not notebook_abs_path.exists():
            print(
                f"Error: Notebook file not found for entry '{title}': {notebook_abs_path}"
            )
            error_count += 1
            continue

        # Prepare the dictionary expected by the converter function
        notebook_info = {
            "file": notebook_abs_path,
            "title": title,
            "featured": recipe_data.get("featured", False),
            "integration": recipe_data.get("integration", False),
            "agent": recipe_data.get("agent", False),
            "tags": recipe_data.get("tags", []),
            "relative_repo_path": notebook_rel_path_str,  # Pass relative path for image fixing
        }

        # Call the conversion function from the module
        try:
            convert_notebook_to_markdown(notebook_info, base_output_path)
            processed_count += 1
        except Exception as e:
            print(f"!!! Critical Error processing notebook {notebook_abs_path}: {e}")
            traceback.print_exc()  # Print full traceback for critical errors
            error_count += 1

    # Final Summary
    print(f"\n--- Generation Complete ---")
    print(f"Successfully processed: {processed_count} notebooks")
    print(f"Errors encountered: {error_count}")
    print(f"Markdown files generated in: {base_output_path.resolve()}")

    if error_count > 0:
        exit(1)  # Indicate failure if there were errors


if __name__ == "__main__":
    main()
