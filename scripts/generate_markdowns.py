# Full updated code:

import argparse
from pathlib import Path
import re
import html  # For potential unescaping if needed later

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


from nbconvert import MarkdownExporter

# from nbconvert.filters.strings import get_lines # Not currently used, uncomment if needed
from nbformat import read, NO_CONVERT

RAW_IMAGE_PATH = "https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/"

# --- Helper Functions for Style Conversion ---


def to_camel_case(snake_str):
    """Converts kebab-case (CSS) or snake_case to camelCase."""
    snake_str = snake_str.replace("-", "_")
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def create_react_style_object_items_string(css_string):
    """
    Parses a CSS style string and converts it into a React style object's
    INNER ITEMS string (key: value, ...). NO outer braces {}.
    """
    style_dict = {}
    declarations = [d.strip() for d in css_string.strip().split(";") if d.strip()]
    for decl in declarations:
        if ":" not in decl:
            continue
        prop, val = decl.split(":", 1)
        prop = prop.strip()
        val = val.strip()
        if prop and val:
            camel_case_prop = to_camel_case(prop)
            escaped_val = val.replace("'", "\\'")  # Escape single quotes for JS
            style_dict[camel_case_prop] = (
                f"'{escaped_val}'"  # Value as JS string literal
            )
    items_str = ", ".join([f"{k}: {v}" for k, v in style_dict.items()])
    return items_str


# Regex to find *any* HTML tag that contains a style="..." attribute.
style_attr_pattern = re.compile(
    r"""
    <                                    # Opening bracket of the tag
    (?P<tag>[a-zA-Z0-9]+)                # Capture tag name (e.g., 'pre', 'div') -> group 'tag'
    (?P<before>                          # Capture attributes BEFORE style -> group 'before'
        (?:                              # Non-capturing group for any attribute
            \s+                          # Whitespace separator
            (?!style\s*=)                # Negative lookahead: NOT the style attribute
            [\w\-]+                      # Attribute name
            (?:                          # Optional attribute value part
                \s*=\s* # Equals sign (optional whitespace)
                (?:
                    (?P<q1>["']).*?(?P=q1) # Quoted value
                    |
                    [^\s>]+               # Unquoted value
                )
            )?
        )* # Match zero or more such attributes
    )?                                   # Make the whole 'before' group optional
    (?P<style_chunk>                     # Capture the entire style attribute chunk -> group 'style_chunk'
        \s+style=                        # The style attribute itself
        (?P<quote>["'])                  # Capture the quote used -> group 'quote'
        (?P<css>.*?)                     # Capture the CSS content -> group 'css'
        (?P=quote)                       # Match the opening quote
    )
    (?P<after>.*?)                       # Capture attributes AFTER style (non-greedy) -> group 'after'
    (?P<selfclose>\s*/?)?                # Optional self-closing slash (allow space before) -> group 'selfclose'
    >                                    # Closing bracket of the tag
    """,
    re.IGNORECASE | re.VERBOSE | re.DOTALL,
)


def process_any_tag_style(match):
    """
    Callback for re.sub to replace standard style="..." attributes
    with React-style style={{...}} on ANY matched tag.
    """
    tag_name = match.group("tag")
    attrs_before = match.group("before") or ""
    css_string = match.group("css") or ""
    attrs_after = match.group("after") or ""
    self_closing_slash = match.group("selfclose") or ""
    react_style_items_str = create_react_style_object_items_string(css_string)
    if react_style_items_str:
        new_style_attr = f" style={{{{{react_style_items_str}}}}}"
    else:
        new_style_attr = ""
    attrs_before = attrs_before.strip()
    attrs_after = attrs_after.strip()
    parts = [attrs_before, new_style_attr, attrs_after]
    final_attrs_str = " ".join(part for part in parts if part)
    final_space = " " if final_attrs_str else ""
    return f"<{tag_name}{final_space}{final_attrs_str}{self_closing_slash.strip()}>"


# --- Other Helper Functions ---


def generate_frontmatter(notebook):
    """Generates the Docusaurus frontmatter for a recipe."""
    tags_string = str(notebook.get("tags", []))
    frontmatter = f"""---
layout: recipe
colab: {notebook.get("colab")}
toc: True
title: "{notebook["title"]}"
featured: {notebook.get("featured", False)}
integration: {notebook.get("integration", False)}
agent: {notebook.get("agent", False)}
tags: {tags_string}
---
    """
    return frontmatter


def clean_colab_dataframe_cells(notebook):
    """Removes problematic text/html output from Colab dataframes."""
    for cell in notebook.cells:
        if cell.cell_type == "code" and "outputs" in cell:
            for output in cell.outputs:
                data = output.get("data", {})
                if "application/vnd.google.colaboratory.intrinsic+json" in data:
                    intrinsic_json = data[
                        "application/vnd.google.colaboratory.intrinsic+json"
                    ]
                    if intrinsic_json.get("type") == "dataframe":
                        data.pop("text/html", None)
    return notebook


# --- Character Escaping Callbacks ---
def escape_standalone_curly_braces(match):
    """Callback to escape standalone '{' not in code or style props."""
    # Groups: 1=fenced code, 2=inline code, 3=JSX style, 4=standalone {
    if match.group(1) or match.group(2) or match.group(3):
        return match.group(0)
    elif match.group(4):
        return r"\{"
    return match.group(0)


def escape_standalone_dollar(match):
    """Callback to escape standalone '$' not inside code blocks."""
    # Groups: 1=fenced code, 2=inline code, 3=standalone $
    if match.group(1) or match.group(2):  # Code block
        return match.group(0)  # Return unchanged
    elif match.group(3):  # Standalone $
        return "&#36;"  # Return HTML entity
    return match.group(0)  # Fallback


# --- <pre> Formatting Logic ---

# Pattern to find <pre> blocks and capture attributes and content
# Used for multiple steps: plain text check, span stripping, final formatting
pre_block_pattern = re.compile(
    r"<pre(?P<attrs>[^>]*)>(?P<content>.*?)</pre>", re.IGNORECASE | re.DOTALL
)


# Callback function to strip <span> tags from within <pre> blocks
def strip_spans_in_pre(match):
    """Callback specifically to remove span tags inside pre tags."""
    pre_attrs = match.group("attrs") or ""
    pre_content = match.group("content") or ""
    content_no_spans = re.sub(r"</?span[^>]*>", "", pre_content, flags=re.IGNORECASE)
    attrs_stripped = pre_attrs.strip()
    attr_space = " " if attrs_stripped else ""
    # Return with cleaned content, but original newline structure for format_pre_content_final to analyze
    return f"<pre{attr_space}{attrs_stripped}>{content_no_spans}</pre>"


# This is the version of the formatter from *before* the incorrect single-line change
# It aims to join the first line with the tag and remove trailing newlines.
def format_pre_content_final(match):
    """
    Callback to format content within a <pre> block AFTER spans have been stripped.
    - Removes the single leading newline immediately after the opening <pre> tag, if present.
    - Removes trailing whitespace/newline before </pre>.
    - Preserves internal newlines within the content block.
    """
    attrs = match.group("attrs") or ""
    content = match.group("content") or ""  # Content already has spans stripped
    content_no_leading_newline = content.lstrip("\n")
    content_final = content_no_leading_newline.rstrip()
    attrs_stripped = attrs.strip()
    attr_space = " " if attrs_stripped else ""
    return f"<pre{attr_space}{attrs_stripped}>{content_final}</pre>"


# --- Main Conversion Function ---


def generate_markdown_from_notebook(notebook, output_path_base):
    """
    Converts a Jupyter notebook to a Docusaurus-compatible Markdown file.
    """
    frontmatter = generate_frontmatter(notebook)
    md_exporter = MarkdownExporter(exclude_output=False)

    notebook_path = Path(notebook["file"])
    if not notebook_path.exists():
        print(f"Error: Notebook file not found: {notebook_path}")
        return

    print(f"Processing {notebook_path}...")

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = read(f, as_version=NO_CONVERT)

    cleaned_nb = clean_colab_dataframe_cells(nb)

    try:
        body, _ = md_exporter.from_notebook_node(cleaned_nb)
    except Exception as e:
        print(f"Error during nbconvert processing for {notebook_path}: {e}")
        return

    # --- Convert Plain-Text <pre> outputs to Fenced Code Blocks ---
    # This happens BEFORE other processing steps
    plain_text_pres = []
    for match in pre_block_pattern.finditer(body):
        content = match.group("content")
        # Check if original content contains any HTML tag (< followed by letter or /)
        if not re.search(r"<[a-zA-Z/]", content):
            plain_text_pres.append(
                {
                    "start": match.start(),
                    "end": match.end(),
                    "content": content.strip(),  # Use stripped content for the code block
                }
            )

    # Replace plain text pre blocks with fenced code blocks, working backwards
    if plain_text_pres:
        print(
            f"Converting {len(plain_text_pres)} plain-text <pre> blocks to ``` blocks."
        )
        for block_info in sorted(
            plain_text_pres, key=lambda x: x["start"], reverse=True
        ):
            start, end, content = (
                block_info["start"],
                block_info["end"],
                block_info["content"],
            )
            # Add language hint 'text'. Add newlines for proper Markdown block formatting.
            fenced_block = f"\n```text\n{content}\n```\n"
            body = body[:start] + fenced_block + body[end:]
    # --- End Plain Text Conversion ---

    # --- Strip <span> tags from within REMAINING <pre> blocks ---
    # Runs AFTER plain text conversion, so pre_block_pattern only matches relevant <pre> tags
    try:
        body = pre_block_pattern.sub(strip_spans_in_pre, body)
    except Exception as e:
        print(
            f"Warning: Regex for stripping spans within remaining <pre> failed on {notebook_path}: {e}"
        )
    # --- End Span Stripping ---

    # --- Docusaurus & Formatting Fixes ---

    # 1. Convert inline styles on ANY remaining tag (including the remaining <pre> tags)
    try:
        body = style_attr_pattern.sub(process_any_tag_style, body)
    except Exception as e:
        print(f"Warning: Regex for style conversion failed on {notebook_path}: {e}")

    # 2. Format REMAINING <pre> block content (Apply 'first line follows tag' logic)
    # This only affects <pre> tags that were NOT converted to ```
    try:
        # Using format_pre_content_final as restored
        body = pre_block_pattern.sub(format_pre_content_final, body)
    except Exception as e:
        print(
            f"Warning: Regex for final <pre> content formatting failed on {notebook_path}: {e}"
        )
    # --- End <pre> Content Formatting ---

    # --- Character Escaping ---

    # 3.1 Specific escape for '<weaviate.' and '<ChatRole' (BEFORE general escape)
    try:
        body = re.sub(r"<(weaviate\.)", r"&lt;weaviate.", body, flags=re.IGNORECASE)
        body = re.sub(
            r"<(ChatRole)", r"&lt;ChatRole", body, flags=re.IGNORECASE
        )  # Added ChatRole
    except Exception as e:
        print(f"Warning: Regex for specific escaping failed on {notebook_path}: {e}")

    # 3.2. General escape for '<' when not part of a known HTML/JSX tag start or end
    try:
        body = re.sub(r"<(?![a-zA-Z/!])", "&lt;", body)
    except Exception as e:
        print(f"Warning: Regex for general escaping '<' failed on {notebook_path}: {e}")

    # 4. Escape standalone '{' characters if they aren't in code or style props
    try:
        curly_escape_pattern = r"(```.*?```)|(`.*?`)|(style=\{\{.*?\}\})|(\{)"
        body = re.sub(
            curly_escape_pattern, escape_standalone_curly_braces, body, flags=re.DOTALL
        )
    except Exception as e:  # <--- CORRECTED: Added missing block
        print(
            f"Warning: Regex for escaping '{{' failed on {notebook_path}: {e}"
        )  # <--- CORRECTED: Added missing block

    # 5. Escape standalone '$' characters if they aren't in code blocks
    try:
        dollar_escape_pattern = r"(```.*?```)|(`.*?`)|(\$)"
        body = re.sub(
            dollar_escape_pattern, escape_standalone_dollar, body, flags=re.DOTALL
        )
    except Exception as e:
        print(f"Warning: Regex for escaping '$' failed on {notebook_path}: {e}")

    # --- End Character Escaping ---

    # --- Fix image paths ---
    filename_stem = notebook_path.stem
    try:
        notebook_dir_rel_path = Path(notebook["relative_repo_path"]).parent
    except KeyError:
        print(
            f"Warning: 'relative_repo_path' missing for {notebook['title']}. Image paths might be incorrect."
        )
        notebook_dir_rel_path = Path(".")

    markdown_image_pattern = re.compile(r"!\[(?P<caption>.*?)\]\((?P<image>.*?)\)")

    def replace_image_path(match):
        img_caption = match.group("caption")
        img_path_original = match.group("image")
        parts = img_path_original.split(" ", 1)
        img_path_cleaned = parts[0]
        img_title = f" {parts[1]}" if len(parts) > 1 else ""
        img_path_rel = img_path_cleaned.removeprefix("./")
        # Ensure notebook_dir_rel_path is treated correctly if it's the root '.'
        if str(notebook_dir_rel_path) == ".":
            full_raw_path_str = f"{RAW_IMAGE_PATH.strip('/')}/{img_path_rel}"
        else:
            full_raw_path_str = f"{RAW_IMAGE_PATH.strip('/')}/{(notebook_dir_rel_path / img_path_rel).as_posix()}"

        full_raw_path = full_raw_path_str.replace("\\", "/")  # Ensure posix paths
        return f"![{img_caption}]({full_raw_path}{img_title})"

    try:
        body = markdown_image_pattern.sub(replace_image_path, body)
    except Exception as e:
        print(f"Warning: Image path replacement failed for {notebook_path}: {e}")

    # Determine output subdirectory
    if notebook.get("agent", False):
        output_subdir = "agents"
    elif notebook.get("integration", False):
        output_subdir = "integrations"
    else:
        output_subdir = "weaviate"

    output_dir = Path(output_path_base) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_dir / f"{filename_stem}.md"

    # Write the final Markdown file
    try:
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(frontmatter + "\n")
            f.write(body)
        print(f"Successfully generated: {output_file_path}")
    except Exception as e:
        print(f"Error writing output file {output_file_path}: {e}")


# --- Main Execution Block ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Jupyter Notebooks to Docusaurus Markdown"
    )
    parser.add_argument(
        "--output",
        dest="output",
        default="markdowns",
        help="Base directory for markdown output",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    root_path = script_dir.parent
    print(f"Project root path: {root_path}")

    index_file = root_path / "index.toml"
    print(f"Looking for index file at: {index_file}")
    if not index_file.exists():
        print(f"Error: index.toml not found at {index_file}")
        exit(1)

    try:
        index_toml_content = index_file.read_text(encoding="utf-8")
        index_data = tomllib.loads(index_toml_content)
        print("Successfully loaded index.toml")
    except Exception as e:
        print(f"Error parsing index.toml: {e}")
        exit(1)

    base_output_path = Path(args.output)
    base_output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {base_output_path.resolve()}")

    if "config" not in index_data or "colab" not in index_data["config"]:
        print("Error: 'config' section with 'colab' URL base missing in index.toml")
        exit(1)
    colab_base_url = index_data["config"]["colab"]

    if "recipe" not in index_data or not index_data["recipe"]:
        print("Warning: No recipes found in 'recipe' section of index.toml.")
        exit(0)

    processed_count = 0
    error_count = 0
    for i, recipe_data in enumerate(index_data["recipe"]):
        print(f"\n--- Processing Recipe {i+1} ---")
        if "notebook" not in recipe_data or "title" not in recipe_data:
            print(
                f"Warning: Skipping recipe entry due to missing 'notebook' or 'title': {recipe_data}"
            )
            error_count += 1
            continue

        notebook_rel_path_str = recipe_data["notebook"]
        notebook_abs_path = (root_path / notebook_rel_path_str).resolve()

        if not notebook_abs_path.exists():
            print(
                f"Error: Notebook file not found for entry '{recipe_data['title']}': {notebook_abs_path}"
            )
            error_count += 1
            continue

        data = {
            "file": notebook_abs_path,
            "title": recipe_data["title"],
            "colab": f"{colab_base_url.rstrip('/')}/{notebook_rel_path_str}",
            "featured": recipe_data.get("featured", False),
            "integration": recipe_data.get("integration", False),
            "agent": recipe_data.get("agent", False),
            "tags": recipe_data.get("tags", []),
            "relative_repo_path": notebook_rel_path_str,
        }

        try:
            generate_markdown_from_notebook(data, args.output)
            processed_count += 1
        except Exception as e:
            print(f"!!! Critical Error processing notebook {notebook_abs_path}: {e}")
            import traceback

            traceback.print_exc()  # Print full traceback for critical errors
            error_count += 1

    print(f"\n--- Generation Complete ---")
    print(f"Successfully processed: {processed_count} notebooks")
    print(f"Errors encountered: {error_count}")
    print(f"Markdown files generated in: {base_output_path.resolve()}")
