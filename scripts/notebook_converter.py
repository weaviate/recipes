# notebook_converter.py
import re
from pathlib import Path
from nbconvert import MarkdownExporter
from nbformat import read, NO_CONVERT
import textwrap  # Added for dedenting

# --- Constants ---
RAW_IMAGE_PATH = "https://raw.githubusercontent.com/weaviate/recipes/refs/heads/main/"

# --- Regex Patterns ---
# Finds HTML tags with style="..." attributes (Keep as is)
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

# Finds <pre> blocks (Still needed for *other* pre blocks, e.g., HTML output)
pre_block_pattern = re.compile(
    r"<pre(?P<attrs>[^>]*)>(?P<content>.*?)</pre>", re.IGNORECASE | re.DOTALL
)

# Finds Markdown image syntax (Keep as is)
markdown_image_pattern = re.compile(r"!\[(?P<caption>.*?)\]\((?P<image>.*?)\)")

# Finds standalone '{' not in code/style (Keep as is)
curly_escape_pattern = re.compile(
    r"(```.*?```)|(`.*?`)|(style=\{\{.*?\}\})|(\{)", flags=re.DOTALL
)

# Finds standalone '$' not in code (Keep as is)
dollar_escape_pattern = re.compile(r"(```.*?```)|(`.*?`)|(\$)", flags=re.DOTALL)

# Regex for removing WhatNext lines ***
what_next_import_pattern = re.compile(
    r"^\s*import\s+WhatsNext\s+from\s+['\"]/_includes/what-next\.mdx['\"]\s*$",
    flags=re.MULTILINE,
)
what_next_component_pattern = re.compile(r"^\s*<WhatsNext\s*/>\s*$", flags=re.MULTILINE)


# --- Helper Functions: Style Conversion --- (Keep as is)
def _to_camel_case(snake_str):
    snake_str = snake_str.replace("-", "_")
    components = snake_str.split("_")
    return components[0] + "".join(x.title() for x in components[1:])


def _create_react_style_object_items_string(css_string):
    style_dict = {}
    declarations = [d.strip() for d in css_string.strip().split(";") if d.strip()]
    for decl in declarations:
        if ":" not in decl:
            continue
        prop, val = decl.split(":", 1)
        prop, val = prop.strip(), val.strip()
        if prop and val:
            camel_case_prop = _to_camel_case(prop)
            escaped_val = val.replace("'", "\\'")
            style_dict[camel_case_prop] = f"'{escaped_val}'"
    return ", ".join([f"{k}: {v}" for k, v in style_dict.items()])


def _process_any_tag_style_match(match):
    tag_name = match.group("tag")
    attrs_before = match.group("before") or ""
    css_string = match.group("css") or ""
    attrs_after = match.group("after") or ""
    self_closing_slash = match.group("selfclose") or ""
    react_style_items_str = _create_react_style_object_items_string(css_string)
    new_style_attr = (
        f" style={{{{{react_style_items_str}}}}}" if react_style_items_str else ""
    )
    final_attrs_str = " ".join(
        part
        for part in [attrs_before.strip(), new_style_attr, attrs_after.strip()]
        if part
    )
    final_space = " " if final_attrs_str else ""
    return f"<{tag_name}{final_space}{final_attrs_str}{self_closing_slash.strip()}>"


# --- Helper Functions: Character Escaping --- (Keep as is)
def _escape_standalone_curly_braces_match(match):
    if (
        match.group(1) or match.group(2) or match.group(3)
    ):  # fenced code, inline code, style prop
        return match.group(0)
    elif match.group(4):  # standalone {
        return r"\{"
    return match.group(0)


def _escape_standalone_dollar_match(match):
    if match.group(1) or match.group(2):  # fenced code, inline code
        return match.group(0)
    elif match.group(3):  # standalone $
        return "&#36;"
    return match.group(0)


# --- Helper Functions: <pre> Tag Formatting --- (Keep as is)
def _strip_spans_in_pre_match(match):
    pre_attrs = match.group("attrs") or ""
    pre_content = match.group("content") or ""
    content_no_spans = re.sub(r"</?span[^>]*>", "", pre_content, flags=re.IGNORECASE)
    attrs_stripped = pre_attrs.strip()
    attr_space = " " if attrs_stripped else ""
    return f"<pre{attr_space}{attrs_stripped}>{content_no_spans}</pre>"


def _format_pre_content_final_match(match):
    attrs = match.group("attrs") or ""
    content = match.group("content") or ""  # Spans already stripped
    content_final = content.lstrip("\n").rstrip()
    attrs_stripped = attrs.strip()
    attr_space = " " if attrs_stripped else ""
    return f"<pre{attr_space}{attrs_stripped}>{content_final}</pre>"


# --- Helper Function: Image Path Replacement --- (Keep as is)
def _replace_image_path_match(match, notebook_dir_rel_path_str):
    img_caption = match.group("caption")
    img_path_original = match.group("image")
    parts = img_path_original.split(" ", 1)
    img_path_cleaned = parts[0]
    img_title = f" {parts[1]}" if len(parts) > 1 else ""
    img_path_rel = img_path_cleaned.removeprefix("./")

    notebook_dir_rel_path = Path(notebook_dir_rel_path_str)

    if str(notebook_dir_rel_path) == ".":
        full_raw_path_str = f"{RAW_IMAGE_PATH.strip('/')}/{img_path_rel}"
    else:
        full_raw_path_str = f"{RAW_IMAGE_PATH.strip('/')}/{(notebook_dir_rel_path / img_path_rel).as_posix()}"

    full_raw_path = full_raw_path_str.replace("\\", "/")  # Ensure posix paths
    return f"![{img_caption}]({full_raw_path}{img_title})"


# --- Core Transformation Steps ---


def _generate_frontmatter(notebook_info):  # (Keep as is)
    tags_string = str(notebook_info.get("tags", []))
    # Use textwrap.dedent for cleaner multiline string formatting
    return textwrap.dedent(
        f"""\
    ---
    layout: recipe
    colab: {notebook_info.get("colab")}
    toc: True
    title: "{notebook_info["title"]}"
    featured: {notebook_info.get("featured", False)}
    integration: {notebook_info.get("integration", False)}
    agent: {notebook_info.get("agent", False)}
    tags: {tags_string}
    ---
    """
    )


def _load_and_clean_notebook(notebook_path):  # (Keep as is)
    print(f"  Reading notebook: {notebook_path}...")
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = read(f, as_version=NO_CONVERT)

    print("  Cleaning Colab dataframe cells...")
    for cell in nb.cells:
        if cell.cell_type == "code" and "outputs" in cell:
            for output in cell.outputs:
                data = output.get("data", {})
                if "application/vnd.google.colaboratory.intrinsic+json" in data:
                    intrinsic_json = data[
                        "application/vnd.google.colaboratory.intrinsic+json"
                    ]
                    if intrinsic_json.get("type") == "dataframe":
                        data.pop("text/html", None)
    return nb


def _export_to_initial_markdown(notebook_node):  # (Keep as is)
    print("  Converting notebook to Markdown using nbconvert...")
    md_exporter = MarkdownExporter(exclude_output=False)
    body, _ = md_exporter.from_notebook_node(notebook_node)
    return body


def _format_indented_output_blocks(markdown_body):  # (Keep as is)
    lines = markdown_body.split("\n")
    result = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Check if this is the start of a Python code block
        if line.strip() == "```python":
            # Add the Python code block
            result.append(line)
            i += 1

            # Find the end of the Python block
            python_block_end = i
            while (
                python_block_end < len(lines)
                and lines[python_block_end].strip() != "```"
            ):
                python_block_end += 1

            # Copy the Python block lines
            for j in range(i, min(python_block_end + 1, len(lines))):
                result.append(lines[j])

            # Update i to after the Python block
            i = python_block_end + 1

            # Look for indented lines after the Python block
            empty_lines = []
            while i < len(lines) and not lines[i].strip():
                empty_lines.append(lines[i])
                i += 1

            # Check if there are indented lines
            if i < len(lines) and lines[i].startswith("    "):
                # Add the empty lines
                result.extend(empty_lines)

                # Add "Python output:" before the text code block
                result.append("Python output:")

                # Start a text code block
                result.append("```text")

                # Find the end of the indented block
                indented_block_end = i
                while indented_block_end < len(lines):
                    if (
                        lines[indented_block_end].startswith("    ")
                        or not lines[indented_block_end].strip()
                    ):
                        indented_block_end += 1
                    else:
                        break

                # Store the content lines
                content_lines = []
                for j in range(i, indented_block_end):
                    if lines[j].startswith("    "):
                        content_lines.append(lines[j][4:])  # Remove indentation
                    else:
                        content_lines.append(lines[j])  # Empty line

                # Remove trailing empty lines
                while content_lines and not content_lines[-1].strip():
                    content_lines.pop()

                # Add content lines to result
                result.extend(content_lines)

                # End the text code block
                result.append("```")

                # Update i to after the indented block
                i = indented_block_end
            else:
                # No indented lines, add the empty lines and continue
                result.extend(empty_lines)
        else:
            # Regular line, just add it
            result.append(line)
            i += 1

    return "\n".join(result)


def _strip_spans_from_remaining_pre(markdown_body):  # (Keep as is)
    """Removes <span> tags from within any remaining <pre> blocks."""
    print("  Stripping <span> tags from remaining <pre> blocks (if any)...")
    try:
        # Check if there are actually any pre tags left to process
        if "<pre" in markdown_body.lower():
            count_before = markdown_body.lower().count("<pre")
            body_after = pre_block_pattern.sub(_strip_spans_in_pre_match, markdown_body)
            count_after = body_after.lower().count("<pre")
            if count_before > 0:
                print(f"    Processed {count_before} <pre> blocks for span stripping.")
            else:
                print("    No <pre> blocks found to strip spans from.")
            return body_after
        else:
            print("    No <pre> blocks found.")
            return markdown_body

    except Exception as e:
        print(f"    Warning: Regex for stripping spans within <pre> failed: {e}")
        return markdown_body  # Return original body on error


def _apply_docusaurus_fixes(markdown_body):  # (Keep as is)
    """Applies Docusaurus specific formatting: inline styles and remaining <pre> content format."""
    print("  Applying Docusaurus formatting fixes (styles, remaining <pre>)...")
    # 1. Convert inline styles on *any* tag
    print("    Converting style='...' to React style={{...}}...")
    try:
        body_step1 = style_attr_pattern.sub(_process_any_tag_style_match, markdown_body)
    except Exception as e:
        print(f"    Warning: Regex for style conversion failed: {e}")
        body_step1 = markdown_body

    # 2. Format remaining <pre> block content (if any exist)
    print("    Formatting content within remaining <pre> blocks (if any)...")
    try:
        # Check if there are pre tags before running the potentially expensive regex
        if "<pre" in body_step1.lower():
            count_before = body_step1.lower().count("<pre")
            body_step2 = pre_block_pattern.sub(
                _format_pre_content_final_match, body_step1
            )
            print(
                f"    Processed {count_before} remaining <pre> blocks for content formatting."
            )
            return body_step2
        else:
            print("    No remaining <pre> blocks found to format.")
            return body_step1
    except Exception as e:
        print(f"    Warning: Regex for final <pre> content formatting failed: {e}")
        return body_step1  # Continue with result from step 1


def _escape_special_characters(markdown_body):  # (Keep as is)
    print("  Escaping special characters for Docusaurus/MDX...")
    body = markdown_body
    # Removed specific <weaviate./<ChatRole escapes as requested in comments previously
    # If needed, they can be re-added here

    # 1. General '<' escape
    print("    Escaping general '<' (not part of tags)...")
    try:
        body = re.sub(r"<(?![a-zA-Z/!])", "&lt;", body)
    except Exception as e:
        print(f"    Warning: Regex for general escaping '<' failed: {e}")

    # 2. Escape standalone '{'
    print("    Escaping standalone '{'...")
    try:
        body = curly_escape_pattern.sub(_escape_standalone_curly_braces_match, body)
    except Exception as e:
        print(f"    Warning: Regex for escaping '{{' failed: {e}")

    # 3. Escape standalone '$'
    print("    Escaping standalone '$'...")
    try:
        body = dollar_escape_pattern.sub(_escape_standalone_dollar_match, body)
    except Exception as e:
        print(f"    Warning: Regex for escaping '$' failed: {e}")

    return body


def _fix_image_paths(markdown_body, notebook_info):  # (Keep as is)
    print("  Fixing image paths...")
    try:
        # Check if markdown_image_pattern actually exists in body
        if "![" in markdown_body and "](" in markdown_body:
            notebook_dir_rel_path_str = str(
                Path(notebook_info["relative_repo_path"]).parent
            )
            count_before = markdown_body.count("![")
            fixed_body = markdown_image_pattern.sub(
                lambda match: _replace_image_path_match(
                    match, notebook_dir_rel_path_str
                ),
                markdown_body,
            )
            print(f"    Processed {count_before} potential image path(s).")
            return fixed_body
        else:
            print("    No potential image paths found.")
            return markdown_body
    except KeyError:
        print(
            f"    Warning: 'relative_repo_path' missing for '{notebook_info['title']}'. Image paths might be incorrect."
        )
        return markdown_body  # Return original body if key is missing
    except Exception as e:
        print(f"    Warning: Image path replacement failed: {e}")
        return markdown_body  # Return original body on other errors


# *** NEW HELPER FUNCTION ***
def _remove_whatnext_component(markdown_body):
    """Removes the WhatNext import and component lines."""
    print("  Removing WhatNext component lines...")
    body = markdown_body
    count_import = 0
    count_component = 0

    # Remove import line
    new_body, num_subs_import = what_next_import_pattern.subn("", body)
    if num_subs_import > 0:
        count_import = num_subs_import
        body = new_body
        print(f"    Removed {count_import} WhatNext import line(s).")

    # Remove component line
    new_body, num_subs_component = what_next_component_pattern.subn("", body)
    if num_subs_component > 0:
        count_component = num_subs_component
        body = new_body
        print(f"    Removed {count_component} WhatNext component tag(s).")

    if count_import == 0 and count_component == 0:
        print("    No WhatNext lines found to remove.")

    # Optional: Clean up potential extra blank lines left by removal
    body = re.sub(r"\n{3,}", "\n\n", body).strip()  # Replace 3+ newlines with 2

    return body


def _determine_output_path(notebook_info, output_path_base):  # (Keep as is)
    notebook_path = Path(notebook_info["file"])
    filename_stem = notebook_path.stem

    if notebook_info.get("agent", False):
        output_subdir = "agents"
    elif notebook_info.get("integration", False):
        output_subdir = "integrations"
    else:
        output_subdir = "weaviate"

    output_dir = Path(output_path_base) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{filename_stem}.md"


def _write_output_file(output_path, frontmatter, final_markdown_body):  # (Keep as is)
    print(f"  Writing final Markdown to: {output_path}")
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(frontmatter)  # Frontmatter already has trailing newline
            f.write(final_markdown_body)  # Write body (cleaned of WhatNext)
        print(f"    Successfully generated: {output_path.name}")
    except Exception as e:
        print(f"    Error writing output file {output_path}: {e}")
        raise  # Re-raise the exception to be caught by the main loop


# --- Main Public Function (Updated Workflow) ---


def convert_notebook_to_markdown(notebook_info, output_path_base):
    """
    Converts a Jupyter notebook to Docusaurus-compatible Markdown.

    Args:
        notebook_info (dict): Dictionary containing notebook metadata
                              (file path, title, tags, etc.).
        output_path_base (str or Path): The base directory for markdown output.
    """
    notebook_path = Path(notebook_info.get("file", ""))
    if not notebook_path or not notebook_path.exists():
        print(
            f"Error: Notebook file path missing or invalid in notebook_info: {notebook_path}"
        )
        raise FileNotFoundError(f"Notebook file not found: {notebook_path}")

    print(f"Processing {notebook_path.name}...")

    # Step 1: Generate Frontmatter
    frontmatter = _generate_frontmatter(notebook_info)

    # Step 2: Load and Clean Notebook Node
    try:
        cleaned_nb_node = _load_and_clean_notebook(notebook_path)
    except Exception as e:
        print(f"  Error loading/cleaning notebook {notebook_path}: {e}")
        raise

    # Step 3: Export to Initial Markdown
    try:
        initial_md_body = _export_to_initial_markdown(cleaned_nb_node)
    except Exception as e:
        print(f"  Error during nbconvert processing for {notebook_path}: {e}")
        raise

    # Step 4: Format Indented Output Blocks
    md_body_step4 = _format_indented_output_blocks(initial_md_body)

    # Step 5: Strip <span> from remaining <pre>
    md_body_step5 = _strip_spans_from_remaining_pre(md_body_step4)

    # Step 6: Apply Docusaurus Fixes (Styles, remaining <pre> formatting)
    md_body_step6 = _apply_docusaurus_fixes(md_body_step5)

    # Step 7: Escape Special Characters
    md_body_step7 = _escape_special_characters(md_body_step6)

    # Step 8: Fix Image Paths
    md_body_step8 = _fix_image_paths(md_body_step7, notebook_info)

    # *** Step 9: Remove WhatNext Component (NEW STEP) ***
    md_body_step9 = _remove_whatnext_component(md_body_step8)

    # Step 10: Determine Output Path (Renumbered)
    output_file_path = _determine_output_path(notebook_info, output_path_base)

    # Step 11: Write Output File (Renumbered)
    # Pass the result from the WhatNext removal step
    _write_output_file(output_file_path, frontmatter, md_body_step9)
