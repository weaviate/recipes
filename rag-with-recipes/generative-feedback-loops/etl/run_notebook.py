import os
import sys

def run_notebook(notebook_path):
    command = f"poetry run jupyter execute {notebook_path}"
    os.system(command)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_notebook.py <notebook_path>")
        sys.exit(1)

    notebook_path = sys.argv[1]
    run_notebook(notebook_path)