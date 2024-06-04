import os
import argparse

def set_env_vars(args):
    """Sets environment variables based on the provided CLI arguments."""

    for arg in vars(args):  
        value = getattr(args, arg)
        if value is not None:  # Only set if a value was provided
            os.environ[arg.upper()] = value  # Set environment variable (uppercase)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set environment variables from CLI arguments.")
    llm_options = [
        "command-r-plus"
    ]
    parser.add_argument("--llm", help=f"Available options are {llm_options}")
    parser.add_argument("--key", help="Your LLM API Key")

    args = parser.parse_args()
    set_env_vars(args)
