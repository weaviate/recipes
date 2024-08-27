import modal

HF_TOKEN = "YOUR_HUGGINGFACE_TOKEN" # Replace this with your HuggingFace Token
MODELS_DIR = "/llamas"
MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_REVISION = "8c22764a7e3675c50d4c7c9a4edb474456022b16"  # pin model revisions to prevent unexpected changes!

volume = modal.Volume.from_name("llamas", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        [
            "huggingface_hub",  # download models from the Hugging Face Hub
            "hf-transfer",  # download models faster with Rust
        ]
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

MINUTES = 60
HOURS = 60 * MINUTES


app = modal.App(image=image, secrets=[modal.Secret.from_name("huggingface-secret")])


@app.function(volumes={MODELS_DIR: volume}, timeout=4 * HOURS)
def download_model(model_name, model_revision, force_download=False):

    from huggingface_hub import snapshot_download

    volume.reload()

    snapshot_download(
        MODEL_ID,
        local_dir=MODELS_DIR,
        ignore_patterns=[
            "*.pt",
            "*.bin",
            "*.pth",
            "original/*",
        ],  # Ensure safetensors
        revision=MODEL_REVISION,
        token=HF_TOKEN,
    )

    volume.commit()


@app.local_entrypoint()
def main(
    model_name: str = MODEL_ID,
    model_revision: str = MODEL_REVISION,
    force_download: bool = False,
):
    download_model.remote(model_name, model_revision, force_download)


def extract_assistant_response(output_text):
    """Model-specific code to extract model responses.

    See this doc for LLaMA 3: https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/.
    """
    # Split the output text by the assistant header token
    parts = output_text.split("<|start_header_id|>assistant<|end_header_id|>")

    if len(parts) > 1:
        # Join the parts after the first occurrence of the assistant header token
        response = parts[1].split("<|eot_id|>")[0].strip()

        # Remove any remaining special tokens and whitespace
        response = response.replace("<|eot_id|>", "").strip()

        return response
    else:
        return output_text
