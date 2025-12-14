# Description: This script downloads the Kronos model and tokenizer from Hugging Face Hub
# and saves them to a local directory named `pretrained_models`.
#
# Usage:
#   1. Make sure you have the `huggingface_hub` library installed (`pip install huggingface_hub`).
#   2. Run this script from your terminal: `python download_model.py`
#
#   This will create a `pretrained_models` directory in your current working directory and
#   download the model and tokenizer files into it. You can then use these local files
#   to run the Kronos model in an offline environment.

import os
from huggingface_hub import snapshot_download

def download_kronos_models():
    """
    Downloads the Kronos model and tokenizer from Hugging Face Hub and
    saves them to a local directory.
    """
    # Create a directory to save the pretrained models
    save_dir = "pretrained_models"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the model and tokenizer names and revisions from the test script
    models = {
        "Kronos-Tokenizer-base": {
            "id": "NeoQuasar/Kronos-Tokenizer-base",
            "revision": "0e0117387f39004a9016484a186a908917e22426"
        },
        "Kronos-small": {
            "id": "NeoQuasar/Kronos-small",
            "revision": "901c26c1332695a2a8f243eb2f37243a37bea320"
        }
    }

    # Download the models and tokenizers
    for model_name, model_info in models.items():
        print(f"Downloading {model_name} from {model_info['id']} at revision {model_info['revision']}...")
        snapshot_download(
            repo_id=model_info['id'],
            revision=model_info['revision'],
            local_dir=os.path.join(save_dir, model_name),
            local_dir_use_symlinks=False  # Set to False to download the actual files
        )
        print(f"Successfully downloaded {model_name} to {os.path.join(save_dir, model_name)}")

    print("\nAll models have been downloaded successfully!")
    print(f"You can now find them in the '{save_dir}' directory.")

if __name__ == "__main__":
    download_kronos_models()
