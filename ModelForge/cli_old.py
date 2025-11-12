import sys
import warnings

from dotenv import load_dotenv
from huggingface_hub import HfApi
from huggingface_hub import errors as hf_errors
import os
import signal
from socket import gaierror

load_dotenv()

def main():
    if len(sys.argv) == 2 and sys.argv[1] == "run":
        ## Validate HuggingFace login
        try:
            api = HfApi()
        except hf_errors.LocalTokenNotFoundError:
            print(f"""
            {'*' * 100}
            You are not logged in to the Hugging Face Hub. 
            1) Create an account on https://huggingface.co/
            2) Generate a finegrained API token from your account settings (https://huggingface.co/docs/hub/en/security-tokens).
            3) Run the command below to login:
                huggingface-cli login
            4) Paste the token when prompted.
            {'*' * 100}
            """)
            os.kill(os.getpid(), signal.SIGTERM)
        except gaierror:
            warnings.warn(
                """
                You are not connected to the internet.\n
                Downloading new models from Hugging Face is not enabled.\n
                Please connect to the internet to finetune any new models.
                """
            )
        except hf_errors.HTTPError:
            warnings.warn(
                """
                You are not connected to the internet.\n
                Downloading new models from Hugging Face is not enabled.\n
                Please connect to the internet to finetune any new models.
                """
            )

        from . import app
        app.main()
    else:
        print("Usage: modelforge run")
        sys.exit(1)

if __name__ == "__main__":
    main()
