import os
from huggingface_hub import HfApi
from fastapi import APIRouter
from fastapi import Request
from starlette.responses import JSONResponse
from huggingface_hub import upload_folder, create_repo
from huggingface_hub.errors import (
    HfHubHTTPError,
    RepositoryNotFoundError,
    GatedRepoError
)
from dotenv import load_dotenv

load_dotenv()

router = APIRouter(
    prefix="/hub",
)

@router.post("/push")
async def push_model_to_hub(request: Request) -> JSONResponse:
    """
    Push a model to HuggingFace Hub.

    Args:
        request: FastAPI request object containing model details.

    Returns:
        JSONResponse with success or error message.
    """
    form = await request.json()
    repo_name = form.get("repo_name")
    model_path = form.get("model_path")
    private = form.get("private", True) # Default to private if not specified

    if not repo_name or not model_path:
        return JSONResponse({"error": "Missing 'repo_name' or 'model_path' in request"}, status_code=400)

    hf_api = HfApi()

    try:
        # Check if the repository already exists
        try:
            hf_api.model_info(repo_name)
            return JSONResponse({"error": f"Repository '{repo_name}' already exists. "
                                          f"We suggest reviewing its contents and using HuggingFace website to update the files manually"
                                          f"to avoid accidental overwrites."}, status_code=400)

        except RepositoryNotFoundError:
            # Create a new repository
            create_repo(repo_name, private=private)

            upload_folder(
                repo_id=repo_name,
                folder_path=model_path,
                path_in_repo="",
                commit_message="Push modelforge model",
                token=os.getenv("HUGGINGFACE_TOKEN"),
            )

            return JSONResponse(
                {"message": f"Model pushed to HuggingFace Hub at '{repo_name}' successfully."
                            f"Navigate to https://huggingface.co/{repo_name} to view your model"
                            f"and edit the model card."},
                status_code=200)
    except GatedRepoError:
        return JSONResponse({"error": f"The repository '{repo_name}' is gated. "
                                      f"Please request access from the repository owner on huggingface.co."}, status_code=403)
    except RepositoryNotFoundError as e:
        print(f"Error: {e}")
        return JSONResponse({"error": f"You do not have the permission to push to '{repo_name}'. "
                                      f"Please ensure your huggingface token grants you write access. "
                                      f"If you are pushing to an organization, contact the administrator for write access."}, status_code=403)
    except HfHubHTTPError as e:
        return JSONResponse({"error": f"Failed to push model to HuggingFace Hub. "
                                      f"Please check your network connection and authentication token. "
                                      f"Error received is: {e}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
