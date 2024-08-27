import modal
import modal.gpu
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from vllm_outlines_setup import Model, app

MINUTE = 60

web_image = modal.Image.debian_slim(python_version="3.10")

auth_scheme = HTTPBearer()


@app.function(
    image=web_image,
    # secrets=[modal.Secret.from_name("my-inference-secret")],
    container_idle_timeout=MINUTE
    * 20,  # keeps web container alive for 20 minutes (the max)
)
@modal.web_endpoint(method="POST")
def generate_web(
    data: dict, token: HTTPAuthorizationCredentials = Depends(auth_scheme)
):
    import os

    '''
    if token.credentials != os.environ["LLM_INFERENCE_KEY"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    '''
    return Model.generate.remote_gen(data["prompts"], settings=None)
