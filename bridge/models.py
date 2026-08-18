"""Local fallback model cards when Mistral /v1/models is unreachable."""

from .config import MODEL


def local_model_card(model_id: str) -> dict:
    return {
        "id": model_id,
        "object": "model",
        "owned_by": "mistral",
        "permission": [],
    }


def local_models_list() -> dict:
    return {"object": "list", "data": [local_model_card(MODEL)]}

