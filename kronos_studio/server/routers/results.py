from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from schemas.saved_result import (SavedResultDetail, SavedResultMeta,
                                  SaveResultRequest)
from services.result_service.save_results import SavedResultService

router = APIRouter()

saved_result_service = SavedResultService()


@router.post("/saved-results", response_model=SavedResultDetail, status_code=201)
def save_result(request: SaveResultRequest) -> SavedResultDetail:
    """Persist a prediction result to the local SQLite database."""
    return saved_result_service.save(
        result_type=request.type,
        data=request.data,
        label=request.label,
    )


@router.get("/saved-results", response_model=list[SavedResultMeta])
def list_saved_results() -> list[SavedResultMeta]:
    """Return metadata for all saved results, newest first."""
    return saved_result_service.list_all()


@router.get("/saved-results/{result_id}", response_model=SavedResultDetail)
def get_saved_result(result_id: str) -> SavedResultDetail:
    """Return the full saved result for *result_id*."""
    record = saved_result_service.get_by_id(result_id)
    if record is None:
        raise HTTPException(
            status_code=404, detail=f"Saved result '{result_id}' not found."
        )
    return record


@router.delete("/saved-results/{result_id}", status_code=204)
def delete_saved_result(result_id: str) -> Response:
    """Delete a saved result by ID."""
    deleted = saved_result_service.delete(result_id)
    if not deleted:
        raise HTTPException(
            status_code=404, detail=f"Saved result '{result_id}' not found."
        )
    return Response(status_code=204)
