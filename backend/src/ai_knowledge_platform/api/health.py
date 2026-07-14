from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get("/")
async def health_check() -> dict[str, str]:
    return {"message": "AI Knowledge Platform API is running"}
