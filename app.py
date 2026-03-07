from __future__ import annotations

import uvicorn

from src.api.app import app as fastapi_app

app = fastapi_app


if __name__ == "__main__":
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)
