"""FastAPI application factory."""

from fastapi import FastAPI

from .routes import router

app = FastAPI(title="Mistral GLM Bridge", version="1.0.0")
app.include_router(router)
