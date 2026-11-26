from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ai_research_assistant.api.container import AppContainer, build_default_container
from ai_research_assistant.api.routes import chat, documents, health, sessions
from ai_research_assistant.config import Settings, get_settings


def create_app(
    container: AppContainer | None = None,
    settings: Settings | None = None,
) -> FastAPI:
    active_settings = (
        container.settings
        if container is not None
        else (settings or get_settings())
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if container is not None:
            app.state.container = container
            await container.sqlite_store.init_db()
        else:
            app.state.container = await build_default_container(active_settings)
        yield

    app = FastAPI(
        title="AI Research Assistant API",
        version="0.6.0",
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router)
    app.include_router(sessions.router)
    app.include_router(documents.router)
    app.include_router(chat.router)
    return app


app = create_app()