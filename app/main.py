from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yaml
import os
from .api.v1.endpoints import disease

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), "config/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)["app"]

app = FastAPI(
    title=config["title"],
    version=config["version"],
    description=config["description"]
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(disease.router, prefix="/api/v1", tags=["Disease Detection"])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=config["host"],
        port=config["port"],
        reload=True
    )
