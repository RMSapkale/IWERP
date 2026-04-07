import os
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base

# Load environment variables
load_dotenv()

# Database URL from environment or fallback
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+asyncpg://postgres:postgres@localhost:5432/oracle_fusion_slm")


engine = create_async_engine(
    DATABASE_URL, 
    echo=False,
    connect_args={"server_settings": {"search_path": "tenant_demo,public"}}
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
