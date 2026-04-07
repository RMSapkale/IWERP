import argparse
import asyncio

from sqlalchemy import text

from .session import Base, engine

RESET_CONFIRMATION = "YES_RESET"

SAFE_ALTER_STATEMENTS = (
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS display_name VARCHAR",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS password_hash VARCHAR",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS moe_enabled BOOLEAN DEFAULT FALSE",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS moe_experiment_group VARCHAR DEFAULT 'control'",
    "ALTER TABLE tenants ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ DEFAULT NOW()",
    "ALTER TABLE tenant_api_keys ADD COLUMN IF NOT EXISTS prefix VARCHAR",
    "ALTER TABLE tenant_api_keys ADD COLUMN IF NOT EXISTS is_active BOOLEAN DEFAULT TRUE",
    "ALTER TABLE tenant_api_keys ADD COLUMN IF NOT EXISTS last_used_at TIMESTAMPTZ",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS metadata_json JSON DEFAULT '{}'::json",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS embedding JSON",
    "ALTER TABLE chunks ADD COLUMN IF NOT EXISTS content_tsvector TSVECTOR",
    "ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS error_message TEXT",
    "ALTER TABLE ingest_jobs ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ DEFAULT NOW()",
    "ALTER TABLE sft_samples ADD COLUMN IF NOT EXISTS difficulty VARCHAR DEFAULT 'medium'",
    "ALTER TABLE sft_samples ADD COLUMN IF NOT EXISTS source_doc_ids JSON DEFAULT '[]'::json",
    "ALTER TABLE trace_records ADD COLUMN IF NOT EXISTS task_type VARCHAR",
    "ALTER TABLE trace_records ADD COLUMN IF NOT EXISTS module VARCHAR",
    "ALTER TABLE trace_records ADD COLUMN IF NOT EXISTS retrieved_chunk_ids JSON DEFAULT '[]'::json",
    "ALTER TABLE trace_records ADD COLUMN IF NOT EXISTS retrieval_scores JSON DEFAULT '{}'::json",
    "ALTER TABLE trace_records ADD COLUMN IF NOT EXISTS final_context_size INTEGER",
    "ALTER TABLE trace_records ADD COLUMN IF NOT EXISTS model_settings JSON DEFAULT '{}'::json",
    "ALTER TABLE trace_records ADD COLUMN IF NOT EXISTS latency_ms JSON DEFAULT '{}'::json",
    "ALTER TABLE trace_records ADD COLUMN IF NOT EXISTS audit_logs JSON DEFAULT '{}'::json",
    "ALTER TABLE trace_records ADD COLUMN IF NOT EXISTS policy_decisions JSON DEFAULT '{}'::json",
    "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS issue_type VARCHAR",
    "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS comment TEXT",
    "ALTER TABLE feedback ADD COLUMN IF NOT EXISTS corrected_answer TEXT",
)

INDEX_STATEMENTS = (
    "CREATE INDEX IF NOT EXISTS ix_tenant_api_keys_prefix ON tenant_api_keys (prefix)",
    "CREATE INDEX IF NOT EXISTS ix_tenant_api_keys_key_hash ON tenant_api_keys (key_hash)",
    "CREATE INDEX IF NOT EXISTS ix_chunks_tenant_id ON chunks (tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_ingest_jobs_tenant_id ON ingest_jobs (tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_sft_samples_tenant_id ON sft_samples (tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_trace_records_tenant_id ON trace_records (tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_feedback_tenant_id ON feedback (tenant_id)",
    "CREATE INDEX IF NOT EXISTS ix_chunks_content_tsvector ON chunks USING GIN (content_tsvector)",
)


async def enable_extensions(conn) -> None:
    print("Enabling pgvector extension if available...")
    await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))


async def ensure_base_schema(conn) -> None:
    print("Ensuring tables exist...")
    await conn.run_sync(Base.metadata.create_all)


async def apply_safe_alter_steps(conn) -> None:
    print("Applying additive schema upgrades...")
    for statement in SAFE_ALTER_STATEMENTS:
        await conn.execute(text(statement))


async def ensure_indexes(conn) -> None:
    print("Ensuring indexes exist...")
    for statement in INDEX_STATEMENTS:
        await conn.execute(text(statement))


async def ensure_chunk_search_objects(conn) -> None:
    print("Ensuring FTS function and trigger exist...")
    await conn.execute(
        text(
            """
            CREATE OR REPLACE FUNCTION chunks_tsvector_trigger() RETURNS trigger AS $$
            BEGIN
              new.content_tsvector := to_tsvector('english', new.content);
              return new;
            END
            $$ LANGUAGE plpgsql;
            """
        )
    )
    await conn.execute(text("DROP TRIGGER IF EXISTS tsvectorupdate ON chunks;"))
    await conn.execute(
        text(
            """
            CREATE TRIGGER tsvectorupdate
            BEFORE INSERT OR UPDATE ON chunks
            FOR EACH ROW EXECUTE FUNCTION chunks_tsvector_trigger();
            """
        )
    )


async def ensure_migration_log(conn) -> None:
    await conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS schema_migration_log (
              id BIGSERIAL PRIMARY KEY,
              migration_name VARCHAR NOT NULL,
              mode VARCHAR NOT NULL,
              applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
    )


async def record_migration(conn, mode: str) -> None:
    await ensure_migration_log(conn)
    await conn.execute(
        text(
            """
            INSERT INTO schema_migration_log (migration_name, mode)
            VALUES ('backend.core.database.migrate', :mode)
            """
        ),
        {"mode": mode},
    )


async def ensure_demo_tenant(conn) -> None:
    print("Ensuring demo tenant seed exists...")
    await conn.execute(
        text(
            """
            INSERT INTO tenants (id, display_name, is_active, moe_enabled, moe_experiment_group)
            VALUES ('demo', 'Demo Tenant', TRUE, FALSE, 'control')
            ON CONFLICT (id) DO UPDATE SET
              display_name = COALESCE(tenants.display_name, EXCLUDED.display_name),
              is_active = COALESCE(tenants.is_active, EXCLUDED.is_active),
              moe_enabled = COALESCE(tenants.moe_enabled, EXCLUDED.moe_enabled),
              moe_experiment_group = COALESCE(tenants.moe_experiment_group, EXCLUDED.moe_experiment_group)
            """
        )
    )


async def migrate(mode: str = "safe") -> None:
    async with engine.begin() as conn:
        await enable_extensions(conn)

        if mode == "reset":
            print("Running destructive reset mode...")
            await conn.run_sync(Base.metadata.drop_all)
        else:
            print("Running safe migration mode...")

        await ensure_base_schema(conn)
        await apply_safe_alter_steps(conn)
        await ensure_indexes(conn)
        await ensure_chunk_search_objects(conn)
        await ensure_demo_tenant(conn)
        await record_migration(conn, mode=mode)

        print(f"Migration complete. mode={mode}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IWFUSION database migration utility.")
    parser.add_argument(
        "--mode",
        choices=("safe", "reset"),
        default="safe",
        help="Use 'safe' for production/upgrade flows, or 'reset' for destructive demo/dev rebuilds.",
    )
    parser.add_argument(
        "--force-reset",
        default="",
        help=f"Required for --mode reset. Must equal {RESET_CONFIRMATION}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "reset" and args.force_reset != RESET_CONFIRMATION:
        raise SystemExit(
            f"Refusing destructive reset. Re-run with --mode reset --force-reset {RESET_CONFIRMATION}"
        )
    asyncio.run(migrate(mode=args.mode))


if __name__ == "__main__":
    main()
