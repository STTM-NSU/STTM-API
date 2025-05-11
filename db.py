import os
import asyncpg
import asyncio

from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "user": os.getenv("POSTGRES_USERNAME"),
    "password": os.getenv("POSTGRES_PASSWORD"),
    "database": os.getenv("POSTGRES_DB_NAME"),
    "host": os.getenv("POSTGRES_HOST"),
    "port": int(os.getenv("POSTGRES_PORT")),
}

pool = None


async def get_db_pool():
    global pool
    if pool is None:
        pool = await asyncpg.create_pool(**DB_CONFIG)
    return pool


async def close_db_pool():
    global pool
    if pool is not None:
        await pool.close()
        pool = None


async def main():
    conn = await asyncpg.connect(**DB_CONFIG)
    version = await conn.fetch('SELECT version()')
    print(version)
    await conn.close()

    db_pool = await get_db_pool()
    async with db_pool.acquire() as connection:
        tables = await connection.fetch('SELECT table_name FROM information_schema.tables WHERE table_schema = \'public\'')
        print("Public tables:", [r['table_name'] for r in tables])

    await close_db_pool()


if __name__ == "__main__":
    asyncio.run(main())
