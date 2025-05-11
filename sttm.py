import asyncio
from datetime import datetime

from decimal import Decimal
from unicodedata import decimal

from db import get_db_pool, close_db_pool


async def set_sttm_index_to_db(sttm_index, instrument, from_date, to_date, alpha, p_value, threshold):
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        await conn.fetch(
            """
            INSERT INTO sttm_indexes (index, instrument_id, from_time, to_time, alpha, p_value, threshold)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            sttm_index, instrument, from_date, to_date, alpha, p_value, threshold
        )
    await close_db_pool()


async def get_sttm_index_from_db(instrument, from_date, to_date, alpha, p_value, threshold):
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT index FROM sttm_indexes
            WHERE instrument_id = $1
              AND from_time = $2
              AND to_time = $3
              AND alpha = $4
              AND p_value = $5
              And threshold = $6
            """,
            instrument, from_date, to_date, alpha, p_value, threshold
        )
    await close_db_pool()

    if row is None:
        return None
    return row['index']


def get_sttm_index(topics_tone, topic_stream):
    sttm_index = 0.0
    for hour, topics in topics_tone.items():
        for topic, tone in topics.items():
            topic_val = topic_stream.get(hour).get(topic, 0.0)
            sttm_index += tone * topic_val
    return sttm_index


async def main():
    await set_sttm_index_to_db(500, "index", datetime(2020, 11, 2), datetime(2020, 11, 9),
                               Decimal(str(0.05)), Decimal(str(0.05)), Decimal(str(0.3)))
    idx = await get_sttm_index_from_db("index", datetime(2020, 11, 2), datetime(2020, 11, 9),
                                       Decimal(str(0.05)), Decimal(str(0.05)), Decimal(str(0.3)))
    print(idx)


# Запуск
if __name__ == "__main__":
    asyncio.run(main())
