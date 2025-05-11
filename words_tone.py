import warnings
from collections import defaultdict
from datetime import timedelta

import pandas as pd
from scipy.stats import pearsonr, ConstantInputWarning

from db import get_db_pool, close_db_pool


async def get_stocks(from_date, to_date, instrument):
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT ts, close_price
            FROM stocks
            WHERE instrument_id = $1 AND ts >= $2 AND ts < $3
            ORDER BY ts;
        """,
                                instrument, from_date, to_date)

    await close_db_pool()
    df = pd.DataFrame([dict(r) for r in rows])
    return df


def get_word_stream(word, word_stream):
    rows = []
    for ts, word_freqs in word_stream.items():
        ts_hour = pd.to_datetime(ts)
        freq = word_freqs.get(word, 0)
        rows.append((ts_hour, freq))

    return pd.DataFrame(rows, columns=['ts', 'freq'])


def get_correlation(stocks, word_stream, p_value=0.05):
    corr = 0.0
    if len(word_stream) > 1:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConstantInputWarning)
            try:
                corr, pval = pearsonr(stocks, word_stream)
                if pval >= p_value:
                    corr = 0.0
            except ConstantInputWarning:
                corr, pval = 0.0, 1.0
        if pval >= 0.05 or corr != corr:
            corr = 0.0
    return corr


async def get_words_tone(from_date, to_date, instrument, word_stream_lda, words_set, p_value):
    stocks = await get_stocks(from_date, to_date, instrument)

    words_tone = defaultdict(float)
    while words_set:
        word = words_set.pop()
        word_stream = get_word_stream(word, word_stream_lda)
        df_merged = pd.merge(stocks, word_stream, on='ts', how='inner').dropna()
        x = df_merged['close_price'].astype(float)
        y = df_merged['freq'].astype(float)
        words_tone[word] = get_correlation(x, y, p_value)
    return words_tone
