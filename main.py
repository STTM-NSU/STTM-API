from datetime import datetime, timedelta
from decimal import Decimal
from typing import List

from fastapi import FastAPI, Query, HTTPException
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from predict import get_token_map, predict_topics_for_docs
from sttm import get_sttm_index_from_db, get_sttm_index, set_sttm_index_to_db
from topics_tone import get_topics_tone, select_words_of_topic_word_distributions
from words_tone import get_words_tone

app = FastAPI()


class STTMQueryParams(BaseModel):
    instrument_ids: str
    from_date: datetime
    to_date: datetime
    alpha: float
    p_value: float
    threshold: float

    def validate_dates(self):
        if (self.to_date - self.from_date).days < 1:
            raise HTTPException(status_code=400, detail="'from' and 'to' must span at least one day")


class STTMIndexResponse(BaseModel):
    indexes: List[float]


@app.get("/get-index", response_model=STTMIndexResponse)
async def get_index(
        instrument_ids: str = Query(...),
        from_: datetime = Query(..., alias="from"),
        to: datetime = Query(...),
        alpha: float = Query(...),
        p_value: float = Query(...),
        threshold: float = Query(...)
):
    params = STTMQueryParams(
        instrument_ids=instrument_ids,
        from_date=from_,
        to_date=to,
        alpha=alpha,
        p_value=p_value,
        threshold=threshold
    )
    params.validate_dates()

    instruments = instrument_ids.split(',')
    indexes = [None] * len(instruments)
    counter = len(instruments)

    for idx, instrument_id in enumerate(instruments):
        cached = await get_sttm_index_from_db(
            instrument_id, from_, to, Decimal(str(alpha)),
            Decimal(str(p_value)), Decimal(str(threshold))
        )
        if cached is not None:
            indexes[idx] = cached
            print(f'{instrument_id}: index was cached = {cached}')
            counter -= 1
    if counter == 0:
        return STTMIndexResponse(indexes=indexes)

    print(f'start')
    token_map = await get_token_map(from_, to, alpha)
    print(f'got token_map')
    words_set = {word for _, hours in token_map.items() for words in hours for word in words}
    prediction = predict_topics_for_docs(token_map)
    print(f'got prediction')
    selected, word_set = select_words_of_topic_word_distributions(
        prediction.get("topic_word_distributions"), threshold, words_set
    )
    print(f'words count {len(word_set)}')

    for idx, instrument_id in enumerate(instruments):
        if indexes[idx] is not None:
            continue
        words_tone = await get_words_tone(
            from_, to, instrument_id,
            prediction.get("word_stream"), word_set.copy(), p_value
        )
        if words_tone is None:
            indexes[idx] = -1000000000
            print(f'{instrument_id}: no stocks')
            continue
        print(f'{instrument_id}: got word tone stream')
        topics_tone = get_topics_tone(selected, words_tone)
        print(f'{instrument_id}: got topic tone stream')
        sttm_index = get_sttm_index(topics_tone, prediction.get("topic_stream"))
        print(f'{instrument_id}: sttm index = {sttm_index}')
        indexes[idx] = sttm_index
        await set_sttm_index_to_db(
            sttm_index, instrument_id, from_, to,
            Decimal(str(alpha)), Decimal(str(p_value)), Decimal(str(threshold))
        )
    return STTMIndexResponse(indexes=indexes)


@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal Server Error", "retry_after": str(timedelta(hours=4))}
    )
