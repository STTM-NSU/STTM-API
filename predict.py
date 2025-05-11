import asyncio
import json
import math
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from db import get_db_pool, close_db_pool
from lda import load_model
from sttm import get_sttm_index
from topics_tone import get_topics_tone, select_words_of_topic_word_distributions
from words_tone import get_words_tone


def get_allowed_words(documents, alpha: float = 0.05):
    texts = [" ".join(doc) for doc in documents]

    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
    vectorizer.fit(texts)

    idf_values = vectorizer.idf_
    vocab = np.array(vectorizer.get_feature_names_out())

    sorted_indices = np.argsort(idf_values)
    n = len(idf_values)
    k = int(n * alpha)

    return set(vocab[sorted_indices[k:n - k]] if k > 0 else vocab)


async def load_bows_with_dates(from_date, to_date):
    from_date = from_date - timedelta(hours=4)
    to_date = to_date - timedelta(hours=4)
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT date, bow FROM news_bows
            WHERE date >= $1 AND date < $2
            ORDER BY date
            """,
            from_date, to_date
        )
    await close_db_pool()

    result = []
    for row in rows:
        dt = row['date']  # type: datetime
        bow = json.loads(row['bow'])
        result.append((dt, bow))
    return result


def bow_dicts_to_token_lists(bow_dicts):
    tokenized = []
    for _, bow in bow_dicts:
        if isinstance(bow, dict):
            tokens = [word for word, count in bow.items() for _ in range(count)]
        elif isinstance(bow, list) and bow and isinstance(bow[0], dict):
            tokens = [word for word, count in bow[0].items() for _ in range(count)]
        else:
            raise ValueError(f"Unexpected BOW format: {type(bow)}")

        tokenized.append(tokens)
    return tokenized


async def get_token_map(from_date, to_date, alpha=0.05):
    bows_with_dates = await load_bows_with_dates(from_date, to_date)
    tokens = bow_dicts_to_token_lists(bows_with_dates)
    allowed_words = get_allowed_words(tokens, alpha)

    token_map = defaultdict(list)
    for dt, bow in bows_with_dates:
        if isinstance(bow, dict):
            tokens = [word for word, count in bow.items() for _ in range(count) if word in allowed_words]
        elif isinstance(bow, list) and bow and isinstance(bow[0], dict):
            tokens = [word for word, count in bow[0].items() for _ in range(count) if word in allowed_words]

        else:
            raise ValueError(f"Unexpected BOW format: {type(bow)}")

        token_map[dt].append(tokens)
    return token_map


def predict_topics_for_docs(tokenized_docs):
    lda, dictionary = load_model()
    topic_stream = defaultdict(lambda: defaultdict(float))
    word_stream = defaultdict(lambda: defaultdict(int))
    topic_word_distributions = defaultdict(dict)
    for hour, hours in tokenized_docs.items():
        bow_corpus = [dictionary.doc2bow(tokens) for tokens in hours]
        doc_topics = [lda.get_document_topics(bow) for bow in bow_corpus]
        for d in doc_topics:
            for topic, value in d:
                topic_stream[hour][topic] += value
        for bow in bow_corpus:
            for k, v in bow:
                word_stream[hour][dictionary[k]] += v
        topic_word_distributions[hour] = {
            i: lda.show_topic(i, topn=10000)
            for i in range(lda.num_topics)
        }

    return {
        "topic_stream": topic_stream,
        "word_stream": word_stream,
        "topic_word_distributions": topic_word_distributions,
    }


async def main():
    from_date = datetime(2022, 11, 1)
    to_date = datetime(2022, 11, 8)

    instruments = ["TCS00A0ZZAC4", "BBG004730N88", "BBG008F2T3T2", "BBG004S686W0"]

    arr = [float]

    p_value = 0.05
    alpha_idf = 0.05
    threshold = 0.3
    token_map = await get_token_map(from_date, to_date, alpha_idf)
    prediction = predict_topics_for_docs(token_map)
    # print(prediction)
    words_set = {word for _, hours in token_map.items() for words in hours for word in words}
    selected, word_set = select_words_of_topic_word_distributions(prediction.get("topic_word_distributions"), threshold,words_set)
    print(len(word_set))
    for instrument in instruments:
        words_tone = await get_words_tone(from_date, to_date, instrument, prediction.get("word_stream"), word_set.copy(), p_value)
        if words_tone is None:
            arr.append(-1000000000)
            print("err")
            continue
        topics_tone = get_topics_tone(selected, words_tone)
        sttm_index = get_sttm_index(topics_tone, prediction.get("topic_stream"))
        arr.append(sttm_index)
        print(f'{instrument}:{sttm_index}')
    print(arr)


# Запуск
if __name__ == "__main__":
    asyncio.run(main())
