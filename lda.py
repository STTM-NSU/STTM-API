import json
import os

import numpy as np
from gensim import corpora, models
from sklearn.feature_extraction.text import TfidfVectorizer

from db import get_db_pool, close_db_pool

MODEL_PATH = "lda_model/lda_model.gensim"
DICT_PATH = "lda_model/lda_dictionary.dict"


def filter_by_idf(documents, alpha: float = 0.05):
    texts = [" ".join(doc) for doc in documents]

    vectorizer = TfidfVectorizer(use_idf=True, smooth_idf=True)
    vectorizer.fit(texts)

    idf_values = vectorizer.idf_
    vocab = np.array(vectorizer.get_feature_names_out())

    sorted_indices = np.argsort(idf_values)
    n = len(idf_values)
    k = int(n * alpha)

    allowed_words = set(vocab[sorted_indices[k:n - k]] if k > 0 else vocab)

    filtered_documents = [
        [word for word in doc if word in allowed_words]
        for doc in documents
    ]

    return filtered_documents


def bow_dicts_to_token_lists(bow_dicts):
    tokenized = []
    for bow in bow_dicts:
        if isinstance(bow, dict):
            tokens = [word for word, count in bow.items() for _ in range(count)]
        elif isinstance(bow, list) and bow and isinstance(bow[0], dict):
            tokens = [word for word, count in bow[0].items() for _ in range(count)]
        else:
            raise ValueError(f"Unexpected BOW format: {type(bow)}")
        tokenized.append(tokens)
    return tokenized


def bow_dicts_to_token_vocab(bow_dicts):
    vocabs = []
    for bow in bow_dicts:

        if isinstance(bow, list) and bow and isinstance(bow[0], dict):
            vocab = [word for word, count in bow[0].items()]
        else:
            raise ValueError(f"Unexpected BOW format: {type(bow)}")
        vocabs.append(vocab)
    return vocabs


def save_model(lda, dictionary):
    lda.save(MODEL_PATH)
    dictionary.save(DICT_PATH)


def load_model():
    lda = models.LdaModel.load(MODEL_PATH)
    dictionary = corpora.Dictionary.load(DICT_PATH)
    return lda, dictionary


async def load_bows_from_db(from_date, to_date):
    db_pool = await get_db_pool()
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT bow FROM news_bows
            WHERE date >= $1 AND date < $2
            ORDER BY date
            """,
            from_date, to_date
        )
    await close_db_pool()
    return [json.loads(row['bow']) for row in rows]


async def train_or_update_lda(from_date, to_date, num_topics=5):
    bow_dicts = await load_bows_from_db(from_date, to_date)
    if not bow_dicts:
        raise ValueError("No BOW documents in the provided date range")

    tokenized = bow_dicts_to_token_lists(bow_dicts)
    tokenized = filter_by_idf(tokenized, 0.05)

    if not os.path.exists(MODEL_PATH) or not os.path.exists(DICT_PATH):
        dictionary = corpora.Dictionary(tokenized)
        corpus = [dictionary.doc2bow(text) for text in tokenized]
        lda = models.LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=10,
            update_every=0,
            alpha='auto',
            per_word_topics=True
        )
    else:
        lda, dictionary = load_model()
        old_dict_size = len(dictionary)
        dictionary.add_documents(tokenized)
        new_dict_size = len(dictionary)
        corpus = [dictionary.doc2bow(text) for text in tokenized]

        if new_dict_size > old_dict_size:
            lda = models.LdaModel(
                corpus=corpus,
                id2word=dictionary,
                num_topics=lda.num_topics,
                passes=10,
                update_every=0,
                alpha='auto',
                per_word_topics=True
            )
        else:
            lda.update(corpus)

    save_model(lda, dictionary)

    doc_topics = [lda.get_document_topics(dictionary.doc2bow(doc)) for doc in tokenized]
    topic_word_distributions = {
        f"topic_{i}": lda.show_topic(i, topn=10)
        for i in range(num_topics)
    }

    return {
        "tts": doc_topics,
        "topics": topic_word_distributions
    }


async def predict_topics_for_docs(from_date, to_date):
    bow_dicts = await load_bows_from_db(from_date, to_date)
    if not bow_dicts:
        raise ValueError("No BOW documents in the provided date range")

    tokenized = bow_dicts_to_token_lists(bow_dicts)
    tokenized_docs = filter_by_idf(tokenized, 0.05)

    lda, dictionary = load_model()

    bow_corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]

    doc_topics = [lda.get_document_topics(bow) for bow in bow_corpus]

    topic_word_distributions = {
        f"topic_{i}": lda.show_topic(i, topn=10)
        for i in range(20)
    }

    return {
        "tts": doc_topics,
        "topics": topic_word_distributions
    }

# async def main():
#     from datetime import datetime
#
#     result = await train_or_update_lda(
#         from_date=datetime(2022, 1, 1),
#         to_date=datetime(2025, 5, 11),
#         num_topics=20
#     )
#     print(result)
#
#     # result = await predict_topics_for_docs(datetime(2024, 5, 1, 1), datetime(2024, 5, 1, 2))
#     # print(result)
#
# asyncio.run(main())
