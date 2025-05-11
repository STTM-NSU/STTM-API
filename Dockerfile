FROM python:3.12-alpine3.21

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_CACHE_DIR=/.cache/pip

RUN apk add --no-cache gcc musl-dev libffi-dev build-base

RUN apk add --no-cache openblas-dev libstdc++ && \
    ln -s /usr/include/locale.h /usr/include/xlocale.h || true

COPY ./ ./

RUN --mount=type=cache,target=/.cache/pip \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
