FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml ./
COPY tvmdbhex ./tvmdbhex
RUN pip install --no-cache-dir .

ENV TVMDBHEX_DB_PATH=/data/tvmdbhex.db
VOLUME ["/data"]
EXPOSE 8000
ENTRYPOINT ["tvmdbhex"]
CMD ["serve", "--port", "8000"]
