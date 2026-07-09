# Wavelength public server.
#
# Models (~3 GB: BandIt + CLAP + their isolated venvs) are NOT baked into
# the image — they download into the /data volume on first boot and persist
# across upgrades, keeping this image small and rebuilds fast.

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

ENV WAVELENGTH_LIBRARY_DIR=/data/library \
    WAVELENGTH_CACHE_DIR=/data/cache

COPY deploy/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8317
VOLUME /data

ENTRYPOINT ["/entrypoint.sh"]
