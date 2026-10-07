# The service, for a host that runs containers. Built for linux/amd64 on whatever machine
# has internet; the target needs neither internet nor a compiler.
#
# The model and the index are NOT baked in. They are mounted at runtime, for two reasons:
# the index is the corpus, and an image carrying institutional documents is a thing that
# gets copied to laptops and registries without anyone deciding to; and an index rebuild
# then costs a volume swap rather than a rebuild, a push and a pull.
FROM --platform=linux/amd64 python:3.11-slim

# Nothing here compiles: every wheel below is manylinux. If this ever needs build-essential
# the right fix is to find the package that lost its wheel, not to add a compiler.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    LABGPT_EMBED_MODEL=/models/bge-small-en-v1.5 \
    LABGPT_INDEX_DIR=/index \
    LABGPT_LOG_PATH=/data/qa.jsonl

WORKDIR /app

# torch from the CPU index. The default wheels carry CUDA for a GPU this host does not
# have and take the image past 5 GB.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      "torch>=2.2" "pyyaml>=6.0" "numpy>=1.24" "sentence-transformers>=3.0" \
      "transformers>=4.44" "rich>=13.0" "fastapi>=0.110" "uvicorn>=0.29"

COPY labrag/ /app/labrag/
COPY web/ /app/web/
COPY serve.py demo.py labgpt_config.py labgpt_metrics.py /app/

# Runs as nobody. The service reads the index and writes one log file; it has no reason to
# be able to write to its own code.
RUN mkdir -p /data /models /index && chown -R nobody:nogroup /data
USER nobody

EXPOSE 8090

# The embedding model is loaded at startup, which takes a few seconds, and the port does
# not accept connections until it is done. start-period covers that so the container is
# not declared unhealthy while it is still coming up.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8090/health', timeout=4).status==200 else 1)"

CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8090"]
