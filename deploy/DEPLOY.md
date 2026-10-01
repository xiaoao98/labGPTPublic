# LabGPT offline bundle

Everything needed to run the service on a host with no internet access and no root:
a Python 3.11 interpreter, every wheel, the application, the embedding model and the
index. Nothing is downloaded during install; `pip` runs with `--no-index`, so a missing
wheel fails loudly here instead of hanging against an unreachable pypi.

Built for x86_64 Linux, glibc 2.28 or newer. Verified on RHEL 9.

## Install

    tar xzf labgpt-bundle.tar.gz
    cd labgpt-bundle
    ./install.sh

Then fill in `labgpt.env`, which install.sh creates with 600 permissions:

    export LABGPT_API_TOKEN=            # generate one, see below
    export AZURE_OPENAI_GATEWAY=
    export AZURE_OPENAI_TEAM_ID=
    export AZURE_OPENAI_MODEL_ID=       # the deployment name, not the model family
    export AZURE_OPENAI_API_VERSION=
    export APIM_OPENAI_SUBSCRIPTION_KEY=

A token: `./venv/bin/python -c "import secrets; print(secrets.token_urlsafe(24))"`

## Run

    ./run.sh                            # foreground, port 8090

`LABGPT_PORT` changes the port. To leave it running after you log out:

    nohup ./run.sh > labgpt.log 2>&1 &

and to bring it back after a reboot, in `crontab -e`:

    @reboot cd /path/to/labgpt-bundle && ./run.sh >> labgpt.log 2>&1

Check it with `curl -sS http://127.0.0.1:8090/health`, which reports what loaded and
never what is in the corpus.

## What is in here

    python-linux-x86_64.tar.gz   CPython 3.11, relocatable, unpacked to ./python
    wheels/                      every dependency, CPU-only torch, no CUDA
    app/                         the service, the retrieval package, the page
    models/bge-small-en-v1.5     the embedding model the index was built with
    index/                       the index itself
    qa.jsonl                     written at runtime: every question and answer

## Two things to keep in mind

`index/` and `qa.jsonl` contain the corpus. The index holds the document text the
answers quote, and the log holds both the questions people asked and the answers they
got. Both belong on this host, readable by the account that runs the service, and not in
a backup that leaves the building.

The service is HTTP, not HTTPS. The access token travels on every request and answers
quote the corpus, so both are readable by anything on the network path. That is the
reason to put TLS in front of this before the address is circulated widely.
