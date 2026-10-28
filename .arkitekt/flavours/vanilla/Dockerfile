FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN mkdir /app
WORKDIR /app

# Outside /app, so copying the project below cannot replace it with a host .venv.
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
# The dependencies alone first, so a change to the code does not install them again.
COPY pyproject.toml uv.lock /app/
# torch for the CPU, from PyTorch's CPU index: no CUDA libraries in the image.
RUN uv sync --frozen --no-dev --no-install-project --extra cpu

ENV PATH="/opt/venv/bin:$PATH"

# The micro-sam weights, baked in before the code, so a pod needs no network for the
# default model and a change to the code does not download them again. The others
# are fetched into the same place on first use, so it stays writable for any user.
ENV MICROSAM_CACHEDIR=/opt/micro_sam
RUN python -c "from micro_sam.util import get_sam_model; get_sam_model('vit_b_lm', device='cpu')" \
    && chmod -R a+rwX /opt/micro_sam

# The whole project: the app target may be any module in it, not only app.py.
COPY . /app
# And the project itself, so `path_sam` (in src/) is importable.
RUN uv sync --frozen --no-dev --extra cpu

# The version of this build, when it is not the one the source declares: a build from
# a branch is passed it by `arkitekt plugin release`. Empty for a release proper.
ARG ARKITEKT_APP_VERSION=""
ENV ARKITEKT_APP_VERSION=$ARKITEKT_APP_VERSION
