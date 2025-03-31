FROM ghcr.io/astral-sh/uv:bookworm

# Copy the project into the image
ADD . /app

# Sync the project into a new environment, using the frozen lockfile
WORKDIR /app
RUN uv sync --all-groups --frozen

CMD ["uv", "run", "agentia", "serve"]