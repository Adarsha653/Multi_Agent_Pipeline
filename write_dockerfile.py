with open("Dockerfile", "w") as f:
    lines = [
        "FROM python:3.11-slim\n",
        "\n",
        "RUN useradd -m -u 1000 user\n",
        "USER user\n",
        "ENV PATH=/home/user/.local/bin:$PATH\n",
        "\n",
        "WORKDIR /app\n",
        "\n",
        "COPY --chown=user requirements.txt requirements.txt\n",
        "RUN pip install --no-cache-dir --upgrade -r requirements.txt\n",
        "\n",
        "COPY --chown=user . /app\n",
        "\n",
        'CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]\n',
    ]
    f.writelines(lines)
print("Dockerfile written")
