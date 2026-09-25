FROM python:3.10-slim

# Hugging Face Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user
# librosa (numba) and matplotlib need a writable cache directory
ENV MPLCONFIGDIR=/tmp/matplotlib NUMBA_CACHE_DIR=/tmp/numba PYTHONUNBUFFERED=1

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
