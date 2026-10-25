FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY autonas ./autonas
COPY nas_graph ./nas_graph
RUN pip install --no-cache-dir .
ENTRYPOINT ["autonas"]
CMD ["bench", "--quick"]
