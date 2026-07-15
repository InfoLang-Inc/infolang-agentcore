# AgentCore Runtime requires a linux/arm64 container that serves HTTP on :8080
# with an /invocations endpoint and a /ping health check. BedrockAgentCoreApp
# provides both.
FROM --platform=linux/arm64 python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent.py .

# InfoLang credentials are injected at deploy time (env var or AgentCore
# Identity / Secrets). Never bake keys into the image.
ENV INFOLANG_API_KEY=""

EXPOSE 8080
CMD ["python", "agent.py"]
