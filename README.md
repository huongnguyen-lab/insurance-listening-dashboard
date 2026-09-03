# Insurance Listening Dashboard — cloud snapshot

Independent, read-only deployment of the dashboard. It does not use or modify
the development dashboard or its `data_run` directory.

## Local Docker check

```bash
docker build -t insurance-dashboard .
docker run --rm -p 8001:8000 insurance-dashboard
```

Open `http://localhost:8001` or check `http://localhost:8001/health`.

## Render

1. Push this directory to a private GitHub repository.
2. In Render, create a Blueprint and select the repository.
3. Render reads `render.yaml`, builds the image, and exposes a permanent URL.

No Gemini or other LLM API key is needed for this read-only dashboard.
