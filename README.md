# Emily AI Server

A small FastAPI server for Emily's text chat.

## Environment variables

Set these in the hosting provider's secret/environment settings. Never commit them.

- `OPENAI_API_KEY` — secret OpenAI API key
- `EMILY_APP_TOKEN` — long random temporary access token for the `/chat` endpoint
- `OPENAI_MODEL` — optional; defaults to `gpt-4.1-mini`

## Endpoints

- `GET /` — service status
- `GET /health` — health check
- `POST /chat` — chat response; requires `X-Emily-App-Token`

Example request body:

```json
{
  "messages": [
    {"role": "user", "content": "Привет, Эмили!"}
  ]
}
```

The token embedded in a mobile app can be extracted; it is only a temporary barrier for a prototype, not a substitute for production authentication, abuse protection, and per-user limits.
