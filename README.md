# TruthCheck AI — iPad deployment package

## Files
- `app.py`: Streamlit app
- `requirements.txt`: Python dependencies

## Deploy
1. Upload these files to the root of a GitHub repository.
2. Open https://share.streamlit.io/ and create an app.
3. Select the repository, branch `main`, and main file `app.py`.
4. Deploy.

## Enable AI analysis
The related-article search works without an OpenAI API key. To enable AI claim analysis with web search, add an API key in Streamlit Community Cloud:

1. Open the app's settings menu in Streamlit Community Cloud.
2. Find **Secrets**.
3. Add the following TOML entry:

```toml
OPENAI_API_KEY = "your-api-key"
```

Never place an API key in `app.py`, GitHub files, screenshots, or public messages. API usage may incur separate charges. The default model is `gpt-6-astra`; if your API project does not have access, set `OPENAI_MODEL` to a model available to that API project in the app environment.

## Important limitations
- Users should paste the article text; the app does not reliably extract full article text from arbitrary URLs.
- AI analysis uses the OpenAI Responses API web search tool and can make mistakes. Always open cited sources and verify them.
- The company-impact calculator compares user-entered stock and market-index values. It does not fetch market data or prove that a news story caused a price change.
- This is an educational prototype, not a professional fact-checking or investment tool.
