# GDPR Guidelines Search App

A Streamlit application for searching and analyzing GDPR guidelines using AI.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env`
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Environment Variables

Required environment variables:
- `OPENAI_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_ANON_KEY`

## Features

- Search through GDPR guidelines
- AI-powered response generation
- Context-aware results
- Export functionality