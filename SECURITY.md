# Security Policy

## Reporting a vulnerability

Please report vulnerabilities privately with GitHub's **"Report a vulnerability"** button (Security → Advisories) on this repository. Do not open a public issue.

Include steps to reproduce, the affected version, and the impact. We aim to acknowledge reports within 3 business days.

## Supported versions

The project is pre-1.0. Only the latest release receives security fixes.

## Design commitments

- The agent requests only the `drive.readonly` Google scope and makes no write calls to Drive.
- Secrets are read only from environment variables or a secret manager, never from config files.
- Document text is sent to the configured embedding provider and LLM. Choose local embeddings and a self-hosted vector store if your data must not leave your infrastructure.
- Indexed files are treated as untrusted: document text is framed as data in the prompt (tags inside it are neutralised), and answers keep only links to the retrieved Drive files, with images removed.
- Data the agent writes (conversations, activity, feedback, sync state) is readable by its owner only.

## Deploying safely

- **Web app and admin portal:** both serve on localhost only, and refuse any other address unless `UI_PASSWORD` (or `ADMIN_PASSWORD`) is set. Wrong passwords are slowed down and locked out after 10 tries in 15 minutes. For a hosted deployment, also put them behind single sign-on (Streamlit's OIDC login or your reverse proxy).
- **Vector store:** the local Qdrant server listens on 127.0.0.1 without a key. Anywhere it is reachable by others (a shared machine, a server, Qdrant Cloud), set `QDRANT_API_KEY`.
- **Logs:** write service logs to a folder only the service account can read (`chmod 700`). Question text stays out of logs unless `SLACK_LOG_QUESTIONS=true`.
- **Cost:** each Slack user may ask `SLACK_QUESTIONS_PER_HOUR` questions an hour (default 30). Also set a spend limit on your LLM API key.
- **Dependencies:** CI audits the locked dependencies against known vulnerabilities on every change to the lock file and weekly.
