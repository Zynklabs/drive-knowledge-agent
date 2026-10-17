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
