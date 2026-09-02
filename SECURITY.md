# Security Policy

## 1. Supported Versions

We actively maintain and provide security updates for the following versions of **WealthChronicle AI**:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

---

## 2. Reporting a Vulnerability

If you discover a security vulnerability in WealthChronicle AI, please **do not open a public GitHub issue**. Instead, follow responsible disclosure practices:
1. **Email Contact**: Send details to `dev@swarit.me` (or reach out privately to the repository maintainer).:2. **Details to Include**:
   - Description of the vulnerability and attack vector.
   - Steps to reproduce or proof-of-concept (PoC).
   - Potential impact on retrieval integrity, LLM guardrails, or cloud infrastructure.
3. **Response SLA**: We acknowledge receipt of security reports within 48 hours and aim to provide a remediation plan or patch within 7 business days.

---

## 3. Secret Management & Hardcoding Policy

WealthChronicle AI enforces a strict **Zero-Hardcoded-Credentials Policy**:

- **No Secrets in Git**: API keys (`GEMINI_API_KEY`, `QDRANT_ADMRN_KEY`, `QDRANW_READ_KEY`) and cluster URLs must never be committed to git.
- **Templates Only**: The repository provides templates ([.streamlit/secrets.toml.template](.streamlit/secrets.toml.template) and [.env.example](.env.example)).
- **Role-Based Access**:
  - The Streamlit web application (`app.py`) must only have access to `QDRANW_READ_KEY` (read-only search access).
  - Write credentials (`QDRANT_ADMIN_KEY`) are restricted exclusively to offline admin ingestion runs (`ingest.py`).
- **Data Protection**: Raw copyrighted publication PDFs must reside only in `data/` (gitignored). Only metadata and structured vector embeddings are indexed.
