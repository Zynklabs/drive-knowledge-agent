# Giving the agent read-only access to Google Drive

The agent only ever requests one Google scope, `https://www.googleapis.com/auth/drive.readonly`, and its code makes no write calls. Pick the option that fits you:

| You are… | Use | Effort |
|---|---|---|
| A company with a folder or Shared Drive to expose | [A1. Service account + share the folder](#a1-service-account-share-the-folder-recommended) | 10 min, no admin needed |
| A company that wants the bot to read "as" a Workspace user | [A2. Domain-wide delegation](#a2-domain-wide-delegation-workspace-admin) | Needs a Workspace super-admin |
| Running on Google Cloud (Cloud Run, GKE) | [A3. Keyless (attached service account)](#a3-keyless-on-google-cloud) | No key files |
| An individual or small team | [B. OAuth (sign in with your Google account)](#b-oauth-sign-in-with-your-google-account) | 10 min |

Every option starts with a Google Cloud project that has the **Google Drive API** enabled. When you're done, run:

```bash
drive-agent check
```

---

## Step 0: Create a project and enable the Drive API

1. Open [console.cloud.google.com](https://console.cloud.google.com/) and create or select a project, e.g. `drive-knowledge-agent`.
2. Go to **APIs & Services → Library**, search for **Google Drive API**, and click **Enable**.

---

## A1. Service account + share the folder (recommended)

This is least privilege: the agent sees only what you explicitly share with it.

1. **IAM & Admin → Service Accounts → Create service account.** Name it `drive-agent`. Grant it **no** project roles.
2. Open the service account → **Keys → Add key → Create new key → JSON**. Store the downloaded file somewhere safe. **Never commit it.**
3. Copy the service account's email, e.g. `drive-agent@my-project.iam.gserviceaccount.com`.
4. In Google Drive, right-click the folder → **Share** → paste the email → role **Viewer** → untick *Notify people* → **Share**.
   - **Shared Drive:** open the Shared Drive → **Manage members** → add the email as **Viewer**.
5. Configure the agent:

   ```bash
   GOOGLE_AUTH_MODE=service_account
   GOOGLE_APPLICATION_CREDENTIALS=/secure/path/drive-agent-key.json
   DRIVE_FOLDER_IDS=https://drive.google.com/drive/folders/<FOLDER_ID>
   ```

To revoke access, remove the email from the folder's Share dialog, or delete the key.

> **Sharing blocked?** Some Workspace orgs don't allow sharing with addresses outside the domain. `check` then reports a Workspace policy error. Ask an admin to allow sharing with the service account, or use A2.

## A2. Domain-wide delegation (Workspace admin)

With domain-wide delegation, the agent reads exactly what one Workspace user can read, e.g. a dedicated `knowledge-bot@company.com` that has already been given access to the right folders. You don't need to share each folder with the service account.

1. Do steps 1 and 2 of A1.
2. On the service account's details page, copy its **Unique ID** (the numeric client ID).
3. A **super-admin** opens [admin.google.com](https://admin.google.com) → **Security → Access and data control → API controls → Manage Domain Wide Delegation → Add new**.
4. Paste the client ID and add **only** this scope:

   ```
   https://www.googleapis.com/auth/drive.readonly
   ```

5. Configure the agent:

   ```bash
   GOOGLE_AUTH_MODE=service_account
   GOOGLE_APPLICATION_CREDENTIALS=/secure/path/drive-agent-key.json
   GOOGLE_IMPERSONATE_USER=knowledge-bot@company.com
   DRIVE_FOLDER_IDS=<FOLDER_ID>
   ```

> **Be careful with this key.** Domain-wide delegation lets the key read Drive as *any* user in the domain. Impersonate a dedicated, least-privileged bot user, keep the key in a secrets manager, and rotate it regularly.

## A3. Keyless on Google Cloud

On Cloud Run, GKE or Compute Engine, attach the `drive-agent` service account to the workload instead of downloading a key. Leave `GOOGLE_APPLICATION_CREDENTIALS` unset and the agent uses Application Default Credentials. Then share the folder with the service account's email, as in A1 step 4.

Domain-wide delegation (A2) still needs a key file.

## B. OAuth (sign in with your Google account)

The agent reads what *your* Google account can read, read-only.

1. Do Step 0.
2. **APIs & Services → OAuth consent screen:**
   - Workspace users: choose **Internal**. No Google verification is needed.
   - Personal Gmail: choose **External**, leave it in **Testing**, and add yourself under **Test users**. Up to 100 test users, no verification needed.
3. Add the scope `.../auth/drive.readonly`.
4. **Credentials → Create credentials → OAuth client ID → Desktop app.** Download the JSON.
5. Configure and sign in:

   ```bash
   GOOGLE_AUTH_MODE=oauth
   GOOGLE_OAUTH_CLIENT_FILE=/path/client_secret.json
   DRIVE_FOLDER_IDS=<FOLDER_ID>
   drive-agent check     # opens a browser to sign in once
   ```

The token is cached at `.drive-agent/token.json` with owner-only file permissions (`0600`). Delete that file to sign out. The agent rejects any cached token that carries more than the read-only scope.

### Why you create your own OAuth client

`drive.readonly` is a *restricted* scope. A single shared public OAuth app would need Google's app verification and a third-party security assessment. It would also mean one project holding everyone's Drive tokens. Having each operator create their own client in Internal or Testing mode avoids both, and keeps your data and audit trail in your own Google Cloud project.

---

## Finding the folder ID

Open the folder in Drive. The ID is the last part of the URL:

```
https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOp   →   1AbCdEfGhIjKlMnOp
```

You can paste the full URL; the agent extracts the ID. To index several folders, separate them with commas.

## What `drive-agent check` tells you

```
✓ Auth  service account drive-agent@my-project.iam.gserviceaccount.com
  scope: drive.readonly (read-only)

✓ Folder  Company Handbook (1AbC…) · Shared Drive
  files to index: 124 (docs 80, pdf 30, pptx 14)
  sub-folders: 12 · shortcuts: 3
  skipped: 7 out of scope (spreadsheet / tabular), 2 unsupported type
```

Add `--show-skipped` to list every skipped file and the reason.

| Message | Fix |
|---|---|
| Google Drive API is not enabled | Enable it (Step 0), wait a minute, retry |
| Folder … was not found or is not shared with … | Check the ID; share the folder with the email shown, as Viewer |
| Domain-wide delegation is not authorized | Super-admin completes A2 step 3–4 with the exact client ID and scope |
| Google does not recognise … | The service account was deleted; create a new key |
| … can edit this folder (warning) | Works, but downgrade to Viewer for least privilege |
| Workspace policy blocks this access | Admin allows external sharing for the service account, or use A2 |

Spreadsheets and tabular files (Excel, Google Sheets, CSV) are out of scope and are always skipped.
