#!/usr/bin/env python3

# ============================================================
# Hermes Stack — one-click Hugging Face Space provisioning
#
# Runs inside GitHub Actions.
#
# Steps:
#   1. Validate environment/secrets
#   2. Authenticate with Hugging Face
#   3. Create or reuse Docker Space
#   4. Create or reuse private backup dataset
#   5. Inject Space secrets
#   6. Inject Space variables
#   7. Upload space/ directory
#   8. Wait until Space is RUNNING
# ============================================================

from __future__ import annotations

import os
import sys
import time
import traceback
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print(
        "::error:: huggingface_hub is not installed.",
        flush=True,
    )
    sys.exit(1)


# ============================================================
# Configuration
# ============================================================

SPACE_SECRETS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_ALLOWED_USERS",
    "HERMES_API_KEY",
    "NINEROUTER_API_KEY",
    "OMNI_ROUTER_KEY",
    "OMNI_ADMIN_PASSWORD",
    "ROUTER_INITIAL_PASSWORD",
    "DASHBOARD_USERNAME",
    "DASHBOARD_PASSWORD",
    "HF_TOKEN",
]

SPACE_VARIABLES = [
    "BACKUP_REPO",
    "HERMES_MODEL",
    "HERMES_TIMEZONE",
    "OMNI_ENABLED",
    "HERMES_WEB_BACKEND",
]

REQUIRED = [
    "HF_TOKEN",
    "HF_USERNAME",
    "HF_SPACE_NAME",
]

DRY_RUN = (
    os.environ.get("DRY_RUN", "")
    .strip()
    .lower()
    in ("1", "true", "yes")
)

PRIVATE_SPACE = (
    os.environ.get("PRIVATE_SPACE", "")
    .strip()
    .lower()
    in ("1", "true", "yes")
)

BUILD_TIMEOUT = int(
    os.environ.get("BUILD_TIMEOUT", "2700")
)

POLL_INTERVAL = 30


# ============================================================
# Helpers
# ============================================================

def log(message: str) -> None:
    """Print message immediately."""
    print(message, flush=True)


def github_error(message: str) -> None:
    """Emit GitHub Actions error annotation."""
    print(f"::error::{message}", flush=True)


def github_warning(message: str) -> None:
    """Emit GitHub Actions warning annotation."""
    print(f"::warning::{message}", flush=True)


def env(name: str, default: str = "") -> str:
    """Read and trim environment variable."""
    return (os.environ.get(name) or default).strip()


def write_summary(text: str) -> None:
    """Append Markdown to GitHub Actions job summary."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")

    if not summary_path:
        return

    with open(summary_path, "a", encoding="utf-8") as summary:
        summary.write(text)


def get_space_dir() -> Path:
    """Return repository space/ directory."""
    return (
        Path(__file__).resolve().parent.parent / "space"
    )


def validate_space_dir(space_dir: Path) -> list[str]:
    """Validate space/ and return its files."""
    if not space_dir.exists():
        raise FileNotFoundError(
            f"Space directory does not exist: {space_dir}"
        )

    if not space_dir.is_dir():
        raise NotADirectoryError(
            f"Space path is not a directory: {space_dir}"
        )

    files = sorted(
        path.relative_to(space_dir).as_posix()
        for path in space_dir.rglob("*")
        if path.is_file()
    )

    if not files:
        raise RuntimeError(
            f"Space directory is empty: {space_dir}"
        )

    return files


# ============================================================
# Environment validation
# ============================================================

def check_env() -> tuple[str, str]:
    """Validate required environment variables."""

    missing = [
        key
        for key in REQUIRED
        if not env(key)
    ]

    if missing:
        if DRY_RUN:
            github_warning(
                "Dry-run without "
                f"{', '.join(missing)} — "
                "using placeholders. "
                "No Hugging Face API calls will be made."
            )

            return "dry-user", "dry-space"

        github_error(
            "Missing required env/secrets: "
            f"{', '.join(missing)}"
        )

        github_error(
            "Add HF_TOKEN / HF_USERNAME / HF_SPACE_NAME "
            "to GitHub Repository Secrets."
        )

        sys.exit(1)

    hf_user = env("HF_USERNAME")
    space_name = env("HF_SPACE_NAME")

    if not space_name.replace("-", "").replace("_", "").isalnum():
        github_error(
            "HF_SPACE_NAME may only contain "
            "letters, digits, '-' and '_'."
        )
        sys.exit(1)

    return hf_user, space_name


# ============================================================
# Dry Run
# ============================================================

def run_dry_run(
    repo_id: str,
    backup_repo: str,
) -> int:
    """Validate deployment locally without HF API calls."""

    space_dir = get_space_dir()

    try:
        files = validate_space_dir(space_dir)
    except Exception as exc:
        github_error(str(exc))
        return 1

    log(
        f"[dry-run] space/ contains "
        f"{len(files)} file(s):"
    )

    for file_name in files:
        log(f"  - {file_name}")

    log(
        "[dry-run] Would set secrets: "
        + ", ".join(SPACE_SECRETS)
    )

    log(
        "[dry-run] Would set variables: "
        + ", ".join(SPACE_VARIABLES)
    )

    write_summary(
        "## 🧪 Dry-run passed\n\n"
        f"- Space folder valid (`{len(files)} files`)\n"
        f"- Would deploy to **`{repo_id}`**\n"
        f"- Backup dataset: `{backup_repo}`\n\n"
        "Run again with `dry_run=false` to deploy. 🚀\n"
    )

    log(
        "[dry-run] OK — workflow logic verified "
        "without touching Hugging Face."
    )

    return 0


# ============================================================
# Authentication
# ============================================================

def authenticate(
    api: HfApi,
    expected_user: str,
) -> bool:
    """Validate HF token."""

    try:
        who = api.whoami()

        login = who.get("name", "?")

        log(
            "[auth] Hugging Face token OK — "
            f"logged in as: {login}"
        )

        if login != expected_user:
            github_warning(
                f"HF_USERNAME ('{expected_user}') "
                f"!= token owner ('{login}')."
            )

        return True

    except Exception as exc:
        github_error(
            "HF token invalid or expired: "
            f"{exc}"
        )

        github_error(
            "Create a WRITE token at "
            "https://huggingface.co/settings/tokens "
            "and update the HF_TOKEN secret."
        )

        return False


# ============================================================
# Create / Reuse Space
# ============================================================

def create_space(
    api: HfApi,
    repo_id: str,
) -> str:
    """
    Create or reuse a Hugging Face Docker Space.

    IMPORTANT:
        space_sdk="docker"

    NOT:
        sdk="docker"
    """

    visibility = (
        "private"
        if PRIVATE_SPACE
        else "public"
    )

    log(
        f"[1/5] Creating (or reusing) "
        f"Space {repo_id} "
        f"(space_sdk=docker, "
        f"cpu-basic, {visibility})…"
    )

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # Old / incorrect:
    #     sdk="docker"
    #
    # Correct Hugging Face API:
    #     space_sdk="docker"
    # --------------------------------------------------------

    url = api.create_repo(
        repo_id=repo_id,
        repo_type="space",
        space_sdk="docker",
        private=PRIVATE_SPACE,
        exist_ok=True,
        space_hardware="cpu-basic",
    )

    return str(url)


# ============================================================
# Backup Dataset
# ============================================================

def create_backup_dataset(
    api: HfApi,
    backup_repo: str,
) -> None:
    """Create or reuse private backup dataset."""

    log(
        "[1/5] Creating (or reusing) "
        f"private backup dataset "
        f"{backup_repo}…"
    )

    try:
        api.create_repo(
            repo_id=backup_repo,
            repo_type="dataset",
            private=True,
            exist_ok=True,
        )

        log(
            "      -> backup dataset ready"
        )

    except Exception as exc:
        github_warning(
            "Could not create backup dataset: "
            f"{exc}"
        )


# ============================================================
# Space Secrets
# ============================================================

def inject_secrets(
    api: HfApi,
    repo_id: str,
) -> None:
    """Inject configured Space secrets."""

    log(
        "[2/5] Injecting Space secrets…"
    )

    for key in SPACE_SECRETS:
        value = env(key)

        if not value:
            log(
                f"      - {key}: "
                "(skipped — empty)"
            )
            continue

        api.add_space_secret(
            repo_id=repo_id,
            key=key,
            value=value,
        )

        log(
            f"      - {key}: ✓ set"
        )


# ============================================================
# Space Variables
# ============================================================

def inject_variables(
    api: HfApi,
    repo_id: str,
    backup_repo: str,
) -> None:
    """Inject configured Space variables."""

    log(
        "[3/5] Injecting Space variables…"
    )

    defaults = {
        "BACKUP_REPO": backup_repo,
        "HERMES_TIMEZONE": "Asia/Tehran",
        "OMNI_ENABLED": "true",
        "HERMES_WEB_BACKEND": "tavily",
    }

    for key in SPACE_VARIABLES:
        value = env(
            key,
            defaults.get(key, ""),
        )

        if not value:
            log(
                f"      - {key}: "
                "(skipped — empty)"
            )
            continue

        api.add_space_variable(
            repo_id=repo_id,
            key=key,
            value=value,
        )

        if key == "BACKUP_REPO":
            log(
                f"      - {key}: {value}"
            )
        else:
            log(
                f"      - {key}: ✓ set"
            )


# ============================================================
# Upload Space
# ============================================================

def upload_space(
    api: HfApi,
    repo_id: str,
) -> None:
    """Upload local space/ directory."""

    space_dir = get_space_dir()

    try:
        files = validate_space_dir(space_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Invalid Space directory: {exc}"
        ) from exc

    log(
        "[4/5] Uploading space/ files "
        f"({len(files)} files)…"
    )

    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=repo_id,
        repo_type="space",
        commit_message=(
            "deploy: hermes-stack "
            "via GitHub Actions"
        ),
    )

    log(
        "      -> files uploaded. "
        "Hugging Face build should start."
    )


# ============================================================
# Wait for Space
# ============================================================

def wait_for_space(
    api: HfApi,
    repo_id: str,
    hf_user: str,
    space_name: str,
    backup_repo: str,
) -> int:
    """Wait for build/runtime status."""

    log(
        f"[5/5] Waiting for build "
        f"(timeout {BUILD_TIMEOUT}s)…"
    )

    start = time.time()
    last_stage = ""

    space_url = (
        f"https://huggingface.co/spaces/{repo_id}"
    )

    app_url = (
        f"https://{hf_user}-{space_name}.hf.space"
    )

    error_stages = {
        "BUILD_ERROR",
        "RUN_ERROR",
        "CONFIG_ERROR",
        "NO_APP_FILE",
    }

    while (
        time.time() - start
        < BUILD_TIMEOUT
    ):
        try:
            runtime = api.get_space_runtime(
                repo_id=repo_id
            )

            stage = getattr(
                runtime,
                "stage",
                str(runtime),
            )

            if stage != last_stage:
                elapsed = int(
                    time.time() - start
                )

                log(
                    f"      [{elapsed:>4}s] "
                    f"stage: {stage}"
                )

                last_stage = stage

            if stage == "RUNNING":
                log("")
                log(
                    "🎉 SPACE IS RUNNING!"
                )

                write_summary(
                    "## 🎉 Deployment successful — "
                    "your Space is RUNNING!\n\n"
                    "| Service | URL |\n"
                    "|---|---|\n"
                    f"| 🏠 Space | {space_url} |\n"
                    f"| 🌐 Router dashboard | {app_url} |\n"
                    f"| 🧠 Agent web dashboard | {app_url}/hermes/ |\n"
                    f"| 🔌 Agent API | {app_url}/hermes-api/v1 |\n"
                    f"| 💾 Backups | https://huggingface.co/datasets/{backup_repo} |\n\n"
                    "**Next steps**\n\n"
                    "1. Open the Telegram bot and send `/start`.\n"
                    "2. The Keep Space Awake workflow can ping "
                    "`/healthz` periodically.\n"
                    "3. Change configuration from "
                    "Space → Settings → Variables and secrets, "
                    "then restart the Space.\n"
                )

                return 0

            if stage in error_stages:
                github_error(
                    f"Space entered error state: "
                    f"{stage}"
                )

                github_error(
                    f"Build logs: "
                    f"{space_url}/logs/build"
                )

                write_summary(
                    "## ❌ Space build failed\n\n"
                    f"State: `{stage}`\n\n"
                    f"[Build logs]({space_url}/logs/build)\n"
                )

                return 1

        except Exception as exc:
            log(
                "      runtime poll failed: "
                f"{exc} — retrying"
            )

        time.sleep(POLL_INTERVAL)

    github_warning(
        f"Timed out after "
        f"{BUILD_TIMEOUT}s. "
        "The build may still be running."
    )

    write_summary(
        "## ⏳ Build still in progress\n\n"
        f"Check Space status here: {space_url}\n"
    )

    return 0


# ============================================================
# Main
# ============================================================

def main() -> int:
    """Main deployment function."""

    hf_user, space_name = check_env()

    repo_id = (
        f"{hf_user}/{space_name}"
    )

    backup_repo = env(
        "BACKUP_REPO",
        f"{hf_user}/{space_name}-backup",
    )

    print("=" * 62)

    print(
        f" Target Space  : "
        f"https://huggingface.co/spaces/{repo_id}"
    )

    print(
        f" Backup dataset: "
        f"{backup_repo} (private)"
    )

    print(
        f" Mode          : "
        f"{'DRY-RUN (no HF calls)' if DRY_RUN else 'LIVE'}"
    )

    print(
        f" Visibility    : "
        f"{'PRIVATE' if PRIVATE_SPACE else 'public'}"
    )

    print("=" * 62)

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if DRY_RUN:
        return run_dry_run(
            repo_id=repo_id,
            backup_repo=backup_repo,
        )

    # --------------------------------------------------------
    # Hugging Face API
    # --------------------------------------------------------

    api = HfApi(
        token=env("HF_TOKEN")
    )

    # --------------------------------------------------------
    # Authentication
    # --------------------------------------------------------

    if not authenticate(
        api=api,
        expected_user=hf_user,
    ):
        return 1

    # --------------------------------------------------------
    # Create / Reuse Space
    # --------------------------------------------------------

    try:
        url = create_space(
            api=api,
            repo_id=repo_id,
        )

        log(
            f"      -> {url}"
        )

    except Exception as exc:
        github_error(
            "Failed to create/reuse "
            f"Hugging Face Space: {exc}"
        )

        traceback.print_exc()

        return 1

    # --------------------------------------------------------
    # Backup dataset
    # --------------------------------------------------------

    create_backup_dataset(
        api=api,
        backup_repo=backup_repo,
    )

    # --------------------------------------------------------
    # Secrets
    # --------------------------------------------------------

    try:
        inject_secrets(
            api=api,
            repo_id=repo_id,
        )

    except Exception as exc:
        github_error(
            "Failed while injecting "
            f"Space secrets: {exc}"
        )

        traceback.print_exc()

        return 1

    # --------------------------------------------------------
    # Variables
    # --------------------------------------------------------

    try:
        inject_variables(
            api=api,
            repo_id=repo_id,
            backup_repo=backup_repo,
        )

    except Exception as exc:
        github_error(
            "Failed while injecting "
            f"Space variables: {exc}"
        )

        traceback.print_exc()

        return 1

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    try:
        upload_space(
            api=api,
            repo_id=repo_id,
        )

    except Exception as exc:
        github_error(
            "Failed while uploading "
            f"Space files: {exc}"
        )

        traceback.print_exc()

        return 1

    # --------------------------------------------------------
    # Wait
    # --------------------------------------------------------

    return wait_for_space(
        api=api,
        repo_id=repo_id,
        hf_user=hf_user,
        space_name=space_name,
        backup_repo=backup_repo,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    try:
        sys.exit(main())

    except SystemExit:
        raise

    except Exception:
        traceback.print_exc()

        github_error(
            "Unexpected failure — "
            "see traceback above."
        )

        sys.exit(1)
