"""Get each result off the machine before the machine goes away.

molab keeps only files that arrived through its file browser, so whatever a run writes to
disk is gone when the session shuts down: twelve GPU hours of finished runs disappeared
that way once. Every finished run is therefore pushed to a branch of the repository the
moment it exists, through the GitHub contents API, with no git binary and no extra
packages. Results are small, tens of kilobytes each, so this costs nothing.

Without a token nothing is uploaded and the run says so, once, in plain words.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

API = "https://api.github.com"
TOKEN_NAMES = ("GITHUB_TOKEN", "GH_TOKEN", "RESULTS_TOKEN")


def find_token() -> Optional[str]:
    """A token from the environment, which in molab comes from the secrets pane."""
    for name in TOKEN_NAMES:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


class ResultStore:
    """Uploads files to a branch, creating the branch the first time."""

    def __init__(self, repo: str = "Mudassiruddin7/Attention-Sink", branch: str = "results",
                 prefix: str = "scaleup", token: Optional[str] = None, verbose: bool = True):
        self.repo = repo
        self.branch = branch
        self.prefix = prefix.strip("/")
        self.token = token if token is not None else find_token()
        self.verbose = verbose
        self.enabled = bool(self.token)
        self._branch_ready = False
        self._warned = False

    # -- plumbing ---------------------------------------------------------------
    def _call(self, method: str, path: str, payload: Optional[Dict] = None) -> Dict:
        request = urllib.request.Request(
            f"{API}{path}", method=method,
            data=None if payload is None else json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.token}",
                     "Accept": "application/vnd.github+json",
                     "X-GitHub-Api-Version": "2022-11-28",
                     "Content-Type": "application/json",
                     "User-Agent": "sinkprobe-scaleup"})
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read()
        return json.loads(body) if body else {}

    def _warn_once(self, message: str) -> None:
        if not self._warned:
            print(message, flush=True)
            self._warned = True

    def _ensure_branch(self) -> None:
        """Point the results branch at the default branch's head if it does not exist."""
        if self._branch_ready:
            return
        try:
            self._call("GET", f"/repos/{self.repo}/git/ref/heads/{self.branch}")
            self._branch_ready = True
            return
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        info = self._call("GET", f"/repos/{self.repo}")
        head = self._call("GET", f"/repos/{self.repo}/git/ref/heads/{info['default_branch']}")
        self._call("POST", f"/repos/{self.repo}/git/refs",
                   {"ref": f"refs/heads/{self.branch}", "sha": head["object"]["sha"]})
        self._branch_ready = True
        if self.verbose:
            print(f"created branch {self.branch} in {self.repo}", flush=True)

    # -- the one call a caller needs --------------------------------------------
    def put(self, local: Path, name: Optional[str] = None) -> bool:
        """Upload one file. Returns whether it landed; never raises at the caller."""
        local = Path(local)
        if not self.enabled:
            self._warn_once(
                "No token found, so results stay on this machine and will be lost when the "
                "session ends. Add GITHUB_TOKEN in the secrets pane to keep them.")
            return False
        if not local.exists():
            return False
        remote = f"{self.prefix}/{name or local.name}".strip("/")
        try:
            self._ensure_branch()
            payload = {
                "message": f"result: {remote}",
                "content": base64.b64encode(local.read_bytes()).decode(),
                "branch": self.branch}
            try:                                     # an update needs the current blob sha
                existing = self._call(
                    "GET", f"/repos/{self.repo}/contents/{remote}?ref={self.branch}")
                if isinstance(existing, dict) and existing.get("sha"):
                    payload["sha"] = existing["sha"]
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
            self._call("PUT", f"/repos/{self.repo}/contents/{remote}", payload)
            if self.verbose:
                print(f"    kept {remote} on {self.branch} "
                      f"({local.stat().st_size / 1e3:.0f} KB)", flush=True)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, KeyError) as exc:
            detail = getattr(exc, "code", type(exc).__name__)
            self._warn_once(f"could not upload {remote} ({detail}); results stay on this "
                            f"machine only")
            return False

    def url(self) -> str:
        return f"https://github.com/{self.repo}/tree/{self.branch}/{self.prefix}"
