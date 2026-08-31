"""Get each result off the machine, and get earlier results back onto it.

molab keeps only files that arrived through its file browser, so whatever a run writes to
disk is deleted when the session shuts down: twelve GPU hours of finished runs disappeared
that way once. Every finished run is therefore uploaded the moment it exists, and a new
session pulls back what previous sessions finished, so the sweep makes progress across
sessions instead of starting over.

The token is read from the environment, never from a file in this repository, because this
repository is public. In molab it goes in the secrets pane as HF_TOKEN.

Everything here is free-tier: a result is tens of kilobytes, uploads are ordinary commits
to a dataset repository, and nothing is downloaded that is already on disk.
"""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

HF_TOKEN_NAMES = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_TOKEN")
GH_TOKEN_NAMES = ("GITHUB_TOKEN", "GH_TOKEN", "RESULTS_TOKEN")
GITHUB_API = "https://api.github.com"


def _from_env(names) -> Optional[str]:
    for name in names:
        value = os.environ.get(name)
        if value and value.strip():
            return value.strip()
    return None


class _Store:
    """What a caller may rely on, whichever backend is behind it."""

    enabled = False
    kind = "none"

    def put(self, local, name: Optional[str] = None) -> bool:
        return False

    def sync_down(self, dest) -> List[str]:
        return []

    def drop(self, name: str) -> bool:
        return False

    def listing(self) -> List[str]:
        return []

    def tidy(self) -> List[str]:
        """Drop the partial curve and checkpoint of every run that has a final record."""
        names = set(self.listing())
        removed = []
        for name in sorted(names):
            if not name.endswith(".json") or name.endswith(".partial.json"):
                continue
            tag = name[:-len(".json")]
            for scratch in (f"{tag}.partial.json", f"{tag}.state.pt"):
                if scratch in names and self.drop(scratch):
                    removed.append(scratch)
        return removed

    def url(self) -> str:
        return ""

    def describe(self) -> str:
        return "nothing is being kept"


class NoStore(_Store):
    """No token, so say so once and let the sweep carry on."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._warned = False

    def put(self, local, name: Optional[str] = None) -> bool:
        if not self._warned and self.verbose:
            print("No token found, so results stay on this machine and will be lost when "
                  "the session ends. Add HF_TOKEN in the secrets pane to keep them.",
                  flush=True)
            self._warned = True
        return False

    def describe(self) -> str:
        return ("no token: results will be lost when this session ends, so download the "
                "zip before it does")


class HfStore(_Store):
    """A dataset repository as the result store."""

    kind = "hf"

    def __init__(self, repo: str = "mudassiruddin7/scaleup", prefix: str = "scaleup",
                 token: Optional[str] = None, verbose: bool = True, private: bool = False):
        self.repo = repo
        self.prefix = prefix.strip("/")
        self.token = token if token is not None else _from_env(HF_TOKEN_NAMES)
        self.verbose = verbose
        self.private = private
        self.enabled = bool(self.token)
        self._api = None
        self._ready = False
        self._warned = False

    def _hub(self):
        if self._api is None:
            from huggingface_hub import HfApi            # present in molab
            self._api = HfApi(token=self.token)
        return self._api

    def _ensure(self) -> None:
        if self._ready:
            return
        from huggingface_hub import create_repo
        create_repo(self.repo, repo_type="dataset", exist_ok=True, private=self.private,
                    token=self.token)
        self._ready = True

    def _warn_once(self, message: str) -> None:
        if not self._warned and self.verbose:
            print(message, flush=True)
            self._warned = True

    def whoami(self) -> Optional[str]:
        """The account the token belongs to, for a connection check that prints no secret."""
        try:
            return self._hub().whoami().get("name")
        except Exception as exc:                          # noqa: BLE001 - report, never raise
            self._warn_once(f"could not reach Hugging Face ({type(exc).__name__})")
            return None

    def put(self, local, name: Optional[str] = None) -> bool:
        local = Path(local)
        if not self.enabled or not local.exists():
            return False
        remote = f"{self.prefix}/{name or local.name}".strip("/")
        try:
            self._ensure()
            self._hub().upload_file(path_or_fileobj=str(local), path_in_repo=remote,
                                    repo_id=self.repo, repo_type="dataset",
                                    commit_message=f"result: {remote}")
            if self.verbose:
                print(f"    kept {remote} ({local.stat().st_size / 1e3:.0f} KB)", flush=True)
            return True
        except Exception as exc:                          # noqa: BLE001
            self._warn_once(f"could not upload {remote} ({type(exc).__name__}); results "
                            f"stay on this machine only")
            return False

    def sync_down(self, dest) -> List[str]:
        """Bring back the result files earlier sessions finished. Skips what is here."""
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if not self.enabled:
            return []
        try:
            from huggingface_hub import hf_hub_download
            files = self._hub().list_repo_files(self.repo, repo_type="dataset")
        except Exception as exc:                          # noqa: BLE001
            self._warn_once(f"could not list {self.repo} ({type(exc).__name__}); starting "
                            f"from what is on disk")
            return []
        restored = []
        for remote in files:
            if not remote.startswith(f"{self.prefix}/") or not remote.endswith(".json"):
                continue
            name = remote.split("/")[-1]
            if (dest / name).exists():
                continue
            try:
                got = hf_hub_download(self.repo, remote, repo_type="dataset",
                                      token=self.token)
                (dest / name).write_bytes(Path(got).read_bytes())
                restored.append(name)
            except Exception as exc:                      # noqa: BLE001
                self._warn_once(f"could not fetch {remote} ({type(exc).__name__})")
        if restored and self.verbose:
            print(f"restored {len(restored)} earlier runs from {self.repo}", flush=True)
        return restored

    def get(self, name: str, dest) -> Optional[Path]:
        """Fetch one stored file by name, or None if it is not there."""
        dest = Path(dest)
        if not self.enabled:
            return None
        try:
            from huggingface_hub import hf_hub_download
            got = hf_hub_download(self.repo, f"{self.prefix}/{name}", repo_type="dataset",
                                  token=self.token)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(Path(got).read_bytes())
            return dest
        except Exception:                                 # noqa: BLE001 - absent is normal
            return None

    def drop(self, name: str) -> bool:
        """Remove one stored file, for the scratch a finished run no longer needs."""
        if not self.enabled:
            return False
        try:
            self._hub().delete_file(f"{self.prefix}/{name}", repo_id=self.repo,
                                    repo_type="dataset",
                                    commit_message=f"tidy: {name}")
            return True
        except Exception:                                 # noqa: BLE001 - absent is fine
            return False

    def listing(self) -> List[str]:
        if not self.enabled:
            return []
        try:
            return [f.split("/")[-1] for f in self._hub().list_repo_files(
                self.repo, repo_type="dataset") if f.startswith(f"{self.prefix}/")]
        except Exception:                                 # noqa: BLE001
            return []

    def url(self) -> str:
        return f"https://huggingface.co/datasets/{self.repo}/tree/main/{self.prefix}"

    def describe(self) -> str:
        return f"results go to {self.url()} as each run finishes"


class GithubStore(_Store):
    """A branch of the code repository, for when only a GitHub token is around."""

    kind = "github"

    def __init__(self, repo: str = "Mudassiruddin7/Attention-Sink", branch: str = "results",
                 prefix: str = "scaleup", token: Optional[str] = None, verbose: bool = True):
        self.repo = repo
        self.branch = branch
        self.prefix = prefix.strip("/")
        self.token = token if token is not None else _from_env(GH_TOKEN_NAMES)
        self.verbose = verbose
        self.enabled = bool(self.token)
        self._branch_ready = False
        self._warned = False

    def _call(self, method: str, path: str, payload: Optional[Dict] = None) -> Dict:
        request = urllib.request.Request(
            f"{GITHUB_API}{path}", method=method,
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
        if not self._warned and self.verbose:
            print(message, flush=True)
            self._warned = True

    def _ensure_branch(self) -> None:
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

    def put(self, local, name: Optional[str] = None) -> bool:
        local = Path(local)
        if not self.enabled or not local.exists():
            return False
        remote = f"{self.prefix}/{name or local.name}".strip("/")
        try:
            self._ensure_branch()
            payload = {"message": f"result: {remote}", "branch": self.branch,
                       "content": base64.b64encode(local.read_bytes()).decode()}
            try:
                existing = self._call(
                    "GET", f"/repos/{self.repo}/contents/{remote}?ref={self.branch}")
                if isinstance(existing, dict) and existing.get("sha"):
                    payload["sha"] = existing["sha"]
            except urllib.error.HTTPError as exc:
                if exc.code != 404:
                    raise
            self._call("PUT", f"/repos/{self.repo}/contents/{remote}", payload)
            if self.verbose:
                print(f"    kept {remote} ({local.stat().st_size / 1e3:.0f} KB)", flush=True)
            return True
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, KeyError) as exc:
            self._warn_once(f"could not upload {remote} "
                            f"({getattr(exc, 'code', type(exc).__name__)})")
            return False

    def sync_down(self, dest) -> List[str]:
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        if not self.enabled:
            return []
        try:
            listing = self._call(
                "GET", f"/repos/{self.repo}/contents/{self.prefix}?ref={self.branch}")
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            return []
        restored = []
        for entry in listing if isinstance(listing, list) else []:
            name = entry.get("name", "")
            if not name.endswith(".json") or (dest / name).exists():
                continue
            try:
                blob = self._call("GET", f"/repos/{self.repo}/git/blobs/{entry['sha']}")
                (dest / name).write_bytes(base64.b64decode(blob["content"]))
                restored.append(name)
            except (urllib.error.HTTPError, urllib.error.URLError, OSError, KeyError):
                continue
        if restored and self.verbose:
            print(f"restored {len(restored)} earlier runs from {self.repo}", flush=True)
        return restored

    def get(self, name: str, dest) -> Optional[Path]:
        dest = Path(dest)
        if not self.enabled:
            return None
        try:
            entry = self._call(
                "GET", f"/repos/{self.repo}/contents/{self.prefix}/{name}?ref={self.branch}")
            blob = self._call("GET", f"/repos/{self.repo}/git/blobs/{entry['sha']}")
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(base64.b64decode(blob["content"]))
            return dest
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, KeyError):
            return None

    def drop(self, name: str) -> bool:
        if not self.enabled:
            return False
        try:
            entry = self._call(
                "GET", f"/repos/{self.repo}/contents/{self.prefix}/{name}?ref={self.branch}")
            self._call("DELETE", f"/repos/{self.repo}/contents/{self.prefix}/{name}",
                       {"message": f"tidy: {name}", "sha": entry["sha"],
                        "branch": self.branch})
            return True
        except (urllib.error.HTTPError, urllib.error.URLError, OSError, KeyError):
            return False

    def listing(self) -> List[str]:
        if not self.enabled:
            return []
        try:
            rows = self._call(
                "GET", f"/repos/{self.repo}/contents/{self.prefix}?ref={self.branch}")
            return [r.get("name", "") for r in rows] if isinstance(rows, list) else []
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            return []

    def url(self) -> str:
        return f"https://github.com/{self.repo}/tree/{self.branch}/{self.prefix}"

    def describe(self) -> str:
        return f"results go to {self.url()} as each run finishes"


class MirrorStore(_Store):
    """Every backend that has a token, so a result exists in more than one place."""

    kind = "mirror"

    def __init__(self, stores):
        self.stores = [s for s in stores if s.enabled]
        self.enabled = bool(self.stores)

    def put(self, local, name: Optional[str] = None) -> bool:
        landed = [s.put(local, name) for s in self.stores]      # every one, not the first
        return any(landed)

    def sync_down(self, dest) -> List[str]:
        seen: List[str] = []
        for store in self.stores:
            for name in store.sync_down(dest):
                if name not in seen:
                    seen.append(name)
        return seen

    def drop(self, name: str) -> bool:
        return any([s.drop(name) for s in self.stores])

    def listing(self) -> List[str]:
        out: List[str] = []
        for store in self.stores:
            out.extend(n for n in store.listing() if n not in out)
        return out

    def tidy(self) -> List[str]:
        out: List[str] = []
        for store in self.stores:
            out.extend(n for n in store.tidy() if n not in out)
        return out

    def get(self, name: str, dest):
        for store in self.stores:
            got = store.get(name, dest) if hasattr(store, "get") else None
            if got is not None:
                return got
        return None

    def url(self) -> str:
        return " and ".join(s.url() for s in self.stores)

    def describe(self) -> str:
        if len(self.stores) > 1:
            return "results go to " + " and ".join(s.url() for s in self.stores) +                    ", each as its run finishes"
        return self.stores[0].describe() if self.stores else "nothing is being kept"


def make_store(kind: str = "all", hf_repo: str = "mudassiruddin7/scaleup",
               gh_repo: str = "Mudassiruddin7/Attention-Sink", prefix: str = "scaleup",
               verbose: bool = True, **extra) -> _Store:
    """Every backend with a token behind it, so a result is never in only one place."""
    built = []
    if kind in ("hf", "auto", "all") and _from_env(HF_TOKEN_NAMES):
        built.append(HfStore(repo=hf_repo, prefix=prefix, verbose=verbose))
    if kind in ("github", "auto", "all") and _from_env(GH_TOKEN_NAMES):
        built.append(GithubStore(repo=gh_repo, prefix=prefix, verbose=verbose))
    if not built:
        return NoStore(verbose=verbose)
    if kind == "auto" or len(built) == 1:
        return built[0]
    return MirrorStore(built)


# the runner used this name before there was more than one backend
ResultStore = make_store
