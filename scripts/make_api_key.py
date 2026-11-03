"""Issue an upload access key for one owner.

    python scripts/make_api_key.py alice

Prints the raw key (give it to the user once; it is not stored anywhere) and the `owner:sha256` entry to append to
the comma-separated API_KEYS secret on the server. Only the hash is ever configured on the server.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import secrets


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("owner", help="Owner id: letters, digits, '-' or '_' (max 64)")
    owner = parser.parse_args().owner
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", owner):
        parser.error("owner must match [A-Za-z0-9_-]{1,64}")
    key = "piq_" + secrets.token_urlsafe(32)
    print(f"Access key for {owner} (share privately, shown once):\n  {key}\n")
    print(f"API_KEYS entry:\n  {owner}:{hashlib.sha256(key.encode()).hexdigest()}")


if __name__ == "__main__":
    main()
