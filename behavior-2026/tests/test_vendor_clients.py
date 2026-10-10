"""The vendored evaluator clients must stay byte-identical to upstream (only the b1k26 header is added)."""

from __future__ import annotations

import hashlib
import pathlib

import pytest

import vendor

REF = pathlib.Path("/tmp/claude-0/-home-user-VihanAggarwal/f9b589dc-68ad-5efe-a354-8d0443bae202/scratchpad/ref")
UPSTREAM = {
    "og_post2_network_utils.py": REF / "og_post2" / "utils" / "network_utils.py",
    "og_2026eval_network_utils.py": REF / "og_2026eval" / "utils" / "network_utils.py",
    "og_2026eval_policies.py": REF / "og_2026eval" / "policies.py",
}


@pytest.mark.parametrize("name", vendor.VENDORED)
def test_vendored_body_matches_declared_hash(name: str) -> None:
    body = vendor.upstream_body(name)
    assert hashlib.sha256(body).hexdigest() == vendor.declared_sha256(name)


@pytest.mark.parametrize("name", vendor.VENDORED)
def test_vendored_body_matches_reference_copy(name: str) -> None:
    ref = UPSTREAM[name]
    if not ref.exists():
        pytest.skip(f"reference copy {ref} not available")
    assert vendor.upstream_body(name) == ref.read_bytes()


def test_vendored_modules_import() -> None:
    pytest.importorskip("torch")
    post2 = vendor.load_post2()
    nu, pol = vendor.load_2026eval()
    assert post2.ACTION_CHUNK_REQUEST_KEY == "__action_chunk_size__"
    assert nu.MAX_RECONNECTS_PER_ROLLOUT == 3 and nu.POLICY_RESPONSE_TIMEOUT == 600
    assert pol.WebsocketClientPolicy is nu.WebsocketClientPolicy
    assert hasattr(pol, "MultiWebsocketPolicy")
