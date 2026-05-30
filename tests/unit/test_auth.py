import time
from uuid import UUID, uuid4

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec, rsa

from auth import verify_token


def _keypair():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    priv = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv, pub


def _ec_keypair():
    key = ec.generate_private_key(ec.SECP256R1())
    priv = key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    pub = key.public_key().public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    return priv, pub


def _token(priv, *, sub, aud="authenticated", exp_offset=3600, algorithm="RS256"):
    payload = {"sub": sub, "aud": aud, "exp": int(time.time()) + exp_offset}
    return jwt.encode(payload, priv, algorithm=algorithm)


def test_valid_token_returns_user_uuid():
    priv, pub = _keypair()
    uid = str(uuid4())
    assert verify_token(_token(priv, sub=uid), _public_key=pub) == UUID(uid)


def test_valid_es256_token_returns_user_uuid():
    priv, pub = _ec_keypair()
    uid = str(uuid4())
    tok = _token(priv, sub=uid, algorithm="ES256")
    assert verify_token(tok, _public_key=pub) == UUID(uid)


def test_expired_token_returns_none():
    priv, pub = _keypair()
    assert verify_token(_token(priv, sub=str(uuid4()), exp_offset=-10), _public_key=pub) is None


def test_missing_exp_returns_none():
    priv, pub = _keypair()
    payload = {"sub": str(uuid4()), "aud": "authenticated"}
    tok = jwt.encode(payload, priv, algorithm="RS256")
    assert verify_token(tok, _public_key=pub) is None


def test_wrong_audience_returns_none():
    priv, pub = _keypair()
    assert verify_token(_token(priv, sub=str(uuid4()), aud="anon"), _public_key=pub) is None


def test_wrong_algorithm_hs256_rejected():
    # A token signed with HS256 (symmetric) must be rejected — only ES256/RS256 allowed.
    _, pub = _keypair()
    tok = jwt.encode(
        {"sub": str(uuid4()), "aud": "authenticated", "exp": int(time.time()) + 3600},
        "some-shared-secret",
        algorithm="HS256",
    )
    assert verify_token(tok, _public_key=pub) is None


def test_tampered_token_returns_none():
    priv, pub = _keypair()
    tok = _token(priv, sub=str(uuid4()))
    tampered = tok[:-3] + ("aaa" if not tok.endswith("aaa") else "bbb")
    assert verify_token(tampered, _public_key=pub) is None


def test_wrong_key_returns_none():
    priv, _ = _keypair()
    _, other_pub = _keypair()
    assert verify_token(_token(priv, sub=str(uuid4())), _public_key=other_pub) is None


def test_non_uuid_sub_returns_none():
    priv, pub = _keypair()
    assert verify_token(_token(priv, sub="not-a-uuid"), _public_key=pub) is None
