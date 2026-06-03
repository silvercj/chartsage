from deps import get_identity_optional


def test_no_headers_is_anonymous_not_error():
    ident = get_identity_optional(authorization=None, x_anon_id=None)
    assert ident.is_authenticated is False
    assert ident.user_id is None


def test_garbage_anon_is_anonymous():
    ident = get_identity_optional(authorization=None, x_anon_id="not-a-uuid")
    assert ident.is_authenticated is False
