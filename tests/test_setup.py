from src import __version__ as version


def test_version(fake):
    """Sanity check that there is a version in the right place"""
    assert version == "1.0.0"
    assert fake.pystr() != version
