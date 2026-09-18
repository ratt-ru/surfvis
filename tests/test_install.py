def test_import():
    import surfvis

    assert hasattr(surfvis, "__version__")


def test_version_is_string():
    from surfvis import __version__

    assert isinstance(__version__, str)
