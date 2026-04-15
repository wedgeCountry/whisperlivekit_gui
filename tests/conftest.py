"""Shared pytest fixtures."""
import pytest


@pytest.fixture(autouse=True)
def reset_i18n_language():
    """Restore the i18n language to 'en' after every test that changes it."""
    from transcribe_app import i18n
    yield
    i18n.set_language("en")
