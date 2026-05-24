"""Shared pytest fixtures."""
import pytest
from tests.helpers.builders import (
    activities_df,
    sales_df,
    degenerate_df,
    negative_duration_df,
)


@pytest.fixture
def activities():
    return activities_df()


@pytest.fixture
def sales():
    return sales_df()


@pytest.fixture
def degenerate():
    return degenerate_df()


@pytest.fixture
def negative_duration():
    return negative_duration_df()
