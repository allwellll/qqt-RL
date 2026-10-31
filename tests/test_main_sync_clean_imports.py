"""Main keeps portable legacy recipes without import-time run reads."""
import importlib
import pytest

@pytest.mark.parametrize("stage", ["06", "07", "08"])
def test_recipe_clean_import_uses_explicit_sources(stage):
    recipe = importlib.import_module(f"scripts.je{stage}_bun_recipe")
    assert isinstance(recipe.SOURCES, list)
    assert recipe.SOURCES
    assert f"scripts/je{stage}_bun_recipe.py" in recipe.SOURCES
