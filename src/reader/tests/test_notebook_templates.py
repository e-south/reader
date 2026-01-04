import ast

from reader.core import notebooks


def _is_app_cell(dec: ast.AST) -> bool:
    return (
        isinstance(dec, ast.Attribute)
        and isinstance(dec.value, ast.Name)
        and dec.value.id == "app"
        and dec.attr == "cell"
    )


class _Collector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # pragma: no cover - skip nested scopes
        return

    def visit_ClassDef(self, node: ast.ClassDef) -> None:  # pragma: no cover - skip nested scopes
        return

    def visit_Lambda(self, node: ast.Lambda) -> None:  # pragma: no cover - skip nested scopes
        return

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Store):
            self.names.add(node.id)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name)


def _find_duplicates(template: str) -> set[str]:
    tree = ast.parse(template)
    seen: set[str] = set()
    dupes: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(_is_app_cell(dec) for dec in node.decorator_list):
            collector = _Collector()
            for stmt in node.body:
                collector.visit(stmt)
            for name in collector.names:
                if name.startswith("_"):
                    continue
                if name in seen:
                    dupes.add(name)
                seen.add(name)
    return dupes


def test_notebook_templates_no_duplicate_globals() -> None:
    templates = {
        "notebook/basic": notebooks.EXPERIMENT_EDA_BASIC_TEMPLATE,
        "notebook/microplate": notebooks.EXPERIMENT_EDA_MICROPLATE_TEMPLATE,
        "notebook/cytometry": notebooks.EXPERIMENT_EDA_CYTOMETRY_TEMPLATE,
        "notebook/plot": notebooks.EXPERIMENT_NOTEBOOK_PLOT_TEMPLATE,
    }
    for name, template in templates.items():
        dupes = sorted(_find_duplicates(template))
        assert not dupes, f"{name} defines the same non-private name in multiple cells: {dupes}"
