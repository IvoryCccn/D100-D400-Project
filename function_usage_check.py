from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple


SRC_DIRS = ["src", "notebooks"]


@dataclass
class FunctionInfo:
    file: str
    name: str
    lineno: int


def iter_py_files(base: Path, folders: List[str]) -> List[Path]:
    files: List[Path] = []
    for d in folders:
        p = base / d
        if p.exists():
            files.extend(p.rglob("*.py"))
    return files


def parse_functions_and_calls(py_path: Path) -> Tuple[List[FunctionInfo], Set[str]]:
    """
    Return:
      - list of functions defined in this file
      - set of function names called in this file (best-effort, by name)
    """
    text = py_path.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(py_path))

    funcs: List[FunctionInfo] = []
    called: Set[str] = set()

    class Visitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node: ast.FunctionDef):
            funcs.append(FunctionInfo(str(py_path), node.name, node.lineno))
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
            funcs.append(FunctionInfo(str(py_path), node.name, node.lineno))
            self.generic_visit(node)

        def visit_Call(self, node: ast.Call):
            if isinstance(node.func, ast.Name):
                called.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                called.add(node.func.attr)
            self.generic_visit(node)

    Visitor().visit(tree)
    return funcs, called


def main():
    base = Path(".").resolve()
    py_files = iter_py_files(base, SRC_DIRS)

    all_functions: List[FunctionInfo] = []
    calls_by_file: Dict[str, Set[str]] = {}

    for f in py_files:
        funcs, called = parse_functions_and_calls(f)
        all_functions.extend(funcs)
        calls_by_file[str(f)] = called

    all_called_names: Set[str] = set()
    for called in calls_by_file.values():
        all_called_names |= called

    # export report
    print("\n" + "=" * 80)
    print("FUNCTION INVENTORY (by file)")
    print("=" * 80)

    funcs_by_file: Dict[str, List[FunctionInfo]] = {}
    for fi in all_functions:
        funcs_by_file.setdefault(fi.file, []).append(fi)

    for file, flist in sorted(funcs_by_file.items()):
        print(f"\n--- {file} ---")
        for fi in sorted(flist, key=lambda x: x.lineno):
            status = "CALLED" if fi.name in all_called_names else "UNUSED?"
            print(f"  line {fi.lineno:>4}  def {fi.name}()   [{status}]")

    # export paths of unuse functions
    unused = [fi for fi in all_functions if fi.name not in all_called_names]

    print("\n" + "=" * 80)
    print("PATH of SUSPECT UNUSED FUNCTIONS")
    print("=" * 80)
    for fi in sorted(unused, key=lambda x: (x.file, x.lineno)):
        print(f"{fi.file}:{fi.lineno}  {fi.name}")

    print("\nNOTE: If a function is called via alias/import rename, reflection, or callbacks, it may show as UNUSED?.")


if __name__ == "__main__":
    main()