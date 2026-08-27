"""
parser.py
=========
Crawl python source code files, parse Abstract Syntax Trees (ASTs), 
and build a semantic dependency graph of the codebase.
"""

import os
import ast
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any

class CodebaseASTParser:
    """Parses Python codebases using AST to map dependencies and class structures."""
    
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.graph = defaultdict(set)
        self.metrics = {}

    def scan(self) -> Dict[str, Set[str]]:
        """Scans the directory for Python files and maps call structures."""
        for path in self.root.rglob("*.py"):
            if ".venv" in path.parts or "node_modules" in path.parts:
                continue
            self.parse_file(path)
        return self.graph

    def parse_file(self, filepath: Path):
        """Parses a single file, extracting classes, functions, and import structures."""
        try:
            content = filepath.read_text(encoding="utf-8", errors="ignore")
            tree = ast.parse(content)
            
            file_key = filepath.name
            self.metrics[file_key] = {
                "classes": 0,
                "functions": 0,
                "complexity_score": 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    self.metrics[file_key]["classes"] += 1
                    # Map inheritance
                    for base in node.bases:
                        if isinstance(base, ast.Name):
                            self.graph[node.name].add(base.id)
                            
                elif isinstance(node, ast.FunctionDef):
                    self.metrics[file_key]["functions"] += 1
                    self.metrics[file_key]["complexity_score"] += len(node.body)
                    
                    # Track function calls inside the body
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                            self.graph[node.name].add(child.func.id)
                            
        except Exception:
            # Handle parsing errors silently
            pass

    def get_summary_metrics(self) -> Dict[str, Any]:
        """Aggregates complexity metrics across the indexed codebase."""
        total_classes = sum(m["classes"] for m in self.metrics.values())
        total_functions = sum(m["functions"] for m in self.metrics.values())
        total_complexity = sum(m["complexity_score"] for m in self.metrics.values())
        
        return {
            "total_files": len(self.metrics),
            "total_classes": total_classes,
            "total_functions": total_functions,
            "mean_complexity": total_complexity / max(1, len(self.metrics))
        }
