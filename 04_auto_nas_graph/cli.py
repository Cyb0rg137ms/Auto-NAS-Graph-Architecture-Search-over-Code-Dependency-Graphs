"""
cli.py
======
Command Line Interface for the Auto-NAS-Graph AutoML Codebase Miner.
Scans codebases, runs architecture searches, and generates PyTorch model definitions.
"""

import sys
import argparse
from pathlib import Path
from nas_graph.parser import CodebaseASTParser
from nas_graph.search import ArchitectureSearchEngine
from nas_graph.evaluator import ModelPerformanceEvaluator

def main():
    parser = argparse.ArgumentParser(description="Auto-NAS-Graph: Codebase AST complexity miner & AutoML architecture generator")
    parser.add_argument("--scan_dir", type=str, default=".", help="Codebase directory to scan")
    parser.add_argument("--max_latency", type=float, default=5.0, help="Maximum latency constraint (ms)")
    parser.add_argument("--max_params", type=float, default=2.5, help="Maximum parameter constraint (millions)")
    parser.add_argument("--input_dim", type=int, default=128, help="Feature input dimension")
    parser.add_argument("--output_code", type=str, default="model_definition.py", help="Output filepath for generated PyTorch class")
    
    args = parser.parse_args()
    
    scan_path = Path(args.scan_dir)
    if not scan_path.exists():
        print(f"Error: Target scan directory '{args.scan_dir}' does not exist.")
        sys.exit(1)
        
    print("==================================================")
    print("      AUTO-NAS-GRAPH: AUTOML CODEBASE SEARCH      ")
    print("==================================================")
    print(f"Scanning target: {scan_path.resolve()}")
    print("Analyzing Abstract Syntax Trees (ASTs)...")
    
    ast_parser = CodebaseASTParser(str(scan_path))
    ast_parser.scan()
    summary = ast_parser.get_summary_metrics()
    
    print("\nCodebase Complexity Summary:")
    print(f"  Total Python Files: {summary['total_files']}")
    print(f"  Total Classes:      {summary['total_classes']}")
    print(f"  Total Functions:    {summary['total_functions']}")
    print(f"  Mean Complexity:    {summary['mean_complexity']:.2f}")
    
    print("\nRunning architecture search under constraints:")
    print(f"  Max Latency Limit:   {args.max_latency} ms")
    print(f"  Max Parameter Limit: {args.max_params} M")
    
    search_engine = ArchitectureSearchEngine(summary)
    search_result = search_engine.search(args.max_latency, args.max_params)
    
    print(f"\nSearch Results (Success):")
    print(f"  Selected Model Depth: {search_result['network_depth']} layers")
    print(f"  Layers configuration: {', '.join(search_result['layers'])}")
    print(f"  Estimated Latency:    {search_result['estimated_latency_ms']:.2f} ms")
    print(f"  Estimated Parameters: {search_result['estimated_params_million']:.2f} million")
    
    print("\nProfiling compiled network...")
    evaluator = ModelPerformanceEvaluator(search_result["layers"], input_dim=args.input_dim)
    profile_results = evaluator.profile()
    
    print(f"  Total Parameter Count: {profile_results['parameter_count']:,} parameters")
    print(f"  Estimated FLOPs:       {profile_results['estimated_flops']:,} operations")
    
    # Generate PyTorch model file
    output_path = Path(args.output_code)
    try:
        model_code = evaluator.generate_pytorch_stub()
        output_path.write_text(model_code, encoding="utf-8")
        print(f"\n[SUCCESS] Generated PyTorch code written to {output_path.resolve()}")
    except Exception as e:
        print(f"\n[WARNING] Could not write model file: {e}")
        
    print("==================================================")

if __name__ == "__main__":
    main()
