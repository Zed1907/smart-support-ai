#!/usr/bin/env python3
"""
SmartSupport AI - System Diagnostics
Run this script to check if all components are properly configured.

Usage:
    python test_setup.py [--verbose]
"""

import sys
import subprocess
import requests
import logging
import argparse
from pathlib import Path

# ═══════════════════════════════════════════════════════════════
# Terminal Colors
# ═══════════════════════════════════════════════════════════════

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    END = '\033[0m'


# ═══════════════════════════════════════════════════════════════
# Logging Configuration
# ═══════════════════════════════════════════════════════════════

def setup_logging(verbose: bool = False):
    """Configure logging based on verbosity"""
    level = logging.DEBUG if verbose else logging.ERROR
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


# ═══════════════════════════════════════════════════════════════
# Print Utilities
# ═══════════════════════════════════════════════════════════════

def print_header(text):
    """Print formatted section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{'═' * 60}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")
    print(f"{Colors.BLUE}{Colors.BOLD}{'═' * 60}{Colors.END}")


def print_success(text):
    """Print success message with checkmark"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text):
    """Print error message with X mark"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text):
    """Print warning message with warning symbol"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_info(text, indent=1):
    """Print info message with indentation"""
    spaces = "   " * indent
    print(f"{Colors.DIM}{spaces}{text}{Colors.END}")


def print_command(text):
    """Print command in monospace style"""
    print(f"{Colors.CYAN}   $ {text}{Colors.END}")


# ═══════════════════════════════════════════════════════════════
# Test Functions
# ═══════════════════════════════════════════════════════════════

def test_python_version():
    """Check Python version"""
    print_header("0. Python Version Check")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major == 3 and version.minor >= 8:
        print_success(f"Python {version_str} (compatible)")
        return True
    else:
        print_error(f"Python {version_str} (requires 3.8+)")
        print_info("Please upgrade Python to version 3.8 or higher")
        return False


def test_dependencies():
    """Check if required Python packages are installed"""
    print_header("1. Python Dependencies")
    
    required_packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'pydantic': 'pydantic',
        'sentence_transformers': 'sentence-transformers',
        'requests': 'requests',
        'msgpack': 'msgpack',
        'orjson': 'orjson',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tqdm': 'tqdm',
    }
    
    missing = []
    installed = []
    
    for module, package in required_packages.items():
        try:
            __import__(module)
            installed.append(package)
        except ImportError:
            missing.append(package)
    
    if not missing:
        print_success(f"All {len(installed)} required packages installed")
        return True
    else:
        print_error(f"Missing {len(missing)} required package(s):")
        for pkg in missing:
            print_info(f"- {pkg}", indent=2)
        print_info("\nInstall missing packages with:")
        print_command("pip install -r requirements.txt")
        return False


def test_csv_file():
    """Check if CSV data file exists"""
    print_header("2. Data File Check")
    
    csv_path = Path("data/cleaned_tickets.csv")
    
    if csv_path.exists():
        size_mb = csv_path.stat().st_size / (1024 * 1024)
        print_success(f"CSV file found ({size_mb:.2f} MB)")
        
        # Try to read first few lines
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, nrows=5)
            required_cols = ["ticket_id", "description", "team", "resolution"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print_warning(f"CSV missing columns: {', '.join(missing_cols)}")
                return False
            
            print_info(f"Found {len(df.columns)} columns: {', '.join(df.columns)}")
            return True
        except Exception as e:
            print_warning(f"Could not validate CSV structure: {e}")
            return True  # File exists, just can't validate
    else:
        print_error("CSV file not found")
        print_info(f"Expected location: {csv_path.absolute()}")
        print_info("Create the data directory and add cleaned_tickets.csv")
        return False


def test_endee():
    """Test if Endee vector database is running"""
    print_header("3. Endee Vector Database")
    
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        
        if response.status_code == 200:
            print_success("Endee is running on port 8080")
            
            # Try to list indexes
            try:
                from backend.endee_client import list_indexes
                indexes = list_indexes()
                if indexes:
                    print_info(f"Found {len(indexes)} index(es)")
                else:
                    print_info("No indexes created yet (run ingest_tickets.py)")
            except:
                pass
            
            return True
        else:
            print_error(f"Endee returned status {response.status_code}")
            print_info("Check Endee logs for errors")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to Endee at http://localhost:8080")
        print_info("\nStart Endee with these commands:")
        print_command("cd ~/endee")
        print_command("export NDD_DATA_DIR=$(pwd)/data")
        print_command("./build/ndd-avx2")
        print_info("\nOr on macOS with ARM:")
        print_command("./build/ndd-neon")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return False


def test_embedding_model():
    """Test if embedding model loads and works"""
    print_header("4. Embedding Model")
    
    try:
        from backend.embedder import embed_text, get_embedding_dimension
        
        print_info("Loading model (this may take a moment)...")
        vector = embed_text("test ticket description")
        
        expected_dim = get_embedding_dimension()
        
        if len(vector) == expected_dim:
            print_success(f"Embedding model working (all-MiniLM-L6-v2, dim={expected_dim})")
            return True
        else:
            print_error(f"Unexpected vector dimension: {len(vector)} (expected {expected_dim})")
            return False
            
    except ImportError as e:
        print_error("sentence-transformers package not installed")
        print_info("Install with:")
        print_command("pip install sentence-transformers")
        return False
    except Exception as e:
        print_error(f"Embedding failed: {str(e)}")
        print_info("Try reinstalling:")
        print_command("pip install --upgrade sentence-transformers torch")
        return False


def test_vector_index():
    """Test if vector index exists and has data"""
    print_header("5. Vector Index Data")
    
    try:
        from backend.embedder import embed_text
        from backend.endee_client import search, get_index_info, INDEX_NAME
        
        # Get index info
        try:
            info = get_index_info()
            print_info(f"Index '{INDEX_NAME}' exists")
            if isinstance(info, dict):
                for key, value in info.items():
                    print_info(f"  {key}: {value}", indent=2)
        except Exception as e:
            print_warning(f"Could not get index info: {e}")
        
        # Try a search
        vector = embed_text("payment issue with credit card")
        result = search(vector, top_k=1)
        
        matches = result.get("results") or result.get("vectors") or []
        
        if matches:
            print_success(f"Index has data (found {len(matches)} result)")
            
            # Show sample metadata
            if matches and matches[0].get("metadata"):
                meta = matches[0]["metadata"]
                print_info("Sample result:")
                print_info(f"  Team: {meta.get('team', 'N/A')}", indent=2)
                print_info(f"  Score: {matches[0].get('score', 'N/A')}", indent=2)
            
            return True
        else:
            print_error("Index exists but has no data")
            print_info("Run data ingestion:")
            print_command("python ingest_tickets.py")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to Endee (required for index check)")
        return False
    except RuntimeError as e:
        if "does not exist" in str(e):
            print_error("Index 'tickets' does not exist")
            print_info("Run data ingestion to create index:")
            print_command("python ingest_tickets.py")
        else:
            print_error(f"Index error: {e}")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return False


def test_ollama():
    """Test if Ollama is running and llama3 is available"""
    print_header("6. Ollama LLM (Optional)")
    
    try:
        # Check if ollama command exists
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5,
            text=True
        )
        
        if result.returncode == 0:
            print_success("Ollama is running")
            
            # Check for llama3 model
            if "llama3" in result.stdout.lower():
                print_success("llama3 model is available")
                print_info("RAG endpoints will work")
                return True
            else:
                print_warning("llama3 model not found")
                print_info("Available models:")
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    if line.strip():
                        print_info(f"  - {line.strip()}", indent=2)
                print_info("\nPull llama3 with:")
                print_command("ollama pull llama3")
                return False
        else:
            print_error("Ollama command failed")
            print_info(f"Error: {result.stderr.strip()}")
            return False
            
    except FileNotFoundError:
        print_warning("Ollama not installed")
        print_info("Ollama is optional - basic endpoints will still work")
        print_info("Install from: https://ollama.com/download")
        print_info("\nWithout Ollama, these endpoints won't work:")
        print_info("  - /assign-rag", indent=2)
        print_info("  - /resolve-rag", indent=2)
        return False
    except subprocess.TimeoutExpired:
        print_error("Ollama command timed out")
        return False
    except Exception as e:
        print_error(f"Ollama error: {str(e)}")
        return False


def test_api_server():
    """Test if FastAPI server is running"""
    print_header("7. FastAPI Server (Optional)")
    
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
        
        if response.status_code in [200, 503]:
            data = response.json()
            print_success("API server is running on port 8000")
            
            # Show health status
            print_info("Component status:")
            for key, value in data.items():
                status = "✓" if value in [True, "healthy"] else "✗"
                print_info(f"  {status} {key}: {value}", indent=2)
            
            print_info("\nAccess the UI at:")
            print_info("  http://127.0.0.1:8000/ui", indent=2)
            print_info("Or API docs at:")
            print_info("  http://127.0.0.1:8000/docs", indent=2)
            return True
        else:
            print_error(f"API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_warning("API server not running")
        print_info("This is optional - start it when ready with:")
        print_command("uvicorn backend.main:app --reload")
        print_info("\nOr for production:")
        print_command("uvicorn backend.main:app --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print_error(f"API error: {str(e)}")
        return False


def test_file_structure():
    """Verify project file structure"""
    print_header("8. Project Structure")
    
    required_files = {
        "backend/": "Backend module directory",
        "backend/embedder.py": "Embedding module",
        "backend/endee_client.py": "Endee client",
        "backend/main.py": "FastAPI application",
        "static/": "Static files directory",
        "static/index.html": "Web UI",
        "ingest_tickets.py": "Data ingestion script",
        "requirements.txt": "Python dependencies",
    }
    
    missing = []
    found = []
    
    for file_path, description in required_files.items():
        path = Path(file_path)
        if path.exists():
            found.append((file_path, description))
        else:
            missing.append((file_path, description))
    
    if not missing:
        print_success(f"All {len(found)} required files/directories present")
        return True
    else:
        print_error(f"Missing {len(missing)} required file(s):")
        for file_path, description in missing:
            print_info(f"- {file_path} ({description})", indent=2)
        return False


# ═══════════════════════════════════════════════════════════════
# Main Test Runner
# ═══════════════════════════════════════════════════════════════

def main():
    """Run all diagnostic tests and provide summary"""
    
    parser = argparse.ArgumentParser(description='SmartSupport AI System Diagnostics')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Print banner
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}{Colors.CYAN}SmartSupport AI - System Diagnostics{Colors.END}")
    print(f"{Colors.DIM}Checking all components and dependencies{Colors.END}")
    print("=" * 60)
    
    # Define test categories
    tests = [
        ("Python Version", test_python_version, True),
        ("Dependencies", test_dependencies, True),
        ("Data File", test_csv_file, True),
        ("Endee Database", test_endee, True),
        ("Embedding Model", test_embedding_model, True),
        ("Vector Index", test_vector_index, True),
        ("Ollama LLM", test_ollama, False),  # Optional
        ("API Server", test_api_server, False),  # Optional
        ("File Structure", test_file_structure, True),
    ]
    
    results = {}
    
    # Run all tests
    for name, test_func, is_critical in tests:
        try:
            results[name] = {
                "passed": test_func(),
                "critical": is_critical
            }
        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Test interrupted by user{Colors.END}")
            sys.exit(130)
        except Exception as e:
            print_error(f"Test crashed: {e}")
            results[name] = {
                "passed": False,
                "critical": is_critical
            }
    
    # Print summary
    print_header("Summary")
    
    critical_passed = 0
    critical_total = 0
    optional_passed = 0
    optional_total = 0
    
    for name, result in results.items():
        status = result["passed"]
        is_critical = result["critical"]
        
        if is_critical:
            critical_total += 1
            if status:
                critical_passed += 1
                print_success(f"{name} (critical)")
            else:
                print_error(f"{name} (critical)")
        else:
            optional_total += 1
            if status:
                optional_passed += 1
                print_success(f"{name} (optional)")
            else:
                print_warning(f"{name} (optional)")
    
    # Overall status
    print(f"\n{Colors.BOLD}Results:{Colors.END}")
    print(f"  Critical: {critical_passed}/{critical_total} passed")
    print(f"  Optional: {optional_passed}/{optional_total} passed")
    
    print("\n" + "=" * 60)
    
    if critical_passed == critical_total:
        if optional_passed == optional_total:
            print(f"{Colors.GREEN}{Colors.BOLD}🎉 All systems ready!{Colors.END}")
            print("\n✨ Next steps:")
            print_command("uvicorn backend.main:app --reload")
            print_info("Then visit: http://127.0.0.1:8000/ui")
            return 0
        else:
            print(f"{Colors.YELLOW}{Colors.BOLD}✓ Core systems ready!{Colors.END}")
            print(f"\n{Colors.DIM}Some optional features are unavailable{Colors.END}")
            print_info("Basic functionality will work")
            print_info("RAG endpoints require Ollama")
            return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ Critical components missing{Colors.END}")
        print(f"\n{Colors.DIM}Fix the failed components above before proceeding{Colors.END}")
        return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Interrupted by user{Colors.END}")
        sys.exit(130)