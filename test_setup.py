#!/usr/bin/env python3
"""
SmartSupport AI - System Diagnostics
Run this script to check if all components are properly configured
"""

import sys
import subprocess
import requests
import logging

# Disable logging for cleaner output
logging.basicConfig(level=logging.ERROR)

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'


def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{text}{Colors.END}")


def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.END}")


def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.END}")


def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.END}")


def print_info(text):
    """Print info with indentation"""
    print(f"   {text}")


def test_endee():
    """Test if Endee is running"""
    print_header("1. Testing Endee Connection")
    
    try:
        response = requests.get("http://localhost:8080/health", timeout=2)
        
        if response.status_code == 200:
            print_success("Endee is running")
            return True
        else:
            print_error(f"Endee returned status {response.status_code}")
            print_info("Start Endee with:")
            print_info("  cd ~/endee")
            print_info("  export NDD_DATA_DIR=$(pwd)/data")
            print_info("  ./build/ndd-avx2")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to Endee")
        print_info("Endee is not running or not accessible at http://localhost:8080")
        print_info("Start Endee with:")
        print_info("  cd ~/endee")
        print_info("  export NDD_DATA_DIR=$(pwd)/data")
        print_info("  ./build/ndd-avx2")
        return False
    except Exception as e:
        print_error(f"Unexpected error: {str(e)}")
        return False


def test_embedding_model():
    """Test if embedding model works"""
    print_header("2. Testing Embedding Model")
    
    try:
        from backend.embedder import embed_text
        
        vector = embed_text("test ticket")
        
        if len(vector) == 384:
            print_success(f"Embedding model working (dimension: {len(vector)})")
            return True
        else:
            print_error(f"Unexpected vector dimension: {len(vector)}")
            return False
            
    except ImportError:
        print_error("sentence-transformers package not installed")
        print_info("Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print_error(f"Embedding failed: {str(e)}")
        print_info("Try reinstalling: pip install --upgrade sentence-transformers")
        return False


def test_index():
    """Test if index exists and has data"""
    print_header("3. Testing Vector Index")
    
    try:
        from backend.embedder import embed_text
        from backend.endee_client import search, get_index_info
        
        # Try to get index info first
        try:
            info = get_index_info()
            print_info(f"Index info: {info}")
        except:
            pass
        
        # Try a search
        vector = embed_text("payment issue")
        result = search(vector, top_k=1)
        
        matches = result.get("results") or result.get("vectors") or []
        
        if matches:
            print_success(f"Index has data (found {len(matches)} result)")
            
            # Show sample metadata
            if matches[0].get("metadata"):
                meta = matches[0]["metadata"]
                print_info(f"Sample team: {meta.get('team', 'N/A')}")
            
            return True
        else:
            print_error("Index exists but has no data")
            print_info("Run data ingestion:")
            print_info("  python ingest_tickets.py")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to Endee (index check requires Endee)")
        return False
    except Exception as e:
        print_error(f"Index error: {str(e)}")
        print_info("Run data ingestion:")
        print_info("  python ingest_tickets.py")
        return False


def test_ollama():
    """Test if Ollama is running"""
    print_header("4. Testing Ollama")
    
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            timeout=5,
            text=True
        )
        
        if result.returncode == 0:
            print_success("Ollama is running")
            
            # Check if llama3 is available
            if "llama3" in result.stdout:
                print_success("llama3 model is available")
                return True
            else:
                print_warning("llama3 model not found")
                print_info("Pull the model with: ollama pull llama3")
                return False
        else:
            print_error("Ollama command failed")
            print_info(f"Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print_error("Ollama not installed")
        print_info("Install from: https://ollama.com/download")
        return False
    except subprocess.TimeoutExpired:
        print_error("Ollama command timed out")
        return False
    except Exception as e:
        print_error(f"Ollama error: {str(e)}")
        return False


def test_api():
    """Test if FastAPI is running"""
    print_header("5. Testing FastAPI Server")
    
    try:
        response = requests.get("http://127.0.0.1:8000/health", timeout=2)
        
        if response.status_code in [200, 503]:
            data = response.json()
            print_success("API server is running")
            print_info(f"Health status: {data}")
            return True
        else:
            print_error(f"API returned status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_warning("API server not running")
        print_info("Start the server with:")
        print_info("  uvicorn backend.main:app --reload")
        print_info("(This is optional - you can start it after fixing other issues)")
        return False
    except Exception as e:
        print_error(f"API error: {str(e)}")
        return False


def test_csv_file():
    """Check if CSV file exists"""
    print_header("0. Testing CSV File")
    
    import os
    
    csv_path = "data/cleaned_tickets.csv"
    
    if os.path.exists(csv_path):
        size_mb = os.path.getsize(csv_path) / (1024 * 1024)
        print_success(f"CSV file found ({size_mb:.1f} MB)")
        return True
    else:
        print_error("CSV file not found")
        print_info(f"Expected location: {csv_path}")
        print_info("Make sure you have the data directory with cleaned_tickets.csv")
        return False


def main():
    """Run all diagnostic tests"""
    
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}SmartSupport AI - System Diagnostics{Colors.END}")
    print("=" * 60)
    
    results = {}
    
    # Run tests in order
    results["CSV"] = test_csv_file()
    results["Endee"] = test_endee()
    results["Embedding"] = test_embedding_model()
    results["Index"] = test_index()
    results["Ollama"] = test_ollama()
    results["API"] = test_api()
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"{Colors.BOLD}SUMMARY{Colors.END}")
    print("=" * 60)
    
    for component, status in results.items():
        if status:
            print_success(component)
        else:
            print_error(component)
    
    # Overall status
    critical = ["CSV", "Endee", "Embedding", "Index"]
    critical_passed = all(results.get(c, False) for c in critical)
    
    print("\n" + "=" * 60)
    
    if all(results.values()):
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 All systems ready!{Colors.END}")
        print("\nYou can now:")
        print("  1. Start the API: uvicorn backend.main:app --reload")
        print("  2. Test endpoints: http://127.0.0.1:8000/docs")
        return 0
    elif critical_passed:
        print(f"{Colors.YELLOW}{Colors.BOLD}⚠️  Core systems ready, but some optional features unavailable{Colors.END}")
        print("\nYou can proceed, but RAG endpoints won't work without Ollama")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ Critical components missing{Colors.END}")
        print("\nFix the failed components above before proceeding")
        return 1


if __name__ == "__main__":
    sys.exit(main())