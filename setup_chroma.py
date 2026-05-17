"""
Setup script for Chroma DB integration.
Run this to ensure all dependencies are installed.
"""

import subprocess
import sys


def install_dependencies():
    """Install required packages."""
    packages = [
        "chromadb",
        "langchain-chroma",
    ]
    
    print("Installing additional dependencies for Chroma DB support...")
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
            return False
    
    return True


def verify_imports():
    """Verify all required modules can be imported."""
    modules = [
        ("chromadb", "Chroma DB"),
        ("langchain_chroma", "LangChain Chroma"),
        ("faiss", "FAISS"),
    ]
    
    print("\nVerifying imports...")
    all_ok = True
    for module, name in modules:
        try:
            __import__(module)
            print(f"✓ {name} available")
        except ImportError:
            print(f"✗ {name} NOT available")
            all_ok = False
    
    return all_ok


def create_directories():
    """Create necessary directories."""
    import os
    
    dirs = ["embeddings", "embeddings/chroma_db", "embeddings/faiss_index", "data"]
    
    print("\nCreating directories...")
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        print(f"✓ {d}")


if __name__ == "__main__":
    print("=" * 50)
    print("Chroma DB Setup Script")
    print("=" * 50)
    
    # Install dependencies
    if not install_dependencies():
        print("\n✗ Some packages failed to install")
        sys.exit(1)
    
    # Verify imports
    if not verify_imports():
        print("\n⚠ Warning: Some modules are missing")
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 50)
    print("✓ Setup complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Run: streamlit run app.py")
    print("2. Select vector store (FAISS or Chroma) in sidebar")
    print("3. Upload a PDF and start using the app")
    print("\nRead CHROMA_IMPLEMENTATION.md for more details.")
