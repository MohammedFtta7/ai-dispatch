#!/usr/bin/env python3
"""
Setup and Installation Script for AI Dispatch Engine
Automatically installs dependencies and verifies setup
"""

import subprocess
import sys
import os
import json
import requests
from pathlib import Path

def print_header():
    """Print setup header"""
    print("🚀 AI Dispatch Engine - Setup Script")
    print("=" * 50)
    print("Setting up your intelligent dispatch system...")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8+ required")
        print(f"   Current version: {sys.version}")
        print("   Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python {sys.version.split()[0]} - Compatible")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Read requirements file
        with open('requirements.txt', 'r') as f:
            requirements = f.read().strip().split('\n')
        
        print(f"   Installing {len(requirements)} packages...")
        
        # Install packages
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ])
        
        print("✅ All dependencies installed successfully")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("   Try running: pip install -r requirements.txt")
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found")
        return False

def verify_imports():
    """Verify all required modules can be imported"""
    print("\n🔍 Verifying imports...")
    
    required_modules = [
        'numpy', 'pandas', 'scipy', 'sklearn', 'deap',
        'requests', 'matplotlib', 'seaborn', 'folium', 'plotly'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import: {', '.join(failed_imports)}")
        print("   Try reinstalling: pip install -r requirements.txt --force-reinstall")
        return False
    
    print("✅ All modules imported successfully")
    return True

def check_osrm_connection():
    """Check if OSRM server is running"""
    print("\n🗺️  Checking OSRM connection...")
    
    osrm_urls = [
        "http://localhost:5000",
        "http://127.0.0.1:5000"
    ]
    
    for url in osrm_urls:
        try:
            # Test with sample Khartoum coordinates
            test_url = f"{url}/route/v1/driving/32.5599,15.5007;32.5342,15.5527"
            response = requests.get(test_url, timeout=5)
            
            if response.status_code == 200:
                print(f"✅ OSRM server connected at {url}")
                return True
                
        except requests.exceptions.RequestException:
            continue
    
    print("⚠️  OSRM server not detected")
    print("   The system will use fallback distance calculations")
    print("   For optimal performance, install OSRM with Khartoum data")
    print("   Expected URL: http://localhost:5000")
    return False

def create_project_structure():
    """Create necessary directories and files"""
    print("\n📁 Creating project structure...")
    
    # Create data directory
    data_dir = Path('data')
    data_dir.mkdir(exist_ok=True)
    
    # Create results directory
    results_dir = Path('results')
    results_dir.mkdir(exist_ok=True)
    
    # Create logs directory
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    print("✅ Project directories created")
    return True

def verify_sample_data():
    """Check if sample data files exist and are valid"""
    print("\n📊 Verifying sample data...")
    
    # Check shipments file
    shipments_file = Path('data/sample_shipments.json')
    if shipments_file.exists():
        try:
            with open(shipments_file, 'r') as f:
                shipments = json.load(f)
            print(f"   ✅ Sample shipments: {len(shipments)} items")
        except json.JSONDecodeError:
            print("   ❌ Invalid shipments JSON format")
            return False
    else:
        print("   ❌ sample_shipments.json not found")
        return False
    
    # Check drivers file
    drivers_file = Path('data/sample_drivers.json')
    if drivers_file.exists():
        try:
            with open(drivers_file, 'r') as f:
                drivers = json.load(f)
            print(f"   ✅ Sample drivers: {len(drivers)} items")
        except json.JSONDecodeError:
            print("   ❌ Invalid drivers JSON format")
            return False
    else:
        print("   ❌ sample_drivers.json not found")
        return False
    
    return True

def run_quick_test():
    """Run a quick test of the system"""
    print("\n🧪 Running quick test...")
    
    try:
        # Import main engine
        from ai_dispatch_engine import AIDispatchEngine
        
        # Initialize with sample data
        engine = AIDispatchEngine()
        
        if engine.load_data('data/sample_shipments.json', 'data/sample_drivers.json'):
            print("   ✅ Data loading test passed")
        else:
            print("   ❌ Data loading test failed")
            return False
        
        # Test distance calculation (quick test)
        engine.calculate_distance_matrix()
        print("   ✅ Distance calculation test passed")
        
        # Test genetic optimizer setup
        from genetic_optimizer import GeneticOptimizer
        optimizer = GeneticOptimizer()
        print("   ✅ AI optimizer test passed")
        
        print("✅ All tests passed - System ready!")
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Test error: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n" + "=" * 50)
    print("🎉 SETUP COMPLETE!")
    print("=" * 50)
    print()
    print("Your AI Dispatch Engine is ready to use!")
    print()
    print("🚀 Quick Start:")
    print("   python ai_dispatch_engine.py")
    print()
    print("📊 Expected Output:")
    print("   - Loads 20 sample shipments and 5 drivers")
    print("   - Optimizes assignments using AI")
    print("   - Creates interactive map and dashboard")
    print("   - Shows performance improvements")
    print()
    print("📁 Generated Files:")
    print("   - ai_dispatch_map.html (Interactive map)")
    print("   - ai_dispatch_dashboard.png (Performance charts)")
    print("   - dispatch_result_*.json (Assignment results)")
    print()
    print("🛠️  Next Steps:")
    print("   1. Replace sample data with your real shipments")
    print("   2. Adjust AI parameters in genetic_optimizer.py")
    print("   3. Set up OSRM server for optimal routing")
    print("   4. Scale up to larger datasets")
    print()
    print("📖 Documentation: README.md")
    print("🐛 Issues? Check troubleshooting in README.md")
    print()
    print("=" * 50)

def main():
    """Main setup function"""
    print_header()
    
    # Check system requirements
    if not check_python_version():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Verify imports
    if not verify_imports():
        return False
    
    # Check OSRM (optional)
    check_osrm_connection()
    
    # Create project structure
    if not create_project_structure():
        return False
    
    # Verify sample data
    if not verify_sample_data():
        return False
    
    # Run quick test
    if not run_quick_test():
        return False
    
    # Print usage instructions
    print_usage_instructions()
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)