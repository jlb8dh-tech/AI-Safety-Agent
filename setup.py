#!/usr/bin/env python3
"""
Setup script for Black Unicorn Agent

This script helps set up the development environment and run initial tests.
"""

import subprocess
import sys
import os
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True


def install_dependencies():
    """Install required dependencies"""
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    directories = ['logs', 'data', 'exports', 'examples_output']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def setup_config():
    """Set up configuration files"""
    print("⚙️ Setting up configuration...")
    
    config_example = Path("agent/config/config.example.yaml")
    config_file = Path("agent/config/config.yaml")
    
    if config_example.exists() and not config_file.exists():
        import shutil
        shutil.copy(config_example, config_file)
        print("✅ Configuration file created")
    elif config_file.exists():
        print("✅ Configuration file already exists")
    else:
        print("⚠️ Configuration template not found")
    
    return True


def run_tests():
    """Run the test suite"""
    return run_command("python -m pytest tests/ -v", "Running tests")


def run_examples():
    """Run example scripts"""
    return run_command("python examples.py", "Running examples")


def main():
    """Main setup function"""
    print("🚀 Setting up Black Unicorn Agent...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        print("❌ Failed to create directories")
        sys.exit(1)
    
    # Setup configuration
    if not setup_config():
        print("❌ Failed to setup configuration")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Review configuration: agent/config/config.yaml")
    print("2. Run examples: python examples.py")
    print("3. Run tests: python -m pytest tests/ -v")
    print("4. Start the agent: python main.py --help")
    
    # Ask if user wants to run tests
    try:
        response = input("\nWould you like to run tests now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("\n" + "=" * 50)
            run_tests()
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")
    
    # Ask if user wants to run examples
    try:
        response = input("\nWould you like to run examples now? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            print("\n" + "=" * 50)
            run_examples()
    except KeyboardInterrupt:
        print("\nSetup interrupted by user")


if __name__ == "__main__":
    main()
