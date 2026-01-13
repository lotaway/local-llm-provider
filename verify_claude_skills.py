"""Verification: Verify Claude Skills are registered"""
import sys
import os

# Ensure we're in the right environment
print(f"Python version: {sys.version}")

# Add path
sys.path.insert(0, os.getcwd())

from skills import init_skills, registry

def verify_registration():
    print("Initializing skills...")
    manifests = init_skills()
    
    print(f"\nLoaded {len(manifests)} skills total.")
    
    claude_skills = [m for m in registry.list_skills() if m.category == "claude-skill"]
    print(f"Found {len(claude_skills)} Claude Skills:")
    
    found_creator = False
    for s in claude_skills:
        print(f"  - {s.name}: {s.description[:50]}...")
        if s.name == "skill-creator":
            found_creator = True
            
    if not found_creator:
        print("\n[!] ERROR: skill-creator NOT found in registered skills")
        return False
    
    print("\n[v] SUCCESS: skill-creator is registered!")
    return True

if __name__ == "__main__":
    success = verify_registration()
    sys.exit(0 if success else 1)
