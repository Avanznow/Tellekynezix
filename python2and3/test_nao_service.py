#!/usr/bin/env python3
"""
NAO Service Connectivity Tester
Verifies that the NAO Python 2.7 service is running and accessible
"""

import sys
import json
import urllib.request
import urllib.error

# Colors for terminal output
class Colors:
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'

NAO_SERVICE_URL = "http://localhost:5000"

def test_health_endpoint():
    """Test the /health endpoint"""
    print(f"\n{Colors.BLUE}=== Testing Health Endpoint ==={Colors.NC}\n")
    
    try:
        with urllib.request.urlopen(f"{NAO_SERVICE_URL}/health", timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"{Colors.GREEN}✓{Colors.NC} Health endpoint reachable")
                print(f"{Colors.CYAN}  Response:{Colors.NC}")
                print(f"    Status: {data.get('status', 'unknown')}")
                print(f"    Python: {data.get('python_version', 'unknown')}")
                print(f"    NAOqi:  {data.get('naoqi', 'unknown')}")
                return True
            else:
                print(f"{Colors.RED}✗{Colors.NC} Unexpected status code: {response.status}")
                return False
    except urllib.error.URLError as e:
        print(f"{Colors.RED}✗{Colors.NC} Cannot reach NAO service")
        print(f"{Colors.RED}  Error: {e.reason}{Colors.NC}")
        return False
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.NC} Error: {e}")
        return False

def test_api_info_endpoint():
    """Test the /api/info endpoint"""
    print(f"\n{Colors.BLUE}=== Testing API Info Endpoint ==={Colors.NC}\n")
    
    try:
        with urllib.request.urlopen(f"{NAO_SERVICE_URL}/api/info", timeout=5) as response:
            if response.status == 200:
                data = json.loads(response.read().decode())
                print(f"{Colors.GREEN}✓{Colors.NC} API info endpoint reachable")
                print(f"{Colors.CYAN}  Available commands:{Colors.NC}")
                commands = data.get('commands', [])
                for cmd in commands:
                    print(f"    - {cmd}")
                return True
            else:
                print(f"{Colors.YELLOW}⚠{Colors.NC}  Unexpected status code: {response.status}")
                return False
    except urllib.error.URLError as e:
        print(f"{Colors.YELLOW}⚠{Colors.NC}  Cannot reach API info endpoint")
        print(f"  This is OK if endpoint doesn't exist")
        return True  # Not critical
    except Exception as e:
        print(f"{Colors.YELLOW}⚠{Colors.NC}  Error: {e}")
        return True  # Not critical

def test_command_endpoint():
    """Test the /api/command endpoint with a mock command"""
    print(f"\n{Colors.BLUE}=== Testing Command Endpoint ==={Colors.NC}\n")
    
    # Create a test command payload
    test_command = {
        "action": "test",
        "test_mode": True
    }
    
    try:
        data = json.dumps(test_command).encode('utf-8')
        req = urllib.request.Request(
            f"{NAO_SERVICE_URL}/api/command",
            data=data,
            headers={'Content-Type': 'application/json'},
            method='POST'
        )
        
        with urllib.request.urlopen(req, timeout=5) as response:
            response_data = json.loads(response.read().decode())
            print(f"{Colors.GREEN}✓{Colors.NC} Command endpoint is accessible")
            print(f"{Colors.CYAN}  Response: {response_data}{Colors.NC}")
            return True
            
    except urllib.error.HTTPError as e:
        # 400 or 404 might be OK if command format is wrong but endpoint exists
        if e.code in [400, 404]:
            print(f"{Colors.GREEN}✓{Colors.NC} Command endpoint exists (returned {e.code})")
            return True
        else:
            print(f"{Colors.RED}✗{Colors.NC} HTTP error: {e.code}")
            return False
    except urllib.error.URLError as e:
        print(f"{Colors.RED}✗{Colors.NC} Cannot reach command endpoint")
        return False
    except Exception as e:
        print(f"{Colors.RED}✗{Colors.NC} Error: {e}")
        return False

def provide_troubleshooting():
    """Provide troubleshooting guidance"""
    print(f"\n{Colors.YELLOW}=== Troubleshooting ==={Colors.NC}\n")
    print("If NAO service is not reachable:")
    print("")
    print("1. Make sure NAO service is running:")
    print("   ./start_nao_service.sh")
    print("")
    print("2. Check if port 5000 is already in use:")
    print("   lsof -i:5000")
    print("")
    print("3. Check NAO service logs:")
    print("   tail -f nao_service.log")
    print("")
    print("4. Verify Python 2.7 is installed:")
    print("   python2.7 --version")
    print("")
    print("5. Verify Flask is installed for Python 2.7:")
    print("   python2.7 -m pip list | grep -i flask")
    print("")

def main():
    """Main test routine"""
    print(f"{Colors.CYAN}╔═══════════════════════════════════════════════════╗{Colors.NC}")
    print(f"{Colors.CYAN}║                                                   ║{Colors.NC}")
    print(f"{Colors.CYAN}║      NAO Service Connectivity Tester             ║{Colors.NC}")
    print(f"{Colors.CYAN}║                                                   ║{Colors.NC}")
    print(f"{Colors.CYAN}╚═══════════════════════════════════════════════════╝{Colors.NC}")
    
    print(f"\n{Colors.BLUE}Testing NAO service at: {NAO_SERVICE_URL}{Colors.NC}")
    
    # Run tests
    health_ok = test_health_endpoint()
    info_ok = test_api_info_endpoint()
    command_ok = test_command_endpoint()
    
    # Summary
    print(f"\n{Colors.BLUE}=== Test Summary ==={Colors.NC}\n")
    
    if health_ok and command_ok:
        print(f"{Colors.GREEN}✓ NAO service is running and operational{Colors.NC}")
        print(f"{Colors.GREEN}✓ All critical endpoints are accessible{Colors.NC}")
        print("")
        print("NAO service is ready for use with Avatar GUI")
        return 0
    elif health_ok:
        print(f"{Colors.YELLOW}⚠ NAO service is running but some endpoints may have issues{Colors.NC}")
        print("This may still work - try running the full Avatar environment")
        return 0
    else:
        print(f"{Colors.RED}✗ NAO service is not accessible{Colors.NC}")
        provide_troubleshooting()
        return 1

if __name__ == "__main__":
    sys.exit(main())
