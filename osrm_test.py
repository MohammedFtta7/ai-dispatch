#!/usr/bin/env python3
"""
OSRM Connection Test Script
Quick diagnostic for OSRM server status
"""

import requests
import json
import time

def test_osrm_connection():
    """Test OSRM server connectivity and response"""
    
    print("🔍 Testing OSRM Server Connection...")
    print("=" * 50)
    
    # Test different possible OSRM URLs
    urls_to_test = [
        "http://localhost:5000",
        "http://127.0.0.1:5000",
        "http://0.0.0.0:5000"
    ]
    
    # Sample Khartoum coordinates for testing
    test_coords = "32.5599,15.5007;32.5342,15.5527"  # Khartoum area
    
    for base_url in urls_to_test:
        print(f"\n🌐 Testing: {base_url}")
        
        try:
            # Test 1: Basic connection
            print("   Testing basic connection...")
            response = requests.get(f"{base_url}/", timeout=5)
            print(f"   ✅ Server responds: {response.status_code}")
            
        except requests.exceptions.ConnectionError:
            print("   ❌ Connection refused - server not running")
            continue
        except requests.exceptions.Timeout:
            print("   ❌ Connection timeout")
            continue
        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue
        
        try:
            # Test 2: Route request
            route_url = f"{base_url}/route/v1/driving/{test_coords}"
            print(f"   Testing route request...")
            print(f"   URL: {route_url}")
            
            route_response = requests.get(route_url, timeout=10)
            
            if route_response.status_code == 200:
                data = route_response.json()
                
                if data.get('code') == 'Ok' and 'routes' in data:
                    route = data['routes'][0]
                    distance = route['distance'] / 1000  # Convert to km
                    duration = route['duration'] / 60    # Convert to minutes
                    
                    print(f"   ✅ OSRM Working!")
                    print(f"   📍 Test route distance: {distance:.2f} km")
                    print(f"   ⏱️  Test route duration: {duration:.1f} minutes")
                    
                    return True, base_url
                else:
                    print(f"   ⚠️  Invalid response: {data}")
                    
            else:
                print(f"   ❌ HTTP Error: {route_response.status_code}")
                print(f"   Response: {route_response.text[:200]}")
                
        except Exception as e:
            print(f"   ❌ Route request failed: {e}")
    
    print(f"\n❌ OSRM Server Not Found!")
    print(f"   No working OSRM server detected on any tested URL")
    return False, None

def test_khartoum_specific_routes():
    """Test specific Khartoum routes to check geographic coverage"""
    
    print(f"\n🗺️  Testing Khartoum-Specific Routes...")
    print("=" * 50)
    
    # Test routes that should show bridge differences
    test_routes = [
        {
            "name": "Khartoum to Khartoum (same city)",
            "coords": "32.5599,15.5007;32.5400,15.5200"
        },
        {
            "name": "Khartoum to Bahri (bridge crossing)",
            "coords": "32.5599,15.5007;32.5800,15.6200"
        },
        {
            "name": "Khartoum to Omdurman (bridge crossing)",
            "coords": "32.5599,15.5007;32.4800,15.4200"
        },
        {
            "name": "Bahri to Omdurman (double bridge)",
            "coords": "32.5800,15.6200;32.4800,15.4200"
        }
    ]
    
    working_url = None
    
    # Find working OSRM URL first
    for url in ["http://localhost:5000", "http://127.0.0.1:5000"]:
        try:
            test_response = requests.get(f"{url}/", timeout=3)
            if test_response.status_code == 200:
                working_url = url
                break
        except:
            continue
    
    if not working_url:
        print("❌ No OSRM server found for geographic testing")
        return
    
    print(f"Using OSRM server: {working_url}")
    
    for route_test in test_routes:
        try:
            route_url = f"{working_url}/route/v1/driving/{route_test['coords']}"
            response = requests.get(route_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == 'Ok' and 'routes' in data:
                    route = data['routes'][0]
                    distance = route['distance'] / 1000
                    duration = route['duration'] / 60
                    
                    # Calculate straight-line distance for comparison
                    coords = route_test['coords'].split(';')
                    start_lon, start_lat = map(float, coords[0].split(','))
                    end_lon, end_lat = map(float, coords[1].split(','))
                    
                    from math import radians, cos, sin, asin, sqrt
                    
                    # Haversine formula
                    lat1, lon1, lat2, lon2 = map(radians, [start_lat, start_lon, end_lat, end_lon])
                    dlat = lat2 - lat1
                    dlon = lon2 - lon1
                    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                    c = 2 * asin(sqrt(a))
                    straight_distance = 6371 * c
                    
                    ratio = distance / (straight_distance + 0.1)
                    bridge_indicator = "🌉 BRIDGE/COMPLEX" if ratio > 1.4 else "🛣️ DIRECT"
                    
                    print(f"\n📍 {route_test['name']}:")
                    print(f"   OSRM Distance: {distance:.2f} km")
                    print(f"   Straight Line: {straight_distance:.2f} km") 
                    print(f"   Ratio: {ratio:.2f} {bridge_indicator}")
                    print(f"   Duration: {duration:.1f} minutes")
                    
                else:
                    print(f"\n❌ {route_test['name']}: Invalid response")
                    
            else:
                print(f"\n❌ {route_test['name']}: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"\n❌ {route_test['name']}: Error - {e}")

def main():
    """Main test function"""
    print("🚀 OSRM Diagnostic Tool")
    print("Testing OSRM server for AI Dispatch Engine")
    print("=" * 60)
    
    # Test basic connectivity
    is_working, working_url = test_osrm_connection()
    
    if is_working:
        print(f"\n🎉 OSRM Server Status: ✅ WORKING")
        print(f"📍 Server URL: {working_url}")
        
        # Test Khartoum-specific routing
        test_khartoum_specific_routes()
        
        print(f"\n✅ OSRM Integration Status: READY")
        print(f"   The AI Dispatch Engine should work with bridge intelligence!")
        
    else:
        print(f"\n❌ OSRM Server Status: NOT WORKING")
        print(f"\n🔧 Possible Solutions:")
        print(f"   1. Install OSRM with Khartoum data")
        print(f"   2. Start OSRM server: osrm-routed --algorithm mld /path/to/khartoum.osrm")
        print(f"   3. Check firewall/port 5000 access")
        print(f"\n⚠️  AI will use fallback straight-line distances (bridge-blind)")

if __name__ == "__main__":
    main()