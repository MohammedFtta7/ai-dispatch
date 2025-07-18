#!/usr/bin/env python3
"""
Debug S002 Assignment Specifically
Why does S002 keep going to Bahri driver?
"""

import sys
import json
from ai_dispatch_engine import AIDispatchEngine

def debug_s002_assignment():
    """Debug why S002 is assigned to Bahri driver"""
    
    print("🔍 Debugging S002 Assignment...")
    print("=" * 50)
    
    # Initialize engine
    engine = AIDispatchEngine()
    engine.load_data('data/sample_shipments.json', 'data/sample_drivers.json')
    engine.calculate_distance_matrix()
    
    # Setup genetic optimizer
    engine.genetic_optimizer.setup(
        shipments=engine.shipments,
        drivers=engine.drivers,
        distance_matrix=engine.distance_matrix,
        population_size=10,
        generations=1,
        osrm_client=engine.osrm_client
    )
    
    # Test specific assignments for S002
    print("\n1. Testing S002 with different drivers...")
    
    # Assignment 1: S002 with Khartoum driver
    assignment_khartoum = {
        "D001": ["S001", "S002", "S003"],  # S002 with Khartoum
        "D002": ["B001", "B002", "B003"],
        "D003": ["O001", "O002", "O003"]
    }
    
    # Assignment 2: S002 with Bahri driver  
    assignment_bahri = {
        "D001": ["S001", "S003", "S004"],
        "D002": ["B001", "B002", "S002"],  # S002 with Bahri
        "D003": ["O001", "O002", "O003"]
    }
    
    print("\n📊 Fitness Comparison:")
    
    # Calculate fitness for both
    try:
        fitness_khartoum = engine.genetic_optimizer._calculate_fitness(assignment_khartoum)
        fitness_bahri = engine.genetic_optimizer._calculate_fitness(assignment_bahri)
        
        print(f"S002 with Khartoum driver (D001): Fitness = {fitness_khartoum:.2f}")
        print(f"S002 with Bahri driver (D002):    Fitness = {fitness_bahri:.2f}")
        
        if fitness_khartoum < fitness_bahri:
            print("✅ AI should prefer Khartoum assignment (lower fitness = better)")
        else:
            print("❌ AI prefers Bahri assignment - this is the problem!")
            
    except Exception as e:
        print(f"❌ Fitness calculation failed: {e}")
        return
    
    # Test complete route costs specifically
    print("\n2. Testing complete route costs...")
    
    try:
        # Route cost for S002 with Khartoum shipments
        khartoum_route_cost = engine.genetic_optimizer._calculate_complete_driver_route(["S001", "S002", "S003"])
        
        # Route cost for S002 with Bahri shipments  
        bahri_route_cost = engine.genetic_optimizer._calculate_complete_driver_route(["B001", "B002", "S002"])
        
        print(f"S002 route cost with Khartoum: {khartoum_route_cost:.2f}")
        print(f"S002 route cost with Bahri:    {bahri_route_cost:.2f}")
        
        if khartoum_route_cost < bahri_route_cost:
            print("✅ Khartoum route is more efficient")
        else:
            print("❌ Bahri route appears more efficient - check delivery-to-pickup matrix")
            
    except Exception as e:
        print(f"❌ Route cost calculation failed: {e}")
    
    # Test delivery-to-pickup matrix specifically
    print("\n3. Testing delivery-to-pickup matrix for S002...")
    
    if hasattr(engine.genetic_optimizer, 'delivery_to_pickup_matrix'):
        matrix = engine.genetic_optimizer.delivery_to_pickup_matrix
        
        # Check key routes involving S002
        test_routes = [
            ("DEL_B001_TO_PICK_S002", "B001 delivery → S002 pickup (should be expensive bridge)"),
            ("DEL_S001_TO_PICK_S002", "S001 delivery → S002 pickup (should be cheaper)"),
            ("DEL_S002_TO_PICK_B001", "S002 delivery → B001 pickup (return route)")
        ]
        
        for key, description in test_routes:
            if key in matrix:
                distance = matrix[key]
                print(f"   {description}: {distance:.2f} km")
            else:
                print(f"   ❌ Missing: {description}")
                
    else:
        print("   ❌ No delivery-to-pickup matrix found!")
    
    # Test individual shipment costs
    print("\n4. Testing individual shipment costs...")
    
    s002_cost = engine.distance_matrix.get("S002", "unknown")
    print(f"S002 individual cost (pickup→delivery): {s002_cost}")
    
    # Compare S002 coordinates to other shipments
    s002 = next(s for s in engine.shipments if s.id == "S002")
    b001 = next(s for s in engine.shipments if s.id == "B001")
    s001 = next(s for s in engine.shipments if s.id == "S001")
    
    print(f"\n5. Geographic analysis:")
    print(f"S002 pickup: ({s002.pickup_lat:.4f}, {s002.pickup_lon:.4f})")
    print(f"B001 pickup: ({b001.pickup_lat:.4f}, {b001.pickup_lon:.4f})")  
    print(f"S001 pickup: ({s001.pickup_lat:.4f}, {s001.pickup_lon:.4f})")
    
    # Calculate straight-line distances
    from math import radians, cos, sin, asin, sqrt
    
    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return 6371 * c
    
    s002_to_b001 = haversine(s002.pickup_lat, s002.pickup_lon, b001.pickup_lat, b001.pickup_lon)
    s002_to_s001 = haversine(s002.pickup_lat, s002.pickup_lon, s001.pickup_lat, s001.pickup_lon)
    
    print(f"S002 to B001 straight-line: {s002_to_b001:.2f} km")
    print(f"S002 to S001 straight-line: {s002_to_s001:.2f} km")
    
    if s002_to_s001 < s002_to_b001:
        print("✅ S002 is closer to Khartoum area")
    else:
        print("⚠️  S002 is actually closer to Bahri area geographically")

def main():
    """Main debug function"""
    
    print("🚀 S002 Assignment Debugger")
    print("=" * 60)
    
    debug_s002_assignment()
    
    print(f"\n🎯 Debug Complete!")
    print(f"This should reveal why S002 keeps going to the wrong driver.")

if __name__ == "__main__":
    main()