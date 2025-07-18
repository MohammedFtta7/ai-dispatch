#!/usr/bin/env python3
"""
Debug AI-OSRM Integration
Check if clustering matrix is being built and used correctly
"""

import sys
import json
from ai_dispatch_engine import AIDispatchEngine

def debug_clustering_matrix():
    """Debug the clustering matrix building process"""
    
    print("🔍 Debugging AI-OSRM Integration...")
    print("=" * 50)
    
    # Initialize engine
    print("1. Initializing AI engine...")
    engine = AIDispatchEngine()
    
    # Load data
    print("2. Loading sample data...")
    if not engine.load_data('data/sample_shipments.json', 'data/sample_drivers.json'):
        print("❌ Failed to load data")
        return False
    
    print(f"   📦 Loaded {len(engine.shipments)} shipments")
    print(f"   🚗 Loaded {len(engine.drivers)} drivers")
    
    # Calculate distance matrix
    print("3. Building individual shipment distance matrix...")
    engine.calculate_distance_matrix()
    
    # Test clustering matrix building
    print("4. Testing clustering matrix integration...")
    
    # Setup genetic optimizer
    engine.genetic_optimizer.setup(
        shipments=engine.shipments,
        drivers=engine.drivers,
        distance_matrix=engine.distance_matrix,
        population_size=10,  # Small for testing
        generations=5
    )
    
    # Pass OSRM client
    engine.genetic_optimizer.osrm_client = engine.osrm_client
    
    # Check if clustering matrix was built
    clustering_matrix = getattr(engine.genetic_optimizer, 'clustering_matrix', None)
    
    if clustering_matrix:
        print("   ✅ Clustering matrix built successfully")
        print(f"   📊 Matrix size: {len(clustering_matrix)} route combinations")
        
        # Test specific routes that should show bridge differences
        test_routes = [
            ("S002", "B001", "Khartoum to Bahri (should show bridge)"),
            ("S001", "S003", "Khartoum to Khartoum (should be direct)"),
            ("B001", "O001", "Bahri to Omdurman (should show double bridge)")
        ]
        
        print("\n   🧪 Testing specific route calculations:")
        
        for route1, route2, description in test_routes:
            key1 = f"{route1}-{route2}"
            key2 = f"{route2}-{route1}"
            
            if key1 in clustering_matrix:
                osrm_distance = clustering_matrix[key1]
            elif key2 in clustering_matrix:
                osrm_distance = clustering_matrix[key2]
            else:
                print(f"   ❌ {description}: Route not found in matrix")
                continue
            
            # Calculate straight-line for comparison
            s1 = next((s for s in engine.shipments if s.id == route1), None)
            s2 = next((s for s in engine.shipments if s.id == route2), None)
            
            if s1 and s2:
                from math import radians, cos, sin, asin, sqrt
                
                lat1, lon1, lat2, lon2 = map(radians, [s1.pickup_lat, s1.pickup_lon, s2.pickup_lat, s2.pickup_lon])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                straight_distance = 6371 * c
                
                ratio = osrm_distance / (straight_distance + 0.1)
                bridge_indicator = "🌉 BRIDGE/COMPLEX" if ratio > 1.4 else "🛣️ DIRECT"
                
                print(f"   📍 {description}:")
                print(f"      OSRM: {osrm_distance:.2f} km")
                print(f"      Straight: {straight_distance:.2f} km")
                print(f"      Ratio: {ratio:.2f} {bridge_indicator}")
            
    else:
        print("   ❌ Clustering matrix NOT built!")
        print("   🔧 Integration problem detected")
        return False
    
    # Test if clustering matrix is used in fitness calculation
    print("\n5. Testing fitness calculation...")
    
    # Create a test assignment
    test_assignment = {
        "D001": ["S001", "S002", "S003"],  # Mix that should trigger bridge penalty
        "D002": ["B001", "B002"],
        "D003": ["O001", "O002"]
    }
    
    try:
        fitness = engine.genetic_optimizer._calculate_fitness(test_assignment)
        print(f"   ✅ Fitness calculation works: {fitness:.2f}")
        
        # Test clustering penalty specifically
        clustering_penalty = engine.genetic_optimizer._calculate_smart_clustering_penalty(test_assignment)
        print(f"   📊 Clustering penalty: {clustering_penalty:.2f}")
        
        if clustering_penalty > 0:
            print("   ✅ Clustering penalty calculated correctly")
        else:
            print("   ⚠️  Clustering penalty is zero - possible issue")
            
    except Exception as e:
        print(f"   ❌ Fitness calculation failed: {e}")
        return False
    
    return True

def test_single_generation():
    """Test a single genetic algorithm generation"""
    
    print("\n6. Testing single AI generation...")
    
    engine = AIDispatchEngine()
    engine.load_data('data/sample_shipments.json', 'data/sample_drivers.json')
    engine.calculate_distance_matrix()
    
    # Run just 1 generation to see what happens
    result = engine.run_optimization(population_size=20, generations=1)
    
    assignment = result['optimal_assignment']
    metrics = result['performance_metrics']
    
    print(f"   📊 Single generation result:")
    print(f"   Total distance: {metrics['total_distance']:.2f} km")
    print(f"   Improvement: {metrics['improvement_over_random']:.2f}%")
    
    # Check S002 assignment
    s002_driver = None
    for driver, shipments in assignment.items():
        if "S002" in shipments:
            s002_driver = driver
            break
    
    if s002_driver:
        driver_shipments = assignment[s002_driver]
        khartoum_count = len([s for s in driver_shipments if s.startswith('S')])
        bahri_count = len([s for s in driver_shipments if s.startswith('B')])
        omdurman_count = len([s for s in driver_shipments if s.startswith('O')])
        
        print(f"   📍 S002 assigned to {s002_driver}:")
        print(f"      Khartoum: {khartoum_count}, Bahri: {bahri_count}, Omdurman: {omdurman_count}")
        
        if bahri_count > 0 and khartoum_count > 0:
            print("   ⚠️  Cross-city assignment detected")
        else:
            print("   ✅ Pure geographic assignment")

def main():
    """Main debug function"""
    
    print("🚀 AI-OSRM Integration Debugger")
    print("=" * 60)
    
    # Step 1: Debug clustering matrix
    if debug_clustering_matrix():
        print("\n✅ Clustering matrix integration: WORKING")
        
        # Step 2: Test actual optimization
        test_single_generation()
        
        print(f"\n🎯 Diagnosis Complete!")
        print(f"   If S002 still goes to Bahri, the issue is in:")
        print(f"   1. Clustering weight too low")
        print(f"   2. Bridge penalties too weak")
        print(f"   3. Other objectives overriding clustering")
        
    else:
        print("\n❌ Clustering matrix integration: BROKEN")
        print(f"   🔧 Need to fix OSRM client passing to genetic optimizer")

if __name__ == "__main__":
    main()