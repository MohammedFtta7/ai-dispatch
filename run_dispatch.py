#!/usr/bin/env python3
"""
Simple Run Script for AI Dispatch Engine
Easy way to run the system with different configurations
"""

import argparse
import sys
import time
from datetime import datetime
import json

def print_banner():
    """Print application banner"""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🚀 AI DISPATCH ENGINE                     ║
    ║                     for Khartoum, Sudan                     ║
    ║                                                              ║
    ║              Intelligent Shipment Assignment                ║
    ║                 Using Genetic Algorithms                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

def run_dispatch(shipments_file, drivers_file, population_size, generations, 
                osrm_url, visualize, save_results):
    """Run the AI dispatch engine with specified parameters"""
    
    try:
        from ai_dispatch_engine import AIDispatchEngine
        
        print(f"🎯 Configuration:")
        print(f"   📦 Shipments: {shipments_file}")
        print(f"   🚗 Drivers: {drivers_file}")
        print(f"   🧬 Population: {population_size}")
        print(f"   🔄 Generations: {generations}")
        print(f"   🗺️  OSRM URL: {osrm_url}")
        print()
        
        # Initialize engine
        print("🚀 Initializing AI Dispatch Engine...")
        engine = AIDispatchEngine(osrm_url=osrm_url)
        
        # Load data
        print(f"📊 Loading data from {shipments_file} and {drivers_file}...")
        if not engine.load_data(shipments_file, drivers_file):
            print("❌ Failed to load data!")
            return False
        
        # Calculate distance matrix
        print("🗺️  Calculating real-world distances using OSRM...")
        start_time = time.time()
        engine.calculate_distance_matrix()
        matrix_time = time.time() - start_time
        print(f"   ✅ Distance matrix calculated in {matrix_time:.1f} seconds")
        
        # Run AI optimization
        print(f"🧠 Running AI optimization ({population_size} population, {generations} generations)...")
        optimization_start = time.time()
        
        result = engine.run_optimization(
            population_size=population_size,
            generations=generations
        )
        
        optimization_time = time.time() - optimization_start
        
        # Display results
        print("\n" + "="*60)
        print("🎉 OPTIMIZATION COMPLETE!")
        print("="*60)
        
        metrics = result['performance_metrics']
        print(f"📏 Total Distance: {metrics['total_distance']:.2f} km")
        print(f"⚖️  Workload Balance: {metrics['workload_balance']:.3f}")
        print(f"📊 Geographic Efficiency: {metrics['geographic_efficiency']:.3f}")
        print(f"📈 Improvement over Random: {metrics['improvement_over_random']:.1f}%")
        print(f"⏱️  Processing Time: {optimization_time:.1f} seconds")
        
        print(f"\n📋 Optimal Assignment:")
        assignment = result['optimal_assignment']
        total_shipments = sum(len(shipments) for shipments in assignment.values())
        
        for driver_id, shipments in assignment.items():
            percentage = (len(shipments) / total_shipments) * 100
            print(f"   🚗 {driver_id}: {len(shipments):2d} shipments ({percentage:.1f}%)")
        
        # Create visualizations
        if visualize:
            print(f"\n📈 Creating visualizations...")
            viz_start = time.time()
            visualizations = engine.visualize_results(result)
            viz_time = time.time() - viz_start
            print(f"   ✅ Visualizations created in {viz_time:.1f} seconds")
            print(f"   📄 Interactive map: ai_dispatch_map.html")
            print(f"   📊 Dashboard: ai_dispatch_dashboard.png")
        
        # Save results
        if save_results:
            print(f"\n💾 Saving results...")
            result_file = engine.save_results(result)
            print(f"   ✅ Results saved to: {result_file}")
        
        # Performance summary
        total_time = time.time() - start_time
        print(f"\n⏱️  Total Runtime: {total_time:.1f} seconds")
        print(f"   Distance Matrix: {matrix_time:.1f}s")
        print(f"   AI Optimization: {optimization_time:.1f}s")
        if visualize:
            print(f"   Visualizations: {viz_time:.1f}s")
        
        print(f"\n🎯 AI Performance:")
        convergence_gen = len(result['optimization_history'])
        print(f"   Converged in: {convergence_gen} generations")
        print(f"   Efficiency: {metrics['improvement_over_random']:.1f}% better than random")
        
        if metrics['improvement_over_random'] > 20:
            print(f"   Status: 🌟 Excellent optimization!")
        elif metrics['improvement_over_random'] > 10:
            print(f"   Status: ✅ Good optimization")
        else:
            print(f"   Status: ⚠️  Fair - consider tuning parameters")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Run setup.py first to install dependencies")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("   Check your data files and OSRM connection")
        return False

def main():
    """Main function with command line arguments"""
    parser = argparse.ArgumentParser(
        description="AI Dispatch Engine for Khartoum - Intelligent Shipment Assignment"
    )
    
    parser.add_argument(
        '--shipments', 
        default='data/sample_shipments.json',
        help='Path to shipments JSON file (default: data/sample_shipments.json)'
    )
    
    parser.add_argument(
        '--drivers',
        default='data/sample_drivers.json', 
        help='Path to drivers JSON file (default: data/sample_drivers.json)'
    )
    
    parser.add_argument(
        '--population',
        type=int,
        default=100,
        help='Genetic algorithm population size (default: 100)'
    )
    
    parser.add_argument(
        '--generations',
        type=int, 
        default=300,
        help='Maximum generations to run (default: 300)'
    )
    
    parser.add_argument(
        '--osrm-url',
        default='http://localhost:5000',
        help='OSRM server URL (default: http://localhost:5000)'
    )
    
    parser.add_argument(
        '--no-visualize',
        action='store_true',
        help='Skip creating visualizations (faster)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true', 
        help='Skip saving results to file'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (small population, few generations)'
    )
    
    args = parser.parse_args()
    
    # Quick mode adjustments
    if args.quick:
        args.population = 20
        args.generations = 50
        print("🏃 Quick test mode enabled")
    
    # Print banner
    print_banner()
    
    # Run dispatch
    success = run_dispatch(
        shipments_file=args.shipments,
        drivers_file=args.drivers,
        population_size=args.population,
        generations=args.generations,
        osrm_url=args.osrm_url,
        visualize=not args.no_visualize,
        save_results=not args.no_save
    )
    
    if success:
        print("\n🎉 AI Dispatch completed successfully!")
        print("📖 Check README.md for more usage examples")
    else:
        print("\n❌ AI Dispatch failed!")
        print("🔧 Check setup and data files")
        sys.exit(1)

if __name__ == "__main__":
    main()