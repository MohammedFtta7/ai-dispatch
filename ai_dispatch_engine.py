"""
AI Dispatch Engine for Khartoum
Intelligent shipment assignment using genetic algorithms and real-world routing
"""

import numpy as np
import pandas as pd
import json
import random
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging

from osrm_client import OSRMClient
from genetic_optimizer import GeneticOptimizer
from performance_monitor import PerformanceMonitor
from visualizer import DispatchVisualizer

@dataclass
class Shipment:
    id: str
    pickup_lat: float
    pickup_lon: float
    delivery_lat: float
    delivery_lon: float
    
    def __hash__(self):
        return hash(self.id)

@dataclass
class Driver:
    id: str
    capacity: float

class AIDispatchEngine:
    def __init__(self, osrm_url="http://localhost:5000"):
        """Initialize AI Dispatch Engine"""
        self.osrm_client = OSRMClient(osrm_url)
        self.genetic_optimizer = GeneticOptimizer()
        self.performance_monitor = PerformanceMonitor()
        self.visualizer = DispatchVisualizer()
        
        # Learning parameters
        self.learning_memory = {}
        self.run_count = 0
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_data(self, shipments_file: str, drivers_file: str):
        """Load shipments and drivers from JSON files"""
        try:
            # Load shipments
            with open(shipments_file, 'r') as f:
                shipments_data = json.load(f)
            self.shipments = [Shipment(**s) for s in shipments_data]
            
            # Load drivers
            with open(drivers_file, 'r') as f:
                drivers_data = json.load(f)
            self.drivers = [Driver(**d) for d in drivers_data]
            
            self.logger.info(f"Loaded {len(self.shipments)} shipments and {len(self.drivers)} drivers")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return False
    
    def calculate_distance_matrix(self):
        """Calculate real-world distance matrix using OSRM"""
        self.logger.info("Calculating distance matrix using OSRM...")
        
        distances = {}
        times = {}
        
        for shipment in self.shipments:
            try:
                # Get route from pickup to delivery
                route_data = self.osrm_client.get_route(
                    shipment.pickup_lat, shipment.pickup_lon,
                    shipment.delivery_lat, shipment.delivery_lon
                )
                
                distances[shipment.id] = route_data['distance']
                times[shipment.id] = route_data['duration']
                
            except Exception as e:
                self.logger.warning(f"OSRM error for shipment {shipment.id}: {e}")
                # Fallback to straight-line distance
                distances[shipment.id] = self._calculate_haversine_distance(
                    shipment.pickup_lat, shipment.pickup_lon,
                    shipment.delivery_lat, shipment.delivery_lon
                )
                times[shipment.id] = distances[shipment.id] / 30 * 60  # Assume 30 km/h
        
        self.distance_matrix = distances
        self.time_matrix = times
        self.logger.info("Distance matrix calculated successfully")
        
    def _calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate straight-line distance as fallback"""
        from math import radians, cos, sin, asin, sqrt
        
        R = 6371  # Earth's radius in kilometers
        
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        return R * c
    
    def run_optimization(self, population_size=100, generations=500):
        """Run genetic algorithm optimization"""
        self.logger.info(f"Starting AI optimization with {population_size} population, {generations} generations")
        
        start_time = time.time()
        
        # Setup genetic optimizer with OSRM integration
        self.genetic_optimizer.setup(
            shipments=self.shipments,
            drivers=self.drivers,
            distance_matrix=self.distance_matrix,
            population_size=population_size,
            generations=generations,
            osrm_client=self.osrm_client  # Pass OSRM client properly
        )
        
        # Run optimization with real-world road awareness
        best_assignment, optimization_history = self.genetic_optimizer.evolve()
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(best_assignment)
        
        # Track performance for learning
        self.performance_monitor.track_run({
            'run_id': self.run_count,
            'total_distance': performance_metrics['total_distance'],
            'balance_score': performance_metrics['workload_balance'],
            'cluster_score': performance_metrics['geographic_efficiency'],
            'generations': len(optimization_history),
            'improvement_%': performance_metrics['improvement_over_random'],
            'processing_time': time.time() - start_time
        })
        
        self.run_count += 1
        
        # Store learning patterns
        self._update_learning_memory(best_assignment, performance_metrics)
        
        result = {
            'optimal_assignment': best_assignment,
            'performance_metrics': performance_metrics,
            'optimization_history': optimization_history,
            'processing_time': time.time() - start_time
        }
        
        self.logger.info(f"Optimization completed in {result['processing_time']:.2f} seconds")
        return result
    
    def _calculate_performance_metrics(self, assignment):
        """Calculate comprehensive performance metrics"""
        total_distance = 0
        driver_loads = []
        
        for driver_id, shipment_ids in assignment.items():
            driver_distance = sum(self.distance_matrix[sid] for sid in shipment_ids)
            total_distance += driver_distance
            driver_loads.append(len(shipment_ids))
        
        # Workload balance (coefficient of variation)
        workload_balance = 1 - (np.std(driver_loads) / np.mean(driver_loads)) if driver_loads else 0
        
        # Geographic efficiency (placeholder - could be improved)
        geographic_efficiency = 0.8  # Simplified for now
        
        # Compare to random assignment
        random_distance = self._calculate_random_baseline()
        improvement_over_random = ((random_distance - total_distance) / random_distance) * 100
        
        return {
            'total_distance': total_distance,
            'workload_balance': workload_balance,
            'geographic_efficiency': geographic_efficiency,
            'improvement_over_random': improvement_over_random,
            'average_shipments_per_driver': np.mean(driver_loads),
            'driver_utilization': driver_loads
        }
    
    def _calculate_random_baseline(self):
        """Calculate baseline performance with random assignment"""
        shipment_ids = [s.id for s in self.shipments]
        random.shuffle(shipment_ids)
        
        # Distribute randomly among drivers
        assignments_per_driver = len(shipment_ids) // len(self.drivers)
        total_random_distance = 0
        
        for i, driver in enumerate(self.drivers):
            start_idx = i * assignments_per_driver
            end_idx = start_idx + assignments_per_driver
            if i == len(self.drivers) - 1:  # Last driver gets remaining
                end_idx = len(shipment_ids)
            
            driver_shipments = shipment_ids[start_idx:end_idx]
            driver_distance = sum(self.distance_matrix[sid] for sid in driver_shipments)
            total_random_distance += driver_distance
        
        return total_random_distance
    
    def _update_learning_memory(self, assignment, metrics):
        """Store successful patterns for future learning"""
        pattern_key = self._generate_pattern_key(assignment)
        
        if pattern_key not in self.learning_memory:
            self.learning_memory[pattern_key] = {
                'success_count': 0,
                'total_distance': [],
                'balance_scores': []
            }
        
        self.learning_memory[pattern_key]['success_count'] += 1
        self.learning_memory[pattern_key]['total_distance'].append(metrics['total_distance'])
        self.learning_memory[pattern_key]['balance_scores'].append(metrics['workload_balance'])
    
    def _generate_pattern_key(self, assignment):
        """Generate a pattern key for learning memory"""
        # Simple pattern: workload distribution
        loads = [len(shipments) for shipments in assignment.values()]
        loads.sort()
        return tuple(loads)
    
    def visualize_results(self, result):
        """Create visualizations for the results"""
        self.logger.info("Generating visualizations...")
        
        # Create map visualization
        map_viz = self.visualizer.create_assignment_map(
            self.shipments, result['optimal_assignment']
        )
        
        # Create performance dashboard
        dashboard = self.visualizer.create_performance_dashboard(
            result['optimization_history'], 
            result['performance_metrics']
        )
        
        # Show learning trends if multiple runs
        if self.run_count > 1:
            learning_viz = self.performance_monitor.show_learning_trends()
            return map_viz, dashboard, learning_viz
        
        return map_viz, dashboard
    
    def save_results(self, result, filename_prefix="dispatch_result"):
        """Save results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save assignment as JSON
        assignment_file = f"{filename_prefix}_{timestamp}.json"
        with open(assignment_file, 'w') as f:
            json.dump({
                'assignment': result['optimal_assignment'],
                'metrics': result['performance_metrics'],
                'timestamp': timestamp
            }, f, indent=2)
        
        self.logger.info(f"Results saved to {assignment_file}")
        return assignment_file

def main():
    """Main function to run the AI dispatch engine"""
    print("🚀 AI Dispatch Engine for Khartoum")
    print("=" * 50)
    
    # Initialize engine
    engine = AIDispatchEngine()
    
    # Load data
    print("📊 Loading data...")
    if not engine.load_data('data/sample_shipments.json', 'data/sample_drivers.json'):
        print("❌ Failed to load data. Check your data files.")
        return
    
    # Calculate distance matrix
    print("🗺️  Calculating distance matrix...")
    engine.calculate_distance_matrix()
    
    # Run optimization
    print("🧠 Running AI optimization...")
    result = engine.run_optimization(population_size=100, generations=300)
    
    # Display results
    print("\n✅ Optimization Complete!")
    print(f"Total Distance: {result['performance_metrics']['total_distance']:.2f} km")
    print(f"Workload Balance: {result['performance_metrics']['workload_balance']:.3f}")
    print(f"Improvement over Random: {result['performance_metrics']['improvement_over_random']:.1f}%")
    print(f"Processing Time: {result['processing_time']:.2f} seconds")
    
    # Show assignment
    print("\n📋 Optimal Assignment:")
    for driver_id, shipments in result['optimal_assignment'].items():
        print(f"  {driver_id}: {len(shipments)} shipments")
    
    # Create visualizations
    print("\n📈 Generating visualizations...")
    visualizations = engine.visualize_results(result)
    
    # Save results
    print("\n💾 Saving results...")
    result_file = engine.save_results(result)
    
    print(f"\n🎉 Complete! Results saved to {result_file}")
    print("Check the generated HTML files for visualizations.")

if __name__ == "__main__":
    main()