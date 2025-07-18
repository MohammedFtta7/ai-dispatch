"""
Genetic Algorithm Optimizer for AI Dispatch
Evolves optimal shipment assignments using genetic programming
"""

import numpy as np
import random
import logging
from typing import List, Dict, Tuple, Any
from dataclasses import dataclass
import copy

@dataclass
class Individual:
    """Represents a solution (assignment) in the genetic algorithm"""
    assignment: Dict[str, List[str]]  # driver_id -> [shipment_ids]
    fitness: float = 0.0
    age: int = 0
    
    def __hash__(self):
        # Create hash from assignment pattern
        pattern = tuple(sorted([(k, tuple(sorted(v))) for k, v in self.assignment.items()]))
        return hash(pattern)

class GeneticOptimizer:
    def __init__(self):
        """Initialize genetic optimizer"""
        self.logger = logging.getLogger(__name__)
        
        # Genetic algorithm parameters
        self.population_size = 100
        self.generations = 500
        self.elite_size = 10
        self.mutation_rate = 0.15
        self.crossover_rate = 0.8
        
        # Multi-objective weights - PRIORITIZE SMART CLUSTERING
        self.clustering_weight = 0.6     # 60% - Geographic clustering efficiency (PRIMARY)
        self.distance_weight = 0.3       # 30% - Individual shipment distances  
        self.balance_weight = 0.1        # 10% - Basic workload balance (minimal)
        
        # Tracking
        self.best_fitness_history = []
        self.diversity_history = []
        self.convergence_threshold = 50  # generations without improvement
        
    def setup(self, shipments, drivers, distance_matrix, population_size=100, generations=500, osrm_client=None):
        """Setup optimizer with problem data"""
        self.shipments = shipments
        self.drivers = drivers
        self.distance_matrix = distance_matrix
        self.population_size = population_size
        self.generations = generations
        self.osrm_client = osrm_client  # Store OSRM client directly
        
        self.shipment_ids = [s.id for s in shipments]
        self.driver_ids = [d.id for d in drivers]
        
        # Build OSRM route matrix for clustering (pickup-to-pickup routes)
        self._build_clustering_matrix()
        
        self.logger.info(f"Genetic optimizer setup: {len(shipments)} shipments, {len(drivers)} drivers")
    
    def _build_clustering_matrix(self):
        """Build matrix of real OSRM routes between all pickup points for clustering"""
        self.logger.info("Building real-world clustering matrix using OSRM...")
        
        self.clustering_matrix = {}
        self.delivery_to_pickup_matrix = {}  # NEW: Pre-calculate delivery-to-pickup routes
        osrm_success_count = 0
        osrm_failure_count = 0
        
        # Debug OSRM client status
        if self.osrm_client:
            self.logger.info(f"OSRM client available: {type(self.osrm_client)}")
            self.logger.info(f"OSRM client connected: {getattr(self.osrm_client, 'connected', 'unknown')}")
        else:
            self.logger.error("NO OSRM CLIENT - this is the problem!")
        
        # Build pickup-to-pickup matrix (existing)
        for i, s1 in enumerate(self.shipments):
            for j, s2 in enumerate(self.shipments):
                if i != j:
                    key = f"{s1.id}-{s2.id}"
                    
                    # Use OSRM to get real route distance between pickup points
                    if self.osrm_client:
                        try:
                            # Debug first few calls
                            if osrm_success_count + osrm_failure_count < 3:
                                self.logger.info(f"Trying OSRM route: {s1.id}({s1.pickup_lat},{s1.pickup_lon}) → {s2.id}({s2.pickup_lat},{s2.pickup_lon})")
                            
                            route = self.osrm_client.get_route(
                                s1.pickup_lat, s1.pickup_lon,
                                s2.pickup_lat, s2.pickup_lon
                            )
                            
                            # Store real route distance (includes bridges, road constraints)
                            self.clustering_matrix[key] = route['distance']
                            osrm_success_count += 1
                            
                            # Debug successful calls
                            if osrm_success_count <= 3:
                                self.logger.info(f"OSRM SUCCESS: {key} = {route['distance']:.2f} km")
                            
                        except Exception as e:
                            # Debug failures
                            if osrm_failure_count < 3:
                                self.logger.error(f"OSRM FAILED for {key}: {type(e).__name__}: {str(e)}")
                            
                            # Fallback to Haversine only if OSRM completely fails
                            fallback_distance = self._haversine_distance(
                                s1.pickup_lat, s1.pickup_lon,
                                s2.pickup_lat, s2.pickup_lon
                            )
                            self.clustering_matrix[key] = fallback_distance
                            osrm_failure_count += 1
                    else:
                        # No OSRM client available, use Haversine
                        fallback_distance = self._haversine_distance(
                            s1.pickup_lat, s1.pickup_lon,
                            s2.pickup_lat, s2.pickup_lon
                        )
                        self.clustering_matrix[key] = fallback_distance
                        osrm_failure_count += 1
        
        # NEW: Build delivery-to-pickup matrix (THIS FIXES THE PERFORMANCE ISSUE!)
        self.logger.info("Building delivery-to-pickup matrix...")
        delivery_to_pickup_count = 0
        
        for i, s1 in enumerate(self.shipments):
            for j, s2 in enumerate(self.shipments):
                if i != j:
                    # Key for delivery of s1 to pickup of s2
                    delivery_pickup_key = f"DEL_{s1.id}_TO_PICK_{s2.id}"
                    
                    if self.osrm_client:
                        try:
                            # Calculate route from s1 delivery to s2 pickup
                            route = self.osrm_client.get_route(
                                s1.delivery_lat, s1.delivery_lon,
                                s2.pickup_lat, s2.pickup_lon
                            )
                            
                            self.delivery_to_pickup_matrix[delivery_pickup_key] = route['distance']
                            delivery_to_pickup_count += 1
                            
                        except Exception:
                            # Fallback
                            fallback_distance = self._haversine_distance(
                                s1.delivery_lat, s1.delivery_lon,
                                s2.pickup_lat, s2.pickup_lon
                            )
                            self.delivery_to_pickup_matrix[delivery_pickup_key] = fallback_distance
                    else:
                        # Fallback
                        fallback_distance = self._haversine_distance(
                            s1.delivery_lat, s1.delivery_lon,
                            s2.pickup_lat, s2.pickup_lon
                        )
                        self.delivery_to_pickup_matrix[delivery_pickup_key] = fallback_distance
        
        self.logger.info(f"Built delivery-to-pickup matrix: {delivery_to_pickup_count} routes")
        
        # Log OSRM usage statistics
        total_routes = osrm_success_count + osrm_failure_count + delivery_to_pickup_count
        if total_routes > 0:
            osrm_percentage = ((osrm_success_count + delivery_to_pickup_count) / total_routes) * 100
            self.logger.info(f"OSRM Usage: {osrm_success_count + delivery_to_pickup_count}/{total_routes} routes ({osrm_percentage:.1f}%)")
            
            if osrm_percentage < 50:
                self.logger.warning("Low OSRM success rate - check server and data coverage")
            else:
                self.logger.info("Good OSRM integration - bridge intelligence active")
        
        # Log some sample comparisons to show bridge detection
        if len(self.clustering_matrix) > 0:
            self._log_sample_route_comparisons()
        
        self.logger.info("Real-world clustering matrix completed")
    
    def _log_sample_route_comparisons(self):
        """Log sample route comparisons to show OSRM vs straight-line differences"""
        sample_count = 0
        
        for i, s1 in enumerate(self.shipments[:3]):  # Just first few for logging
            for j, s2 in enumerate(self.shipments[:3]):
                if i != j and sample_count < 3:
                    key = f"{s1.id}-{s2.id}"
                    
                    if key in self.clustering_matrix:
                        osrm_distance = self.clustering_matrix[key]
                        straight_distance = self._haversine_distance(
                            s1.pickup_lat, s1.pickup_lon,
                            s2.pickup_lat, s2.pickup_lon
                        )
                        
                        ratio = osrm_distance / (straight_distance + 0.1)
                        bridge_indicator = "🌉 BRIDGE/COMPLEX" if ratio > 1.5 else "🛣️ DIRECT"
                        
                        self.logger.info(
                            f"Route {s1.id}→{s2.id}: "
                            f"OSRM={osrm_distance:.1f}km, "
                            f"Straight={straight_distance:.1f}km, "
                            f"Ratio={ratio:.2f} {bridge_indicator}"
                        )
                        sample_count += 1
        
        # Log complete route example for debugging
        if len(self.shipments) >= 3:
            sample_shipments = self.shipments[:3]
            self.logger.info("🚗 Sample complete driver route calculation:")
            
            for i in range(len(sample_shipments) - 1):
                current = sample_shipments[i]
                next_ship = sample_shipments[i + 1]
                
                # Use pre-calculated delivery-to-pickup distance
                delivery_pickup_key = f"DEL_{current.id}_TO_PICK_{next_ship.id}"
                
                if delivery_pickup_key in self.delivery_to_pickup_matrix:
                    distance = self.delivery_to_pickup_matrix[delivery_pickup_key]
                    
                    straight = self._haversine_distance(
                        current.delivery_lat, current.delivery_lon,
                        next_ship.pickup_lat, next_ship.pickup_lon
                    )
                    
                    ratio = distance / (straight + 0.1)
                    bridge_indicator = "🌉 BRIDGE" if ratio > 1.4 else "🛣️ DIRECT"
                    
                    self.logger.info(
                        f"   {current.id} delivery → {next_ship.id} pickup: "
                        f"{distance:.1f}km {bridge_indicator} (ratio: {ratio:.2f})"
                    )
                else:
                    self.logger.warning(f"   Missing delivery-to-pickup data for {current.id} → {next_ship.id}")
        else:
            self.logger.info("   Insufficient data for complete route example")
    
    def evolve(self) -> Tuple[Dict[str, List[str]], List[Dict]]:
        """Main evolution loop"""
        self.logger.info("Starting genetic evolution...")
        
        # Initialize population
        population = self._initialize_population()
        
        # Track optimization history
        history = []
        best_fitness = float('inf')
        generations_without_improvement = 0
        
        for generation in range(self.generations):
            # Evaluate fitness for all individuals
            for individual in population:
                individual.fitness = self._calculate_fitness(individual.assignment)
                individual.age += 1
            
            # Sort by fitness (lower is better)
            population.sort(key=lambda x: x.fitness)
            current_best = population[0].fitness
            
            # Track progress
            if current_best < best_fitness:
                best_fitness = current_best
                generations_without_improvement = 0
            else:
                generations_without_improvement += 1
            
            # Record history
            diversity = self._calculate_diversity(population)
            history.append({
                'generation': generation,
                'best_fitness': current_best,
                'average_fitness': np.mean([ind.fitness for ind in population]),
                'diversity': diversity,
                'convergence_rate': generations_without_improvement
            })
            
            self.best_fitness_history.append(current_best)
            self.diversity_history.append(diversity)
            
            # Log progress
            if generation % 50 == 0:
                self.logger.info(f"Generation {generation}: Best={current_best:.2f}, "
                               f"Diversity={diversity:.3f}, No improvement={generations_without_improvement}")
            
            # Check for convergence
            if generations_without_improvement >= self.convergence_threshold:
                self.logger.info(f"Converged at generation {generation}")
                break
            
            # Create next generation
            population = self._create_next_generation(population)
        
        # Return best solution
        best_individual = min(population, key=lambda x: x.fitness)
        self.logger.info(f"Evolution complete. Best fitness: {best_individual.fitness:.2f}")
        
        return best_individual.assignment, history
    
    def _initialize_population(self) -> List[Individual]:
        """Create initial population with diverse strategies"""
        population = []
        
        # Strategy 1: Random assignments (40%)
        for _ in range(int(self.population_size * 0.4)):
            assignment = self._random_assignment()
            population.append(Individual(assignment=assignment))
        
        # Strategy 2: Geographic clustering (30%)
        for _ in range(int(self.population_size * 0.3)):
            assignment = self._geographic_assignment()
            population.append(Individual(assignment=assignment))
        
        # Strategy 3: Balanced workload (20%)
        for _ in range(int(self.population_size * 0.2)):
            assignment = self._balanced_assignment()
            population.append(Individual(assignment=assignment))
        
        # Strategy 4: Distance-based (10%)
        remaining = self.population_size - len(population)
        for _ in range(remaining):
            assignment = self._distance_based_assignment()
            population.append(Individual(assignment=assignment))
        
        return population
    
    def _random_assignment(self) -> Dict[str, List[str]]:
        """Create random assignment"""
        assignment = {driver_id: [] for driver_id in self.driver_ids}
        
        shipment_list = self.shipment_ids.copy()
        random.shuffle(shipment_list)
        
        # Round-robin assignment
        for i, shipment_id in enumerate(shipment_list):
            driver_id = self.driver_ids[i % len(self.driver_ids)]
            assignment[driver_id].append(shipment_id)
        
        return assignment
    
    def _geographic_assignment(self) -> Dict[str, List[str]]:
        """Create assignment based on geographic clustering"""
        from sklearn.cluster import KMeans
        
        # Get pickup coordinates
        coordinates = np.array([[s.pickup_lat, s.pickup_lon] for s in self.shipments])
        
        # Cluster into groups (one per driver)
        n_clusters = min(len(self.driver_ids), len(self.shipments))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(coordinates)
        
        # Assign clusters to drivers
        assignment = {driver_id: [] for driver_id in self.driver_ids}
        
        for i, cluster_id in enumerate(clusters):
            driver_idx = cluster_id % len(self.driver_ids)
            driver_id = self.driver_ids[driver_idx]
            assignment[driver_id].append(self.shipment_ids[i])
        
        return assignment
    
    def _balanced_assignment(self) -> Dict[str, List[str]]:
        """Create balanced workload assignment"""
        assignment = {driver_id: [] for driver_id in self.driver_ids}
        
        # Sort shipments by distance (longest first)
        shipments_by_distance = sorted(
            self.shipment_ids, 
            key=lambda x: self.distance_matrix[x], 
            reverse=True
        )
        
        # Assign to driver with smallest current load
        for shipment_id in shipments_by_distance:
            # Find driver with minimum current workload
            min_driver = min(self.driver_ids, key=lambda d: len(assignment[d]))
            assignment[min_driver].append(shipment_id)
        
        return assignment
    
    def _distance_based_assignment(self) -> Dict[str, List[str]]:
        """Create assignment prioritizing short distances"""
        assignment = {driver_id: [] for driver_id in self.driver_ids}
        
        # Sort by distance (shortest first)
        shipments_by_distance = sorted(
            self.shipment_ids,
            key=lambda x: self.distance_matrix[x]
        )
        
        # Assign to drivers in round-robin
        for i, shipment_id in enumerate(shipments_by_distance):
            driver_id = self.driver_ids[i % len(self.driver_ids)]
            assignment[driver_id].append(shipment_id)
        
        return assignment
    
    def _calculate_fitness(self, assignment: Dict[str, List[str]]) -> float:
        """Calculate multi-objective fitness score with smart clustering priority"""
        
        # Check capacity constraints with 10% flexibility
        if not self._is_assignment_valid(assignment):
            return float('inf')  # Invalid assignment gets worst possible score
        
        # Objective 1: Total distance (minimize)
        total_distance = sum(
            sum(self.distance_matrix[shipment_id] for shipment_id in shipments)
            for shipments in assignment.values()
        )
        distance_score = total_distance
        
        # Objective 2: Workload balance (minimize variance) - now less important
        workloads = [len(shipments) for shipments in assignment.values() if len(shipments) > 0]
        if len(workloads) > 1:
            workload_variance = np.var(workloads)
            balance_score = workload_variance * 5  # Reduced penalty scale
        else:
            balance_score = 0
        
        # Objective 3: Smart geographic clustering (minimize driver travel between pickups)
        clustering_score = self._calculate_smart_clustering_penalty(assignment)
        
        # Combined fitness (lower is better) - CLUSTERING NOW DOMINATES
        fitness = (
            self.clustering_weight * clustering_score +
            self.distance_weight * distance_score +
            self.balance_weight * balance_score
        )
        
        return fitness
    
    def _is_assignment_valid(self, assignment: Dict[str, List[str]]) -> bool:
        """Check if assignment respects capacity constraints with 10% flexibility"""
        for driver_id, shipment_ids in assignment.items():
            # Find driver capacity
            driver = next((d for d in self.drivers if d.id == driver_id), None)
            if driver and len(shipment_ids) > driver.capacity * 1.1:  # 110% flexibility
                return False
        return True
    
    def _calculate_smart_clustering_penalty(self, assignment: Dict[str, List[str]]) -> float:
        """Calculate intelligent geographic clustering penalty using COMPLETE driver routes"""
        total_penalty = 0.0
        
        for driver_id, shipment_ids in assignment.items():
            if len(shipment_ids) <= 1:
                continue
            
            # Calculate COMPLETE driver route cost: pickup1→delivery1→pickup2→delivery2→...
            complete_route_cost = self._calculate_complete_driver_route(shipment_ids)
            
            # Apply penalty based on total route inefficiency
            total_penalty += complete_route_cost * 100  # Scale for proper impact
        
        return total_penalty
    
    def _calculate_complete_driver_route(self, shipment_ids: List[str]) -> float:
        """Calculate the complete route a driver must travel for all assigned shipments"""
        if len(shipment_ids) <= 1:
            return 0.0
        
        total_route_cost = 0.0
        
        # Get shipment objects
        shipments = [next(s for s in self.shipments if s.id == sid) for sid in shipment_ids]
        
        # Method 1: Calculate delivery-to-next-pickup distances using PRE-CALCULATED matrix
        for i in range(len(shipments) - 1):
            current_shipment = shipments[i]
            next_shipment = shipments[i + 1]
            
            # Use pre-calculated delivery-to-pickup distance (NO MORE OSRM CALLS!)
            delivery_pickup_key = f"DEL_{current_shipment.id}_TO_PICK_{next_shipment.id}"
            
            if delivery_pickup_key in self.delivery_to_pickup_matrix:
                delivery_to_pickup_distance = self.delivery_to_pickup_matrix[delivery_pickup_key]
            else:
                # Fallback calculation (shouldn't happen)
                delivery_to_pickup_distance = self._haversine_distance(
                    current_shipment.delivery_lat, current_shipment.delivery_lon,
                    next_shipment.pickup_lat, next_shipment.pickup_lon
                )
            
            total_route_cost += delivery_to_pickup_distance
        
        # Method 2: Add pickup clustering penalty (secondary)
        pickup_clustering_penalty = 0.0
        for i, shipment1 in enumerate(shipments):
            for shipment2 in shipments[i+1:]:
                # Distance between pickup points
                pickup_key = f"{shipment1.id}-{shipment2.id}"
                reverse_key = f"{shipment2.id}-{shipment1.id}"
                
                if pickup_key in self.clustering_matrix:
                    pickup_distance = self.clustering_matrix[pickup_key]
                elif reverse_key in self.clustering_matrix:
                    pickup_distance = self.clustering_matrix[reverse_key]
                else:
                    pickup_distance = self._haversine_distance(
                        shipment1.pickup_lat, shipment1.pickup_lon,
                        shipment2.pickup_lat, shipment2.pickup_lon
                    )
                
                pickup_clustering_penalty += pickup_distance
        
        # Total route cost = delivery-to-pickup travel + pickup clustering
        return total_route_cost + (pickup_clustering_penalty * 0.5)  # Weight delivery-to-pickup higher
    
    def _detect_bridge_crossings_in_route(self, shipment_ids: List[str]) -> float:
        """Detect bridge crossings in complete driver route using pre-calculated matrix"""
        if len(shipment_ids) <= 1:
            return 0.0
        
        bridge_penalty = 0.0
        shipments = [next(s for s in self.shipments if s.id == sid) for sid in shipment_ids]
        
        # Check for bridge crossings in delivery-to-pickup segments using pre-calculated data
        for i in range(len(shipments) - 1):
            current = shipments[i]
            next_shipment = shipments[i + 1]
            
            # Get pre-calculated delivery-to-pickup distance
            delivery_pickup_key = f"DEL_{current.id}_TO_PICK_{next_shipment.id}"
            
            if delivery_pickup_key in self.delivery_to_pickup_matrix:
                osrm_distance = self.delivery_to_pickup_matrix[delivery_pickup_key]
                
                # Compare to straight-line
                straight_distance = self._haversine_distance(
                    current.delivery_lat, current.delivery_lon,
                    next_shipment.pickup_lat, next_shipment.pickup_lon
                )
                
                ratio = osrm_distance / (straight_distance + 0.1)
                
                # Detect probable bridge crossing
                if ratio > 1.4:  # Route 40% longer than straight-line
                    bridge_penalty += (ratio - 1.0) * 3000  # Heavy bridge penalty
        
        return bridge_penalty
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        return 6371 * c  # Earth's radius in kilometers
    
    def _create_next_generation(self, population: List[Individual]) -> List[Individual]:
        """Create next generation using selection, crossover, and mutation"""
        next_generation = []
        
        # Elitism: Keep best individuals
        elite_count = min(self.elite_size, len(population))
        next_generation.extend(population[:elite_count])
        
        # Generate rest through crossover and mutation
        while len(next_generation) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
                next_generation.extend([child1, child2])
            else:
                next_generation.extend([
                    Individual(assignment=copy.deepcopy(parent1.assignment)),
                    Individual(assignment=copy.deepcopy(parent2.assignment))
                ])
        
        # Mutation
        for individual in next_generation[elite_count:]:  # Don't mutate elites
            if random.random() < self.mutation_rate:
                self._mutate(individual)
        
        # Trim to exact population size
        return next_generation[:self.population_size]
    
    def _tournament_selection(self, population: List[Individual], tournament_size: int = 3) -> Individual:
        """Tournament selection for parent selection"""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return min(tournament, key=lambda x: x.fitness)
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Order crossover for assignments"""
        # Simple crossover: randomly distribute shipments from both parents
        all_shipments = set()
        for shipments in parent1.assignment.values():
            all_shipments.update(shipments)
        
        # Create two children
        child1_assignment = {driver_id: [] for driver_id in self.driver_ids}
        child2_assignment = {driver_id: [] for driver_id in self.driver_ids}
        
        shipment_list = list(all_shipments)
        random.shuffle(shipment_list)
        
        # Split randomly between children
        for i, shipment_id in enumerate(shipment_list):
            # Bias towards parent assignments
            p1_driver = self._find_driver_for_shipment(parent1.assignment, shipment_id)
            p2_driver = self._find_driver_for_shipment(parent2.assignment, shipment_id)
            
            if random.random() < 0.5:
                # Child 1 inherits from parent 1, child 2 from parent 2
                child1_assignment[p1_driver].append(shipment_id)
                child2_assignment[p2_driver].append(shipment_id)
            else:
                # Child 1 inherits from parent 2, child 2 from parent 1
                child1_assignment[p2_driver].append(shipment_id)
                child2_assignment[p1_driver].append(shipment_id)
        
        return (
            Individual(assignment=child1_assignment),
            Individual(assignment=child2_assignment)
        )
    
    def _find_driver_for_shipment(self, assignment: Dict[str, List[str]], shipment_id: str) -> str:
        """Find which driver has a specific shipment"""
        for driver_id, shipments in assignment.items():
            if shipment_id in shipments:
                return driver_id
        return random.choice(self.driver_ids)  # Fallback
    
    def _mutate(self, individual: Individual):
        """Mutate an individual by moving shipments between drivers"""
        assignment = individual.assignment
        
        # Choose mutation type
        mutation_type = random.choice(['swap', 'move', 'redistribute'])
        
        if mutation_type == 'swap':
            self._mutation_swap_shipments(assignment)
        elif mutation_type == 'move':
            self._mutation_move_shipment(assignment)
        else:
            self._mutation_redistribute(assignment)
    
    def _mutation_swap_shipments(self, assignment: Dict[str, List[str]]):
        """Swap shipments between two drivers"""
        drivers_with_shipments = [d for d, s in assignment.items() if len(s) > 0]
        
        if len(drivers_with_shipments) >= 2:
            driver1, driver2 = random.sample(drivers_with_shipments, 2)
            
            if assignment[driver1] and assignment[driver2]:
                shipment1 = random.choice(assignment[driver1])
                shipment2 = random.choice(assignment[driver2])
                
                # Swap
                assignment[driver1].remove(shipment1)
                assignment[driver2].remove(shipment2)
                assignment[driver1].append(shipment2)
                assignment[driver2].append(shipment1)
    
    def _mutation_move_shipment(self, assignment: Dict[str, List[str]]):
        """Move a shipment from one driver to another"""
        drivers_with_shipments = [d for d, s in assignment.items() if len(s) > 0]
        
        if drivers_with_shipments:
            source_driver = random.choice(drivers_with_shipments)
            target_driver = random.choice(self.driver_ids)
            
            if assignment[source_driver]:
                shipment = random.choice(assignment[source_driver])
                assignment[source_driver].remove(shipment)
                assignment[target_driver].append(shipment)
    
    def _mutation_redistribute(self, assignment: Dict[str, List[str]]):
        """Redistribute shipments to balance workload"""
        # Find overloaded and underloaded drivers
        workloads = [(d, len(s)) for d, s in assignment.items()]
        workloads.sort(key=lambda x: x[1])
        
        if len(workloads) >= 2:
            underloaded = workloads[0][0]
            overloaded = workloads[-1][0]
            
            if assignment[overloaded]:
                shipment = random.choice(assignment[overloaded])
                assignment[overloaded].remove(shipment)
                assignment[underloaded].append(shipment)
    
    def _calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity"""
        if len(population) <= 1:
            return 0.0
        
        # Calculate pairwise differences in assignments
        diversity_sum = 0.0
        comparisons = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                diff = self._assignment_difference(
                    population[i].assignment, 
                    population[j].assignment
                )
                diversity_sum += diff
                comparisons += 1
        
        return diversity_sum / comparisons if comparisons > 0 else 0.0
    
    def _assignment_difference(self, assignment1: Dict[str, List[str]], 
                             assignment2: Dict[str, List[str]]) -> float:
        """Calculate difference between two assignments"""
        total_differences = 0
        total_shipments = sum(len(shipments) for shipments in assignment1.values())
        
        if total_shipments == 0:
            return 0.0
        
        for driver_id in self.driver_ids:
            set1 = set(assignment1.get(driver_id, []))
            set2 = set(assignment2.get(driver_id, []))
            
            # Count shipments that are different
            differences = len(set1.symmetric_difference(set2))
            total_differences += differences
        
        return total_differences / (2 * total_shipments)  # Normalize

# Example usage and testing
if __name__ == "__main__":
    # Mock data for testing
    from ai_dispatch_engine import Shipment, Driver
    
    # Create test data
    shipments = [
        Shipment("S1", 15.5007, 32.5599, 15.5527, 32.5342),
        Shipment("S2", 15.6031, 32.5298, 15.5877, 32.5439),
        Shipment("S3", 15.4875, 32.5456, 15.5123, 32.5678),
        Shipment("S4", 15.5200, 32.5400, 15.5300, 32.5500),
    ]
    
    drivers = [
        Driver("D1", 50),
        Driver("D2", 40),
    ]
    
    distance_matrix = {
        "S1": 12.5,
        "S2": 8.3,
        "S3": 15.7,
        "S4": 6.2,
    }
    
    # Test genetic optimizer
    optimizer = GeneticOptimizer()
    optimizer.setup(shipments, drivers, distance_matrix, population_size=20, generations=50)
    
    best_assignment, history = optimizer.evolve()
    
    print("Best assignment:")
    for driver_id, shipments in best_assignment.items():
        print(f"  {driver_id}: {shipments}")
    
    print(f"\nEvolution completed in {len(history)} generations")
    print(f"Final fitness: {history[-1]['best_fitness']:.2f}")
    print(f"Final diversity: {history[-1]['diversity']:.3f}")