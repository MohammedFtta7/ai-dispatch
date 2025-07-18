"""
OSRM Client for real-world routing calculations
Handles communication with local OSRM server for Khartoum
"""

import requests
import time
import logging
from typing import Dict, List, Tuple, Optional

class OSRMClient:
    def __init__(self, base_url="http://localhost:5000"):
        """Initialize OSRM client"""
        self.base_url = base_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
        
        # Test connection
        if not self._test_connection():
            self.logger.warning("OSRM server connection failed - will use fallback calculations")
            self.connected = False
        else:
            self.connected = True
            self.logger.info("OSRM server connected successfully")
    
    def _test_connection(self):
        """Test if OSRM server is accessible"""
        try:
            # Test with Khartoum coordinates
            test_url = f"{self.base_url}/route/v1/driving/32.5599,15.5007;32.5342,15.5527"
            response = requests.get(test_url, timeout=5)
            return response.status_code == 200
        except Exception as e:
            self.logger.error(f"OSRM connection test failed: {e}")
            return False
    
    def get_route(self, start_lat: float, start_lon: float, 
                  end_lat: float, end_lon: float) -> Dict:
        """Get route information between two points"""
        
        if not self.connected:
            return self._fallback_calculation(start_lat, start_lon, end_lat, end_lon)
        
        try:
            # OSRM expects lon,lat format
            url = f"{self.base_url}/route/v1/driving/{start_lon},{start_lat};{end_lon},{end_lat}"
            
            params = {
                'overview': 'false',
                'geometries': 'geojson',
                'steps': 'false'
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['code'] == 'Ok' and data['routes']:
                    route = data['routes'][0]
                    return {
                        'distance': route['distance'] / 1000,  # Convert to km
                        'duration': route['duration'] / 60,    # Convert to minutes
                        'status': 'success'
                    }
                else:
                    self.logger.warning(f"OSRM routing failed: {data.get('message', 'Unknown error')}")
                    return self._fallback_calculation(start_lat, start_lon, end_lat, end_lon)
            else:
                self.logger.warning(f"OSRM HTTP error: {response.status_code}")
                return self._fallback_calculation(start_lat, start_lon, end_lat, end_lon)
                
        except Exception as e:
            self.logger.warning(f"OSRM request failed: {e}")
            return self._fallback_calculation(start_lat, start_lon, end_lat, end_lon)
    
    def get_distance_matrix(self, coordinates: List[Tuple[float, float]]) -> Dict:
        """Get distance matrix for multiple points"""
        
        if not self.connected or len(coordinates) > 25:  # OSRM limit
            return self._fallback_matrix_calculation(coordinates)
        
        try:
            # Format coordinates for OSRM (lon,lat)
            coord_string = ';'.join([f"{lon},{lat}" for lat, lon in coordinates])
            url = f"{self.base_url}/table/v1/driving/{coord_string}"
            
            params = {
                'annotations': 'distance,duration'
            }
            
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['code'] == 'Ok':
                    distances = data['distances']
                    durations = data['durations']
                    
                    return {
                        'distances': [[d/1000 for d in row] for row in distances],  # Convert to km
                        'durations': [[d/60 for d in row] for row in durations],    # Convert to minutes
                        'status': 'success'
                    }
                else:
                    self.logger.warning(f"OSRM matrix failed: {data.get('message', 'Unknown error')}")
                    return self._fallback_matrix_calculation(coordinates)
            else:
                self.logger.warning(f"OSRM matrix HTTP error: {response.status_code}")
                return self._fallback_matrix_calculation(coordinates)
                
        except Exception as e:
            self.logger.warning(f"OSRM matrix request failed: {e}")
            return self._fallback_matrix_calculation(coordinates)
    
    def _fallback_calculation(self, start_lat: float, start_lon: float, 
                            end_lat: float, end_lon: float) -> Dict:
        """Fallback calculation using Haversine distance"""
        distance = self._haversine_distance(start_lat, start_lon, end_lat, end_lon)
        
        # Estimate duration based on average city speed (25 km/h)
        duration = (distance / 25) * 60  # minutes
        
        return {
            'distance': distance,
            'duration': duration,
            'status': 'fallback'
        }
    
    def _fallback_matrix_calculation(self, coordinates: List[Tuple[float, float]]) -> Dict:
        """Fallback matrix calculation using Haversine distance"""
        n = len(coordinates)
        distances = [[0.0] * n for _ in range(n)]
        durations = [[0.0] * n for _ in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    lat1, lon1 = coordinates[i]
                    lat2, lon2 = coordinates[j]
                    dist = self._haversine_distance(lat1, lon1, lat2, lon2)
                    distances[i][j] = dist
                    durations[i][j] = (dist / 25) * 60  # Assume 25 km/h
        
        return {
            'distances': distances,
            'durations': durations,
            'status': 'fallback'
        }
    
    def _haversine_distance(self, lat1: float, lon1: float, 
                          lat2: float, lon2: float) -> float:
        """Calculate great circle distance between two points"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        
        # Radius of earth in kilometers
        r = 6371
        return c * r
    
    def batch_routes(self, route_requests: List[Tuple[float, float, float, float]], 
                    batch_size: int = 10, delay: float = 0.1) -> List[Dict]:
        """Process multiple route requests in batches"""
        results = []
        
        for i in range(0, len(route_requests), batch_size):
            batch = route_requests[i:i + batch_size]
            
            batch_results = []
            for start_lat, start_lon, end_lat, end_lon in batch:
                result = self.get_route(start_lat, start_lon, end_lat, end_lon)
                batch_results.append(result)
                
                # Small delay to avoid overwhelming server
                if delay > 0:
                    time.sleep(delay)
            
            results.extend(batch_results)
            
            if i + batch_size < len(route_requests):
                self.logger.info(f"Processed {i + batch_size}/{len(route_requests)} routes")
        
        return results
    
    def get_nearest_road(self, lat: float, lon: float) -> Dict:
        """Find nearest road point to given coordinates"""
        
        if not self.connected:
            return {
                'latitude': lat,
                'longitude': lon,
                'status': 'fallback'
            }
        
        try:
            url = f"{self.base_url}/nearest/v1/driving/{lon},{lat}"
            
            response = requests.get(url, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                
                if data['code'] == 'Ok' and data['waypoints']:
                    waypoint = data['waypoints'][0]
                    location = waypoint['location']
                    
                    return {
                        'latitude': location[1],
                        'longitude': location[0],
                        'distance_to_road': waypoint.get('distance', 0),
                        'status': 'success'
                    }
                else:
                    return {
                        'latitude': lat,
                        'longitude': lon,
                        'status': 'fallback'
                    }
            else:
                return {
                    'latitude': lat,
                    'longitude': lon,
                    'status': 'fallback'
                }
                
        except Exception as e:
            self.logger.warning(f"OSRM nearest request failed: {e}")
            return {
                'latitude': lat,
                'longitude': lon,
                'status': 'fallback'
            }

# Example usage and testing
if __name__ == "__main__":
    # Test the OSRM client
    client = OSRMClient()
    
    # Test single route (sample Khartoum coordinates)
    print("Testing single route...")
    result = client.get_route(15.5007, 32.5599, 15.5527, 32.5342)
    print(f"Route result: {result}")
    
    # Test distance matrix
    print("\nTesting distance matrix...")
    coords = [
        (15.5007, 32.5599),  # Khartoum center
        (15.5527, 32.5342),  # Khartoum North
        (15.4875, 32.5456),  # Omdurman
    ]
    matrix_result = client.get_distance_matrix(coords)
    print(f"Matrix result status: {matrix_result['status']}")
    
    # Test nearest road
    print("\nTesting nearest road...")
    nearest = client.get_nearest_road(15.5007, 32.5599)
    print(f"Nearest road: {nearest}")
    
    print("OSRM client test completed!")