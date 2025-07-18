# 🚀 AI Dispatch Engine for Khartoum

An intelligent shipment dispatch system that uses genetic algorithms and machine learning to optimize delivery assignments in Khartoum, Sudan.

## 🎯 Features

- **Smart AI Optimization**: Uses genetic algorithms to find optimal shipment-to-driver assignments
- **Real-World Routing**: Integrates with OSRM for accurate distance calculations using Khartoum road network
- **Learning System**: Tracks performance and improves over time
- **Interactive Visualizations**: Maps and dashboards showing assignments and performance
- **Geographic Intelligence**: Clusters shipments by location for efficiency

## 📊 What the AI Does

The AI doesn't just assign randomly - it:
- **Evolves Solutions**: Tries thousands of assignment combinations and keeps the best ones
- **Multi-Objective Optimization**: Balances distance minimization, workload distribution, and geographic clustering
- **Learns Patterns**: Remembers successful strategies and improves over time
- **Real-Time Adaptation**: Adjusts to different shipment distributions automatically

## 🏗️ Project Structure

```
ai-dispatch-khartoum/
├── ai_dispatch_engine.py      # Main AI engine
├── osrm_client.py            # OSRM integration for real routing
├── genetic_optimizer.py      # Genetic algorithm implementation
├── performance_monitor.py    # Learning progress tracker
├── visualizer.py            # Maps and charts creator
├── data/
│   ├── sample_shipments.json # Sample shipment data
│   └── sample_drivers.json   # Sample driver data
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Python 3.8+**
- **OSRM Server** running locally with Khartoum data
- **Git** (optional)

### 2. Install Dependencies
```bash
# Clone or download the project
git clone <repository-url>
cd ai-dispatch-khartoum

# Install Python packages
pip install -r requirements.txt
```

### 3. OSRM Setup
Make sure you have OSRM running locally with Khartoum map data:
```bash
# Test OSRM connection
curl "http://localhost:5000/route/v1/driving/32.5599,15.5007;32.5342,15.5527"
```

### 4. Verify Installation
```bash
python -c "import numpy, pandas, folium, deap; print('All dependencies installed!')"
```

## 🚀 How to Run

### Quick Start
```bash
python ai_dispatch_engine.py
```

### Step-by-Step Usage

1. **Prepare Your Data**
   - Update `data/sample_shipments.json` with your actual shipments
   - Update `data/sample_drivers.json` with your driver capacities

2. **Run the AI Engine**
   ```python
   from ai_dispatch_engine import AIDispatchEngine
   
   # Initialize engine
   engine = AIDispatchEngine(osrm_url="http://localhost:5000")
   
   # Load your data
   engine.load_data('data/sample_shipments.json', 'data/sample_drivers.json')
   
   # Calculate real-world distances
   engine.calculate_distance_matrix()
   
   # Run AI optimization
   result = engine.run_optimization(population_size=100, generations=300)
   
   # View results
   print("Optimal Assignment:")
   for driver, shipments in result['optimal_assignment'].items():
       print(f"{driver}: {len(shipments)} shipments")
   ```

3. **View Visualizations**
   - **Interactive Map**: `ai_dispatch_map.html`
   - **Performance Dashboard**: `ai_dispatch_dashboard.png`
   - **Interactive Dashboard**: `ai_dispatch_interactive_dashboard.html`

## 📋 Data Format

### Shipments Format (`shipments.json`)
```json
[
  {
    "id": "S001",
    "pickup_lat": 15.5007,
    "pickup_lon": 32.5599,
    "delivery_lat": 15.5527,
    "delivery_lon": 32.5342
  }
]
```

### Drivers Format (`drivers.json`)
```json
[
  {
    "id": "D001",
    "capacity": 50
  }
]
```

## 🧠 AI Intelligence Features

### Genetic Algorithm Evolution
- **Population**: 100+ different assignment solutions
- **Generations**: 300-500 evolution cycles
- **Selection**: Tournament selection for best performers
- **Crossover**: Combines successful assignment patterns
- **Mutation**: Explores new solution variations

### Multi-Objective Optimization
- **Distance Weight (60%)**: Minimize total travel distance
- **Balance Weight (25%)**: Distribute workload evenly
- **Clustering Weight (15%)**: Group nearby shipments

### Learning System
- **Pattern Memory**: Remembers successful assignment strategies
- **Performance Tracking**: Monitors improvement over multiple runs
- **Adaptive Parameters**: Self-tunes for better performance

## 📈 Performance Monitoring

The AI tracks its own learning:
- **Distance Optimization**: How much better than random assignment
- **Convergence Speed**: How quickly it finds optimal solutions
- **Workload Balance**: How evenly work is distributed
- **Learning Trends**: Improvement over multiple runs

## 🎨 Visualizations

### Interactive Map (`ai_dispatch_map.html`)
- Shows pickup/delivery points for each driver
- Color-coded routes and assignments
- Layer controls to show/hide drivers
- Popup information for each location

### Performance Dashboard
- Fitness evolution charts
- Population diversity graphs
- Convergence analysis
- Performance metrics summary

## ⚙️ Configuration

You can customize the AI behavior:

```python
# In ai_dispatch_engine.py, modify these parameters:
engine.run_optimization(
    population_size=100,    # Number of solutions to evolve
    generations=500         # Number of evolution cycles
)

# In genetic_optimizer.py, adjust weights:
self.distance_weight = 0.6    # Priority for distance minimization
self.balance_weight = 0.25    # Priority for workload balance
self.clustering_weight = 0.15  # Priority for geographic clustering
```

## 🧪 Testing

Run with sample data:
```bash
python ai_dispatch_engine.py
```

Expected output:
- Total distance optimization
- Balanced driver assignments
- Interactive map and charts
- Performance improvement metrics

## 📊 Example Results

```
🚀 AI Dispatch Engine for Khartoum
==================================================
📊 Loading data...
🗺️  Calculating distance matrix...
🧠 Running AI optimization...

✅ Optimization Complete!
Total Distance: 445.2 km
Workload Balance: 0.892
Improvement over Random: 23.4%
Processing Time: 28.5 seconds

📋 Optimal Assignment:
  D001: 4 shipments
  D002: 4 shipments  
  D003: 4 shipments
  D004: 4 shipments
  D005: 4 shipments

📈 Generating visualizations...
💾 Saving results...
```

## 🐛 Troubleshooting

### OSRM Connection Issues
```python
# Check OSRM status
from osrm_client import OSRMClient
client = OSRMClient("http://localhost:5000")
# Should show "OSRM server connected successfully"
```

### Memory Issues
```python
# Reduce population size for large datasets
result = engine.run_optimization(population_size=50, generations=200)
```

### Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

## 🔄 Next Steps

1. **Add More Data**: Replace sample data with your real shipments
2. **Fine-tune Parameters**: Adjust AI weights for your specific needs
3. **Integrate APIs**: Connect to your dispatch management system
4. **Scale Up**: Test with larger datasets (100+ shipments)
5. **Real-time Mode**: Adapt for live dispatching throughout the day

## 📝 License

This project is open source. Feel free to modify and adapt for your needs.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make improvements
4. Submit a pull request

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments
3. Test with sample data first
4. Verify OSRM connectivity

---

**Built with ❤️ for efficient logistics in Khartoum**