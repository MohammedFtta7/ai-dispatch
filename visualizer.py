"""
Visualizer for AI Dispatch Engine
Creates interactive maps and performance charts
"""

import folium
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

class DispatchVisualizer:
    def __init__(self):
        """Initialize visualizer"""
        self.logger = logging.getLogger(__name__)
        
        # Khartoum center coordinates
        self.khartoum_center = [15.5007, 32.5599]
        
        # Color palette for drivers
        self.driver_colors = [
            'red', 'blue', 'green', 'orange', 'purple', 
            'darkred', 'lightblue', 'darkgreen', 'pink', 'gray',
            'black', 'darkblue', 'lightgreen', 'cadetblue', 'darkpurple'
        ]
        
        # Set plot style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def create_assignment_map(self, shipments, assignment: Dict[str, List[str]], 
                            save_html: bool = True) -> folium.Map:
        """Create interactive map showing shipment assignments"""
        self.logger.info("Creating assignment visualization map...")
        
        # Create base map centered on Khartoum
        m = folium.Map(
            location=self.khartoum_center,
            zoom_start=11,
            tiles='OpenStreetMap'
        )
        
        # Add title
        title_html = '''
        <h3 align="center" style="font-size:20px"><b>AI Dispatch - Optimal Assignment</b></h3>
        '''
        m.get_root().html.add_child(folium.Element(title_html))
        
        # Create driver groups for layer control
        driver_groups = {}
        
        for i, (driver_id, shipment_ids) in enumerate(assignment.items()):
            color = self.driver_colors[i % len(self.driver_colors)]
            
            # Create feature group for this driver
            driver_group = folium.FeatureGroup(name=f'Driver {driver_id} ({len(shipment_ids)} shipments)')
            
            for shipment_id in shipment_ids:
                # Find shipment details
                shipment = next((s for s in shipments if s.id == shipment_id), None)
                if not shipment:
                    continue
                
                # Add pickup marker
                pickup_popup = folium.Popup(
                    f"""
                    <b>Driver {driver_id}</b><br>
                    Shipment: {shipment_id}<br>
                    Type: Pickup<br>
                    Location: {shipment.pickup_lat:.4f}, {shipment.pickup_lon:.4f}
                    """,
                    max_width=200
                )
                
                folium.Marker(
                    [shipment.pickup_lat, shipment.pickup_lon],
                    popup=pickup_popup,
                    icon=folium.Icon(color=color, icon='play', prefix='fa'),
                    tooltip=f"Driver {driver_id} - Pickup {shipment_id}"
                ).add_to(driver_group)
                
                # Add delivery marker
                delivery_popup = folium.Popup(
                    f"""
                    <b>Driver {driver_id}</b><br>
                    Shipment: {shipment_id}<br>
                    Type: Delivery<br>
                    Location: {shipment.delivery_lat:.4f}, {shipment.delivery_lon:.4f}
                    """,
                    max_width=200
                )
                
                folium.Marker(
                    [shipment.delivery_lat, shipment.delivery_lon],
                    popup=delivery_popup,
                    icon=folium.Icon(color=color, icon='stop', prefix='fa'),
                    tooltip=f"Driver {driver_id} - Delivery {shipment_id}"
                ).add_to(driver_group)
                
                # Add route line
                folium.PolyLine(
                    [[shipment.pickup_lat, shipment.pickup_lon], 
                     [shipment.delivery_lat, shipment.delivery_lon]],
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"Driver {driver_id} - Route {shipment_id}"
                ).add_to(driver_group)
            
            # Add driver group to map
            driver_group.add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add legend
        legend_html = self._create_map_legend(assignment)
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Save to HTML file
        if save_html:
            filename = "ai_dispatch_map.html"
            m.save(filename)
            self.logger.info(f"Interactive map saved to {filename}")
        
        return m
    
    def _create_map_legend(self, assignment: Dict[str, List[str]]) -> str:
        """Create HTML legend for the map"""
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Driver Assignments</h4>
        '''
        
        for i, (driver_id, shipment_ids) in enumerate(assignment.items()):
            color = self.driver_colors[i % len(self.driver_colors)]
            legend_html += f'''
            <p><span style="color:{color};">●</span> Driver {driver_id}: {len(shipment_ids)} shipments</p>
            '''
        
        legend_html += '''
        <p><i class="fa fa-play" style="color:black;"></i> Pickup Point</p>
        <p><i class="fa fa-stop" style="color:black;"></i> Delivery Point</p>
        </div>
        '''
        
        return legend_html
    
    def create_performance_dashboard(self, optimization_history: List[Dict], 
                                   performance_metrics: Dict[str, Any]) -> plt.Figure:
        """Create comprehensive performance dashboard"""
        self.logger.info("Creating performance dashboard...")
        
        # Convert history to DataFrame
        df_history = pd.DataFrame(optimization_history)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('AI Dispatch Engine - Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Fitness Evolution
        axes[0, 0].plot(df_history['generation'], df_history['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
        axes[0, 0].plot(df_history['generation'], df_history['average_fitness'], 'r--', alpha=0.7, label='Average Fitness')
        axes[0, 0].set_title('Fitness Evolution', fontweight='bold')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness Score')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Population Diversity
        axes[0, 1].plot(df_history['generation'], df_history['diversity'], 'g-', linewidth=2)
        axes[0, 1].set_title('Population Diversity', fontweight='bold')
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Diversity Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Convergence Analysis
        axes[0, 2].plot(df_history['generation'], df_history['convergence_rate'], 'orange', linewidth=2)
        axes[0, 2].set_title('Convergence Progress', fontweight='bold')
        axes[0, 2].set_xlabel('Generation')
        axes[0, 2].set_ylabel('Generations Without Improvement')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Performance Metrics Summary
        metrics = ['Total Distance', 'Workload Balance', 'Geographic Efficiency', 'Improvement']
        values = [
            performance_metrics['total_distance'],
            performance_metrics['workload_balance'] * 100,  # Convert to percentage
            performance_metrics['geographic_efficiency'] * 100,
            performance_metrics['improvement_over_random']
        ]
        colors = ['red', 'green', 'blue', 'orange']
        
        bars = axes[1, 0].bar(metrics, values, color=colors, alpha=0.7)
        axes[1, 0].set_title('Performance Metrics', fontweight='bold')
        axes[1, 0].set_ylabel('Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        # 5. Driver Utilization
        if 'driver_utilization' in performance_metrics:
            utilization = performance_metrics['driver_utilization']
            driver_ids = [f'Driver {i+1}' for i in range(len(utilization))]
            
            axes[1, 1].bar(driver_ids, utilization, color='skyblue', alpha=0.7)
            axes[1, 1].set_title('Driver Workload Distribution', fontweight='bold')
            axes[1, 1].set_xlabel('Drivers')
            axes[1, 1].set_ylabel('Number of Shipments')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            # Add average line
            avg_utilization = np.mean(utilization)
            axes[1, 1].axhline(y=avg_utilization, color='red', linestyle='--', 
                              label=f'Average: {avg_utilization:.1f}')
            axes[1, 1].legend()
        
        # 6. Algorithm Performance Summary
        total_generations = len(df_history)
        final_fitness = df_history['best_fitness'].iloc[-1]
        improvement_rate = (df_history['best_fitness'].iloc[0] - final_fitness) / df_history['best_fitness'].iloc[0] * 100
        
        summary_text = f"""
        Algorithm Performance Summary:
        
        • Total Generations: {total_generations}
        • Final Fitness: {final_fitness:.2f}
        • Improvement Rate: {improvement_rate:.1f}%
        • Convergence: {'Yes' if df_history['convergence_rate'].iloc[-1] > 10 else 'No'}
        
        Optimization Quality:
        • Distance Reduction: {performance_metrics['improvement_over_random']:.1f}%
        • Workload Balance: {performance_metrics['workload_balance']:.3f}
        • Geographic Efficiency: {performance_metrics['geographic_efficiency']:.3f}
        """
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                        fontsize=10, verticalalignment='top', 
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Summary Report', fontweight='bold')
        
        plt.tight_layout()
        
        # Save dashboard
        filename = "ai_dispatch_dashboard.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"Performance dashboard saved to {filename}")
        
        plt.show()
        return fig
    
    def create_interactive_dashboard(self, optimization_history: List[Dict], 
                                   performance_metrics: Dict[str, Any]) -> go.Figure:
        """Create interactive dashboard using Plotly"""
        self.logger.info("Creating interactive dashboard...")
        
        df_history = pd.DataFrame(optimization_history)
        
        # Create subplot figure
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Fitness Evolution', 'Population Diversity', 'Convergence Analysis',
                           'Performance Metrics', 'Driver Utilization', 'Improvement Timeline'),
            specs=[[{"secondary_y": True}, {}, {}],
                   [{"type": "bar"}, {"type": "bar"}, {}]]
        )
        
        # 1. Fitness Evolution
        fig.add_trace(
            go.Scatter(x=df_history['generation'], y=df_history['best_fitness'],
                      mode='lines', name='Best Fitness', line=dict(color='blue', width=3)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df_history['generation'], y=df_history['average_fitness'],
                      mode='lines', name='Average Fitness', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # 2. Population Diversity
        fig.add_trace(
            go.Scatter(x=df_history['generation'], y=df_history['diversity'],
                      mode='lines', name='Diversity', line=dict(color='green', width=2)),
            row=1, col=2
        )
        
        # 3. Convergence Analysis
        fig.add_trace(
            go.Scatter(x=df_history['generation'], y=df_history['convergence_rate'],
                      mode='lines', name='Convergence Rate', line=dict(color='orange', width=2)),
            row=1, col=3
        )
        
        # 4. Performance Metrics
        metrics = ['Distance', 'Balance', 'Efficiency', 'Improvement']
        metric_values = [
            performance_metrics['total_distance'],
            performance_metrics['workload_balance'] * 100,
            performance_metrics['geographic_efficiency'] * 100,
            performance_metrics['improvement_over_random']
        ]
        
        fig.add_trace(
            go.Bar(x=metrics, y=metric_values, name='Metrics',
                  marker_color=['red', 'green', 'blue', 'orange']),
            row=2, col=1
        )
        
        # 5. Driver Utilization
        if 'driver_utilization' in performance_metrics:
            utilization = performance_metrics['driver_utilization']
            driver_names = [f'Driver {i+1}' for i in range(len(utilization))]
            
            fig.add_trace(
                go.Bar(x=driver_names, y=utilization, name='Workload',
                      marker_color='skyblue'),
                row=2, col=2
            )
        
        # 6. Improvement Timeline
        if len(df_history) > 1:
            improvement_over_time = []
            initial_fitness = df_history['best_fitness'].iloc[0]
            for fitness in df_history['best_fitness']:
                improvement = ((initial_fitness - fitness) / initial_fitness) * 100
                improvement_over_time.append(improvement)
            
            fig.add_trace(
                go.Scatter(x=df_history['generation'], y=improvement_over_time,
                          mode='lines+markers', name='Cumulative Improvement',
                          line=dict(color='purple', width=2)),
                row=2, col=3
            )
        
        # Update layout
        fig.update_layout(
            title_text="AI Dispatch Engine - Interactive Performance Dashboard",
            title_x=0.5,
            showlegend=False,
            height=800
        )
        
        # Save interactive dashboard
        filename = "ai_dispatch_interactive_dashboard.html"
        fig.write_html(filename)
        self.logger.info(f"Interactive dashboard saved to {filename}")
        
        fig.show()
        return fig
    
    def create_route_animation(self, shipments, assignment: Dict[str, List[str]]) -> folium.Map:
        """Create animated route visualization (simplified)"""
        self.logger.info("Creating route animation...")
        
        # Create base map
        m = folium.Map(location=self.khartoum_center, zoom_start=11)
        
        # Add animated routes for each driver
        for i, (driver_id, shipment_ids) in enumerate(assignment.items()):
            color = self.driver_colors[i % len(self.driver_colors)]
            
            # Create route coordinates
            route_coords = []
            for shipment_id in shipment_ids:
                shipment = next((s for s in shipments if s.id == shipment_id), None)
                if shipment:
                    route_coords.extend([
                        [shipment.pickup_lat, shipment.pickup_lon],
                        [shipment.delivery_lat, shipment.delivery_lon]
                    ])
            
            if route_coords:
                # Add animated route
                folium.PolyLine(
                    route_coords,
                    color=color,
                    weight=4,
                    opacity=0.8,
                    popup=f"Driver {driver_id} Route"
                ).add_to(m)
        
        # Save animated map
        filename = "ai_dispatch_routes_animated.html"
        m.save(filename)
        self.logger.info(f"Route animation saved to {filename}")
        
        return m
    
    def create_comparison_chart(self, results_comparison: Dict[str, Dict]) -> plt.Figure:
        """Create comparison chart between different algorithms"""
        self.logger.info("Creating algorithm comparison chart...")
        
        algorithms = list(results_comparison.keys())
        metrics = ['total_distance', 'balance_score', 'processing_time']
        metric_labels = ['Total Distance (km)', 'Balance Score', 'Processing Time (s)']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold')
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            values = [results_comparison[alg][metric] for alg in algorithms]
            colors = ['red', 'orange', 'green'] if len(algorithms) == 3 else plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
            
            bars = axes[i].bar(algorithms, values, color=colors, alpha=0.7)
            axes[i].set_title(label, fontweight='bold')
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[i].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                           f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save comparison chart
        filename = "algorithm_comparison.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        self.logger.info(f"Comparison chart saved to {filename}")
        
        plt.show()
        return fig

# Example usage and testing
if __name__ == "__main__":
    from ai_dispatch_engine import Shipment
    
    # Create test data
    shipments = [
        Shipment("S1", 15.5007, 32.5599, 15.5527, 32.5342),
        Shipment("S2", 15.6031, 32.5298, 15.5877, 32.5439),
        Shipment("S3", 15.4875, 32.5456, 15.5123, 32.5678),
        Shipment("S4", 15.5200, 32.5400, 15.5300, 32.5500),
    ]
    
    assignment = {
        "D1": ["S1", "S3"],
        "D2": ["S2", "S4"]
    }
    
    # Mock optimization history
    history = [
        {'generation': i, 'best_fitness': 1000 - i*5, 'average_fitness': 1200 - i*3, 
         'diversity': 0.8 - i*0.01, 'convergence_rate': min(i, 50)}
        for i in range(100)
    ]
    
    # Mock performance metrics
    metrics = {
        'total_distance': 450.5,
        'workload_balance': 0.85,
        'geographic_efficiency': 0.78,
        'improvement_over_random': 23.5,
        'driver_utilization': [2, 2]
    }
    
    # Test visualizer
    visualizer = DispatchVisualizer()
    
    # Create map
    map_viz = visualizer.create_assignment_map(shipments, assignment)
    
    # Create dashboard
    dashboard = visualizer.create_performance_dashboard(history, metrics)
    
    # Create interactive dashboard
    interactive_dashboard = visualizer.create_interactive_dashboard(history, metrics)
    
    print("Visualizer test completed!")