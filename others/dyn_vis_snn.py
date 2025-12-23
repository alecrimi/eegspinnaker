import nest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict, deque

class DynamicSpikingNNVisualizer:
    """
    Dynamic real-time visualizer for PyNEST spiking neural networks.
    Shows spikes as they occur with visual effects and propagation.
    """
    
    def __init__(self, figsize=(18, 10)):
        """Initialize the visualizer with figure setup."""
        self.fig = plt.figure(figsize=figsize)
        self.spike_data = defaultdict(list)
        self.neuron_positions = {}
        self.connections = []
        self.neuron_types = {}
        self.weights = {}
        
        # For dynamic visualization
        self.spike_decay_time = 50.0  # ms - how long spike glow lasts
        self.propagation_speed = 0.5  # Speed of spike propagation visualization
        
    def register_neurons(self, neuron_gids, positions=None, neuron_type='excitatory'):
        """
        Register neurons with their positions and types.
        
        Args:
            neuron_gids: List of neuron global IDs
            positions: Dict or array of 3D positions {gid: (x, y, z)}
            neuron_type: 'excitatory', 'inhibitory', or 'input'
        """
        if positions is None:
            # Auto-generate 3D positions in a cube
            n = len(neuron_gids)
            side = int(np.ceil(n ** (1/3)))
            positions = {}
            for idx, gid in enumerate(neuron_gids):
                x = (idx % side) / side
                y = ((idx // side) % side) / side
                z = (idx // (side * side)) / side
                positions[gid] = (x, y, z)
        
        for gid in neuron_gids:
            self.neuron_positions[gid] = positions.get(gid, (0, 0, 0))
            self.neuron_types[gid] = neuron_type
    
    def register_connections(self, source_gids, target_gids, weights=None):
        """
        Register connections between neurons.
        
        Args:
            source_gids: List of source neuron IDs
            target_gids: List of target neuron IDs
            weights: Optional list of connection weights
        """
        for i, (src, tgt) in enumerate(zip(source_gids, target_gids)):
            self.connections.append((src, tgt))
            if weights is not None:
                self.weights[(src, tgt)] = weights[i]
    
    def record_spikes(self, spike_recorder):
        """
        Extract spike data from a NEST spike recorder.
        
        Args:
            spike_recorder: NEST spike recorder device
        """
        events = nest.GetStatus(spike_recorder, 'events')[0]
        times = events['times']
        senders = events['senders']
        
        for t, gid in zip(times, senders):
            self.spike_data[gid].append(t)
    
    def get_neuron_activity(self, gid, current_time, decay_time):
        """
        Calculate neuron activity level based on recent spikes.
        Returns value between 0 and 1.
        """
        if gid not in self.spike_data or not self.spike_data[gid]:
            return 0.0
        
        activity = 0.0
        for spike_time in self.spike_data[gid]:
            time_diff = current_time - spike_time
            if 0 <= time_diff <= decay_time:
                # Exponential decay
                activity = max(activity, np.exp(-time_diff / (decay_time / 3)))
        
        return min(activity, 1.0)
    
    def get_active_connections(self, current_time, window=5.0):
        """
        Get connections that should show spike propagation.
        Returns list of (src, tgt, progress) where progress is 0-1.
        """
        active_conns = []
        
        for src, tgt in self.connections:
            if src not in self.spike_data:
                continue
                
            for spike_time in self.spike_data[src]:
                time_diff = current_time - spike_time
                if 0 <= time_diff <= window:
                    progress = time_diff / window
                    active_conns.append((src, tgt, progress))
        
        return active_conns
    
    def create_movie(self, duration=None, fps=30, time_step=1.0, 
                     filename='spiking_network.mp4', view_angle=(30, 45)):
        """
        Create an animated movie of network activity.
        
        Args:
            duration: Simulation duration to visualize (None = all data)
            fps: Frames per second
            time_step: Time step between frames (ms)
            filename: Output filename
            view_angle: (elevation, azimuth) for 3D view
        """
        # Get time range
        all_spikes = [t for times in self.spike_data.values() for t in times]
        if not all_spikes:
            print("No spike data to animate")
            return
        
        t_min, t_max = min(all_spikes), max(all_spikes)
        if duration:
            t_max = min(t_max, t_min + duration)
        
        time_points = np.arange(t_min, t_max, time_step)
        n_frames = len(time_points)
        
        print(f"Creating movie with {n_frames} frames ({t_min:.1f} to {t_max:.1f} ms)...")
        
        # Setup figure with subplots
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        
        # Main 3D network view
        ax_3d = self.fig.add_subplot(gs[0, :], projection='3d')
        
        # Raster plot
        ax_raster = self.fig.add_subplot(gs[1, 0])
        
        # Activity meter
        ax_activity = self.fig.add_subplot(gs[1, 1])
        
        # Set 3D view angle
        ax_3d.view_init(elev=view_angle[0], azim=view_angle[1])
        
        # Activity history for plotting
        activity_history = deque(maxlen=200)
        time_history = deque(maxlen=200)
        
        def update(frame):
            current_time = time_points[frame]
            
            # Clear axes
            ax_3d.clear()
            ax_raster.clear()
            ax_activity.clear()
            
            # ===== 3D Network View =====
            # Plot static connections (dim)
            for src, tgt in self.connections:
                if src in self.neuron_positions and tgt in self.neuron_positions:
                    pos_src = self.neuron_positions[src]
                    pos_tgt = self.neuron_positions[tgt]
                    
                    ax_3d.plot3D([pos_src[0], pos_tgt[0]], 
                               [pos_src[1], pos_tgt[1]], 
                               [pos_src[2], pos_tgt[2]], 
                               color='gray', alpha=0.05, linewidth=0.5, zorder=1)
            
            # Plot active connections (spike propagation)
            active_conns = self.get_active_connections(current_time, window=10.0)
            for src, tgt, progress in active_conns:
                pos_src = np.array(self.neuron_positions[src])
                pos_tgt = np.array(self.neuron_positions[tgt])
                
                # Interpolate position along connection
                pos_current = pos_src + progress * (pos_tgt - pos_src)
                
                # Draw propagating spike
                weight = self.weights.get((src, tgt), 1.0)
                color = 'lime' if weight > 0 else 'orangered'
                alpha = 1.0 - progress
                
                # Draw the connection with glow
                ax_3d.plot3D([pos_src[0], pos_tgt[0]], 
                           [pos_src[1], pos_tgt[1]], 
                           [pos_src[2], pos_tgt[2]], 
                           color=color, alpha=alpha*0.6, linewidth=2, zorder=2)
                
                # Draw moving spike marker
                ax_3d.scatter(pos_current[0], pos_current[1], pos_current[2],
                            c=color, s=50, alpha=alpha, marker='o', 
                            edgecolors='white', linewidth=1, zorder=3)
            
            # Plot neurons with activity-based coloring
            for gid, pos in self.neuron_positions.items():
                ntype = self.neuron_types.get(gid, 'excitatory')
                activity = self.get_neuron_activity(gid, current_time, self.spike_decay_time)
                
                # Base colors
                if ntype == 'excitatory':
                    base_color = np.array([0.2, 0.4, 0.8])  # Blue
                elif ntype == 'inhibitory':
                    base_color = np.array([0.8, 0.2, 0.2])  # Red
                else:
                    base_color = np.array([0.2, 0.8, 0.2])  # Green
                
                # Brighten based on activity
                active_color = base_color + activity * (np.array([1.0, 1.0, 0.3]) - base_color)
                active_color = np.clip(active_color, 0, 1)
                
                size = 50 + activity * 150  # Pulse size
                alpha = 0.5 + activity * 0.5
                
                ax_3d.scatter(pos[0], pos[1], pos[2], 
                            c=[active_color], s=size, alpha=alpha, 
                            edgecolors='white', linewidth=1, zorder=4)
            
            ax_3d.set_xlabel('X')
            ax_3d.set_ylabel('Y')
            ax_3d.set_zlabel('Z')
            ax_3d.set_title(f'Network Activity - Time: {current_time:.1f} ms', 
                          fontsize=12, fontweight='bold')
            ax_3d.view_init(elev=view_angle[0], azim=view_angle[1])
            ax_3d.set_xlim([-0.1, 1.1])
            ax_3d.set_ylim([-0.1, 1.1])
            ax_3d.set_zlim([-0.1, 1.1])
            
            # ===== Raster Plot (Recent Activity) =====
            window_size = 100.0  # ms
            for gid, spike_times in self.spike_data.items():
                recent_spikes = [t for t in spike_times 
                               if current_time - window_size <= t <= current_time]
                
                if recent_spikes:
                    ntype = self.neuron_types.get(gid, 'excitatory')
                    color = 'blue' if ntype == 'excitatory' else 'red'
                    
                    # Highlight very recent spikes
                    for t in recent_spikes:
                        alpha = 0.3 + 0.7 * max(0, 1 - (current_time - t) / 20.0)
                        ax_raster.scatter([t], [gid], c=color, s=3, alpha=alpha)
            
            ax_raster.axvline(current_time, color='yellow', linewidth=2, 
                            linestyle='--', alpha=0.7, label='Current Time')
            ax_raster.set_xlim(current_time - window_size, current_time)
            ax_raster.set_xlabel('Time (ms)', fontsize=9)
            ax_raster.set_ylabel('Neuron ID', fontsize=9)
            ax_raster.set_title('Recent Spike Activity', fontsize=10, fontweight='bold')
            ax_raster.legend(loc='upper right', fontsize=8)
            ax_raster.grid(True, alpha=0.3)
            
            # ===== Population Activity Over Time =====
            # Count spikes in small time bins
            bin_size = 5.0
            n_spikes = sum(1 for gid in self.spike_data 
                          for t in self.spike_data[gid] 
                          if current_time - bin_size <= t <= current_time)
            
            firing_rate = n_spikes / (bin_size / 1000.0) / len(self.neuron_positions)
            activity_history.append(firing_rate)
            time_history.append(current_time)
            
            if len(activity_history) > 1:
                ax_activity.plot(list(time_history), list(activity_history), 
                               color='darkblue', linewidth=2)
                ax_activity.fill_between(list(time_history), list(activity_history), 
                                        alpha=0.3, color='blue')
            
            ax_activity.scatter([current_time], [firing_rate], 
                              c='red', s=100, zorder=5, edgecolors='white', linewidth=2)
            ax_activity.set_xlabel('Time (ms)', fontsize=9)
            ax_activity.set_ylabel('Firing Rate (Hz)', fontsize=9)
            ax_activity.set_title('Population Activity', fontsize=10, fontweight='bold')
            ax_activity.grid(True, alpha=0.3)
            ax_activity.set_xlim(max(t_min, current_time - 200), current_time + 10)
            
            # Progress indicator
            progress = (frame + 1) / n_frames * 100
            self.fig.suptitle(f'Spiking Neural Network Visualization | Progress: {progress:.1f}%', 
                            fontsize=14, fontweight='bold')
            
            # Print progress
            if (frame + 1) % max(1, n_frames // 20) == 0:
                print(f"  Frame {frame + 1}/{n_frames} ({progress:.1f}%)")
        
        # Create animation
        anim = FuncAnimation(self.fig, update, frames=n_frames, 
                           interval=1000/fps, repeat=False)
        
        # Save the animation
        print(f"Saving animation to {filename}...")
        if filename.endswith('.gif'):
            writer = PillowWriter(fps=fps)
            anim.save(filename, writer=writer)
        else:
            # For MP4, use ffmpeg writer
            anim.save(filename, writer='ffmpeg', fps=fps, dpi=100)
        print(f"Animation saved successfully!")
        
        return anim
    
    def show_live_animation(self, time_step=1.0, interval=50, view_angle=(30, 45)):
        """
        Show live animation in matplotlib window (not saved to file).
        
        Args:
            time_step: Time step between frames (ms)
            interval: Delay between frames (ms)
            view_angle: (elevation, azimuth) for 3D view
        """
        all_spikes = [t for times in self.spike_data.values() for t in times]
        if not all_spikes:
            print("No spike data to animate")
            return
        
        t_min, t_max = min(all_spikes), max(all_spikes)
        time_points = np.arange(t_min, t_max, time_step)
        
        # Setup figure
        self.fig.clear()
        gs = self.fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.3, wspace=0.3)
        ax_3d = self.fig.add_subplot(gs[0, :], projection='3d')
        ax_raster = self.fig.add_subplot(gs[1, 0])
        ax_activity = self.fig.add_subplot(gs[1, 1])
        
        ax_3d.view_init(elev=view_angle[0], azim=view_angle[1])
        
        activity_history = deque(maxlen=200)
        time_history = deque(maxlen=200)
        
        def update(frame):
            current_time = time_points[frame]
            
            ax_3d.clear()
            ax_raster.clear()
            ax_activity.clear()
            
            # Plot connections
            for src, tgt in self.connections:
                if src in self.neuron_positions and tgt in self.neuron_positions:
                    pos_src = self.neuron_positions[src]
                    pos_tgt = self.neuron_positions[tgt]
                    ax_3d.plot3D([pos_src[0], pos_tgt[0]], 
                               [pos_src[1], pos_tgt[1]], 
                               [pos_src[2], pos_tgt[2]], 
                               color='gray', alpha=0.05, linewidth=0.5)
            
            # Active connections
            active_conns = self.get_active_connections(current_time, window=10.0)
            for src, tgt, progress in active_conns:
                pos_src = np.array(self.neuron_positions[src])
                pos_tgt = np.array(self.neuron_positions[tgt])
                pos_current = pos_src + progress * (pos_tgt - pos_src)
                
                weight = self.weights.get((src, tgt), 1.0)
                color = 'lime' if weight > 0 else 'orangered'
                alpha = 1.0 - progress
                
                ax_3d.plot3D([pos_src[0], pos_tgt[0]], 
                           [pos_src[1], pos_tgt[1]], 
                           [pos_src[2], pos_tgt[2]], 
                           color=color, alpha=alpha*0.6, linewidth=2)
                ax_3d.scatter(pos_current[0], pos_current[1], pos_current[2],
                            c=color, s=50, alpha=alpha, marker='o')
            
            # Neurons
            for gid, pos in self.neuron_positions.items():
                ntype = self.neuron_types.get(gid, 'excitatory')
                activity = self.get_neuron_activity(gid, current_time, self.spike_decay_time)
                
                base_color = np.array([0.2, 0.4, 0.8]) if ntype == 'excitatory' else np.array([0.8, 0.2, 0.2])
                active_color = base_color + activity * (np.array([1.0, 1.0, 0.3]) - base_color)
                active_color = np.clip(active_color, 0, 1)
                
                size = 50 + activity * 150
                alpha = 0.5 + activity * 0.5
                
                ax_3d.scatter(pos[0], pos[1], pos[2], c=[active_color], 
                            s=size, alpha=alpha, edgecolors='white', linewidth=1)
            
            ax_3d.set_title(f'Time: {current_time:.1f} ms', fontsize=12, fontweight='bold')
            ax_3d.view_init(elev=view_angle[0], azim=view_angle[1])
            
            # Raster
            window_size = 100.0
            for gid, spike_times in self.spike_data.items():
                recent_spikes = [t for t in spike_times 
                               if current_time - window_size <= t <= current_time]
                if recent_spikes:
                    color = 'blue' if self.neuron_types.get(gid) == 'excitatory' else 'red'
                    ax_raster.scatter(recent_spikes, [gid]*len(recent_spikes), 
                                    c=color, s=3, alpha=0.6)
            
            ax_raster.axvline(current_time, color='yellow', linewidth=2, linestyle='--')
            ax_raster.set_xlim(current_time - window_size, current_time)
            ax_raster.set_title('Recent Spikes', fontsize=10)
            
            # Activity
            bin_size = 5.0
            n_spikes = sum(1 for gid in self.spike_data 
                          for t in self.spike_data[gid] 
                          if current_time - bin_size <= t <= current_time)
            firing_rate = n_spikes / (bin_size / 1000.0) / len(self.neuron_positions)
            activity_history.append(firing_rate)
            time_history.append(current_time)
            
            if len(activity_history) > 1:
                ax_activity.plot(list(time_history), list(activity_history), 
                               color='darkblue', linewidth=2)
                ax_activity.fill_between(list(time_history), list(activity_history), 
                                        alpha=0.3, color='blue')
            
            ax_activity.set_title('Population Activity', fontsize=10)
            ax_activity.set_xlim(max(t_min, current_time - 200), current_time + 10)
        
        anim = FuncAnimation(self.fig, update, frames=len(time_points), 
                           interval=interval, repeat=True)
        plt.show()
        return anim


# Example simulation
def example_dynamic_simulation():
    """Example simulation with dynamic movie visualization."""
    
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": 0.1})
    
    # Create network
    n_excitatory = 80
    n_inhibitory = 20
    
    excitatory = nest.Create("iaf_psc_alpha", n_excitatory)
    inhibitory = nest.Create("iaf_psc_alpha", n_inhibitory)
    
    poisson = nest.Create("poisson_generator", params={"rate": 5000.0})
    spike_recorder = nest.Create("spike_recorder")
    
    # Connections
    nest.Connect(poisson, excitatory, syn_spec={"weight": 50.0})
    
    nest.Connect(excitatory, excitatory, 
                 conn_spec={"rule": "pairwise_bernoulli", "p": 0.1},
                 syn_spec={"weight": nest.random.uniform(10.0, 50.0)})
    
    nest.Connect(excitatory, inhibitory, 
                 conn_spec={"rule": "pairwise_bernoulli", "p": 0.3},
                 syn_spec={"weight": 40.0})
    
    nest.Connect(inhibitory, excitatory, 
                 conn_spec={"rule": "pairwise_bernoulli", "p": 0.3},
                 syn_spec={"weight": -80.0})
    
    nest.Connect(excitatory + inhibitory, spike_recorder)
    
    # Simulate
    print("Running simulation...")
    nest.Simulate(500.0)
    
    # Create visualizer
    print("\nCreating dynamic visualizer...")
    vis = DynamicSpikingNNVisualizer(figsize=(18, 10))
    
    # Register neurons with 3D positions
    exc_positions = {}
    side = int(np.ceil(np.sqrt(n_excitatory)))
    for i, gid in enumerate(excitatory.tolist()):
        x = (i % side) / side
        y = (i // side) / side
        z = 0.2
        exc_positions[gid] = (x, y, z)
    
    vis.register_neurons(excitatory.tolist(), exc_positions, 'excitatory')
    
    inh_positions = {}
    side_inh = int(np.ceil(np.sqrt(n_inhibitory)))
    for i, gid in enumerate(inhibitory.tolist()):
        x = (i % side_inh) / side_inh
        y = (i // side_inh) / side_inh
        z = 0.8
        inh_positions[gid] = (x, y, z)
    
    vis.register_neurons(inhibitory.tolist(), inh_positions, 'inhibitory')
    
    # Get connections
    connections = nest.GetConnections(source=excitatory + inhibitory, 
                                      target=excitatory + inhibitory)
    sources = connections.get('source')
    targets = connections.get('target')
    weights = connections.get('weight')
    
    vis.register_connections(sources, targets, weights)
    vis.record_spikes(spike_recorder)
    
    # Create movie or show live
    print("\nChoose visualization mode:")
    print("1. Save as movie (MP4/GIF) - slower but saves to file")
    print("2. Show live animation - faster, interactive")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("\nCreating movie (this may take a few minutes)...")
        # Reduced frames for faster generation: 100ms duration, larger time steps
        vis.create_movie(duration=100, fps=15, time_step=1.0, 
                         filename='spiking_network.gif', view_angle=(20, 45))
    else:
        print("\nShowing live animation (close window to exit)...")
        vis.show_live_animation(time_step=2.0, interval=50, view_angle=(20, 45))
    
    return vis


if __name__ == "__main__":
    visualizer = example_dynamic_simulation()
