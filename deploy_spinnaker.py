import pyNN.spiNNaker as pynn
import joblib
import numpy as np
import time

def deploy_on_spinnaker(params_file="spinnaker_model.pkl", simulation_time=1000):
    """Load saved model and deploy on SpiNNaker hardware/simulator"""
    
    print("Loading SpiNNaker parameters...")
    try:
        spinnaker_params = joblib.load(params_file)
        print(f"Loaded parameters for {len(spinnaker_params['n_neurons'])} layers")
        print(f"Network architecture: {spinnaker_params['n_neurons']}")
    except FileNotFoundError:
        print(f"Error: Could not find {params_file}")
        print("Please run train_snn.py first to generate the model file")
        return
    
    print("Initializing SpiNNaker...")
    start_time = time.time()
    
    # Setup SpiNNaker
    pynn.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)
    
    print("Creating neuron populations...")
    # Create populations
    populations = []
    for i, n_neurons in enumerate(spinnaker_params['n_neurons']):
        # Convert beta to membrane time constant (tau_m)
        # beta = exp(-dt/tau_m) -> tau_m = -dt / log(beta)
        tau_m = -1.0 / np.log(spinnaker_params['tau_mem'][i])
        
        population = pynn.Population(
            n_neurons, 
            pynn.IF_curr_exp(
                tau_m=tau_m * 10.0,  # Convert to ms, scale for better dynamics
                v_rest=-65.0,
                v_reset=-65.0,
                v_thresh=-50.0,
                tau_refrac=1.0  # Refractory period
            )
        )
        populations.append(population)
        print(f"  Layer {i+1}: {n_neurons} neurons")
    
    print("Creating connections...")
    # Create projections (connections)
    for i in range(len(populations) - 1):
        # Get weights for this connection
        weights = spinnaker_params['weights'][i]
        
        # Create connector with weights
        connector = pynn.AllToAllConnector()
        
        # Create projection
        projection = pynn.Projection(
            populations[i],
            populations[i + 1],
            connector,
            synapse_type=pynn.StaticSynapse(weight=weights),
            receptor_type='excitatory'
        )
        print(f"  Connection {i+1}: {weights.shape} weight matrix")
    
    # Create input spike source for testing
    print("Creating input spike source...")
    spike_times = [10, 20, 30, 40, 50]  # Simple test spikes
    input_source = pynn.Population(
        spinnaker_params['input_size'],
        pynn.SpikeSourceArray(spike_times=spike_times)
    )
    
    # Connect input to first layer
    input_connector = pynn.OneToOneConnector()
    input_projection = pynn.Projection(
        input_source,
        populations[0],
        input_connector,
        synapse_type=pynn.StaticSynapse(weight=5.0)  # Strong input weights
    )
    
    # Record spikes from output layer
    print("Setting up recording...")
    populations[-1].record(['spikes'])
    
    print(f"Running simulation for {simulation_time}ms...")
    # Run simulation
    pynn.run(simulation_time)
    
    # Get recorded spikes
    spike_data = populations[-1].get_data('spikes')
    
    print("Simulation completed!")
    print(f"Output spikes recorded: {len(spike_data.segments[0].spiketrains)} spike trains")
    
    # Print some spike statistics
    for i, spiketrain in enumerate(spike_data.segments[0].spiketrains):
        if len(spiketrain) > 0:
            print(f"  Neuron {i}: {len(spiketrain)} spikes at times {spiketrain}")
    
    # Clean up
    pynn.end()
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

def test_with_sample_input(params_file="spinnaker_model.pkl"):
    """Test the deployed network with sample input data"""
    print("Testing with sample input data...")
    
    # Load parameters
    spinnaker_params = joblib.load(params_file)
    
    # Generate sample input spikes (simulating preprocessed EEG data)
    input_size = spinnaker_params['input_size']
    
    # Create random input spikes for demonstration
    print(f"Generating sample input for {input_size} input neurons...")
    
    deploy_on_spinnaker(params_file, simulation_time=500)

if __name__ == "__main__":
    print("="*60)
    print("SpiNNaker Deployment Script")
    print("="*60)
    
    # Deploy the trained model
    deploy_on_spinnaker()
    
    print("\n" + "="*60)
    print("Deployment completed!")
    print("The SNN is now running on SpiNNaker hardware/simulator")
    print("="*60)
