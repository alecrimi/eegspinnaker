import pyNN.spiNNaker as pynn

def deploy_on_spinnaker(spinnaker_params, input_spikes):
    """Deploy the converted model on SpiNNaker hardware"""
    
    pynn.setup(timestep=1.0)
    
    # Create populations
    populations = []
    for i, n_neurons in enumerate(spinnaker_params['n_neurons']):
        population = pynn.Population(
            n_neurons, 
            pynn.IF_curr_exp(
                tau_m=spinnaker_params['tau_mem'][i] * 20.0,  # Convert to ms
                v_rest=-65.0,
                v_reset=-65.0,
                v_thresh=-50.0
            )
        )
        populations.append(population)
    
    # Create projections (connections)
    for i in range(len(populations) - 1):
        connector = pynn.AllToAllConnector()
        projection = pynn.Projection(
            populations[i],
            populations[i + 1],
            connector,
            synapse_type=pynn.StaticSynapse(weight=spinnaker_params['weights'][i])
        )
    
    # Run simulation
    pynn.run(1000)  # Run for 1000ms
    pynn.end()
