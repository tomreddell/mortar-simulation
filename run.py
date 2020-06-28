if __name__ == "__main__":

    from mortar_simulation import MortarSimulation
    from utils import RK4, generate_orifice_function
    
    import CoolProp.CoolProp as CP
    import numpy as np
    import matplotlib.pyplot as plt

    COMMAND_LINE_EXECUTION = False # Set to False if using an IDE which auto displays plots

    plt.close("all")

    #
    # Preprocessing
    #
    regulator_output_pressure = 900E3    # Pa
    regulator_output_temperature = 250.0 # K

    #
    # This function calculates the volume flow rate of the solenoid valve into the rear chamber (m^3/s)
    # It roughly approximates the flow rates from a plot like valve_flow_rates.png
    # There is definitely a more scientific way to do this, the valve I'm looking at is SMC VX214A
    # C = 0.63, b = 0.63, https://fluidpowerjournal.com/sonic-conductance-for-everyone/
    #
    fill_function = generate_orifice_function(0.63, 0.63, regulator_output_pressure, regulator_output_temperature)

    #
    # Initialise simulation
    # 
    simulation = MortarSimulation(

        # Model inputs
        fill_function = fill_function,              # Volume flow rate into the tank as a function of p1/p2 (m3/s)
        regulator_output_pressure = regulator_output_pressure,       # Regulator output pressure (Pa)
        regulator_output_temperature = regulator_output_temperature, # Regulator output temperature (K)
        gas_model = CP.AbstractState("HEOS", "N2"), # Gas equation of state calculator
        radius_piston = 0.0595,                     # Radius of the piston (m)
        cross_area_orifice = 0.0595 ** 2 * np.pi,   # Cross sectional area of orifice (m2)
        volume_tank = 0.0014,                       # Volume of the plenum/accumulator (m3)
        external_pressure = 13032.0,                # External pressure (Pa)
        shear_pin_force_limit = 8800.0,             # Force required to shear pins (N)
        piston_mass = 0.57,                         # Mass of the piston (kg)
        parachute_mass = 1.2,                       # Mass of the payload (kg)
        friction_factor = 150.0,                    # Viscous friction factor coefficient (-)
        body_acceleration = 0.0,                    # Acceleration of reference frame (m/s2)
        static_friction_force = 100.0,              # Static frictional force (N)
        maximum_compression = 0.001,                # Max expansion before shear pins break (m)
        initial_density = 0.146,                    # Initial density in tank (kg/m3)
        initial_pressure = 13032.0,                 # Initial pressure in tank (Pa)
        mortar_temperature = 300.0,                 # Temperature of the mortar (K)
        heat_transfer_coefficient = 0.0,            # Coefficient of heat transfer from gas to mortar (W/m2.K)

        # Simulation control and logging
        target_velocity = 25.0,                     # Sop simulaton once this velocity achieved (m/s)
        time_offset = 0.0,                          # Offset time by this many seconds (s)
        initial_delta_time = 2.5E-6,                # Initial time step (s)
        end_simulation_time = 0.25,                 # Stop after this time elapsed (s)
        minimum_run_time = 0.0,                     # Run for at least this duration (s)
        low_frequency_logging = 2000.0,             # Frequency to log state during fill phase (Hz)
        high_frequency_logging = 10000.0,           # Frequency to log state during expansion phase (Hz)
        integrator = RK4(),                         # Time integration algorithm
        reserve_logger_size = 2 ** 17)              # Reserve this many elements in each of the output arrays
    
    #
    # Incriment simulation by one tick until termination conditions are met
    # 
    while (simulation.is_complete == False):
        simulation.tick()

    #
    # Clean up
    # 
    simulation.end()

    #
    # Analyse results
    # 
    simulation.analyse()

    if (COMMAND_LINE_EXECUTION == True):
        plt.show()
