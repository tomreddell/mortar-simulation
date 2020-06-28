import numpy as np
import matplotlib.pyplot as plt
import CoolProp.CoolProp as CP

class Logger:
    """
    Stores time history of interesting fields
    """
    def __init__(self, size):
        self.index = 0
        self.size = size

        # Logged states
        self.time = np.zeros(size)
        self.pressure = np.zeros(size)
        self.density = np.zeros(size)
        self.displacement = np.zeros(size)
        self.velocity = np.zeros(size)
        self.mass = np.zeros(size)
        self.pressure_force = np.zeros(size)
        self.drag_force = np.zeros(size)
        self.net_force = np.zeros(size)
        self.mass_flow_in = np.zeros(size)
        self.energy = np.zeros(size)

        print("Logger initialised successfully")

    def log (self, simulation):
        """ Logs the current simulation state """

        # Check if we need to resize the arrays
        if (self.index >= self.size):
            self.resize(self.index + 1000)

        self.time[self.index] = simulation.time
        self.pressure[self.index] = simulation.pressure
        self.density[self.index] = simulation.density
        self.displacement[self.index] = simulation.displacement
        self.velocity[self.index] = simulation.velocity
        self.mass[self.index] = simulation.mass
        self.pressure_force[self.index] = simulation.pressure_force
        self.drag_force[self.index] = simulation.drag_force
        self.net_force[self.index] = simulation.net_force
        self.mass_flow_in[self.index] = simulation.mass_flow_in
        self.energy[self.index] = simulation.energy

        self.index += 1

    def resize(self, new_size):
        """ Resize the arrays """

        print(f"Resizing logger fields to {new_size}")

        self.size = new_size

        self.time.resize(new_size)
        self.pressure.resize(new_size)
        self.density.resize(new_size)
        self.displacement.resize(new_size)
        self.velocity.resize(new_size)
        self.mass.resize(new_size)
        self.pressure_force.resize(new_size)
        self.drag_force.resize(new_size)
        self.net_force.resize(new_size)
        self.mass_flow_in.resize(new_size)
        self.energy.resize(new_size)

class MortarSimulation:
    """

    x  |        |
    ^  |        |   
    |  |        |  
    |  |--------|  
    -  |--------|  Parachute + Piston
       |        |
       |        | Tank Volume
       ----  ----
           ^^
           || mass flow in

    Defines a Mortar Simulation, encapsulates all the behaviour needed to describe the physics of the system
    and automates its time integration and logging functionality.
    """

    # Initialise simulation
    def __init__(self,
                 fill_function,
                 regulator_output_pressure,
                 regulator_output_temperature,
                 gas_model,
                 radius_piston,
                 cross_area_orifice,
                 volume_tank,
                 external_pressure,
                 shear_pin_force_limit,
                 piston_mass,
                 parachute_mass,
                 friction_factor,
                 body_acceleration,
                 maximum_compression,
                 static_friction_force,
                 target_velocity,
                 initial_delta_time,
                 initial_density,
                 initial_pressure,
                 mortar_temperature,
                 heat_transfer_coefficient,
                 time_offset,
                 end_simulation_time,
                 low_frequency_logging,
                 high_frequency_logging,
                 integrator,
                 reserve_logger_size = 32768, # 2^15 by default
                 minimum_run_time = 0.0       
    ):
        # Function describing the volume flow rate of the solenoid into the rear chamber (m3/2)
        self._fill_function = fill_function

        # Pressure of incoming gas from the regulator 
        self._regulator_output_pressure = regulator_output_pressure

        # Temperature of incoming gas from the regulator
        self._regulator_output_temperature = regulator_output_temperature

        self._fluid = gas_model # Fluid state calculator

        # Calculate tank state
        self._fluid.update(CP.DmassP_INPUTS, initial_density, initial_pressure)
        initial_mass = initial_density * volume_tank
        initial_energy = initial_mass * self._fluid.umass()

        # Set to regulator state
        self._fluid.update(CP.PT_INPUTS, regulator_output_pressure, regulator_output_temperature)
        
        self._regulator_output_density = self._fluid.rhomass()
        self._regulator_output_static_enthalpy = self._fluid.hmass()
        self._cross_area_orifice = cross_area_orifice # Cross sectional area of the orifice (m2)
        self._external_pressure = external_pressure # External pressure (Pa)
        self._driven_mass = piston_mass + parachute_mass # Sum of the parachute and piston mass (kg)
        self._friction_factor = friction_factor # Viscous loss coefficient (-)
        self._target_velocity = target_velocity # Target velocity to reach (m/s)
        self._shear_pin_force_limit = shear_pin_force_limit # Burst force of the shear pins (N)
        self._body_acceleration = body_acceleration # Acceleration of the reference frame (m/s2)
        self._static_friction_force = static_friction_force # Minimum force before dynamic motion occurs (N)
        self._volume_tank = volume_tank # Volume of the plenum/accumulator (m3)
        self._mortar_temperature = mortar_temperature # Constant temperature of the mortar (K)
        self._heat_transfer_coefficient = heat_transfer_coefficient # heat transfer coeff to environment (W/m2.K)
        self._radius_piston = radius_piston # Radius of the piston (m)
        self._cross_area_piston = np.pi * radius_piston * radius_piston # Cross sectional area of the piston (m2)
        self._height_tank = volume_tank / self._cross_area_piston # initial height of the piston in the tank (m)

        # Maximum displacement of the piston before loading the shear pins (m)
        self._maximum_compression = maximum_compression 

        self._integrator = integrator # Time integration method

        #
        # Initialise state
        #

        self.volume_piston = 0.0 # Volume expanded by the piston (m3)
        self.pressure = 0.0 # Pressure within the chamber (Pa)
        self.density = 0.0  # Density within the chamber (kg/m3)
        self.inflow_velocity = 0.0 # Velocity at the inlet (m/s)
        self._shear_entropy = 0.0 # Specific Entropy at Shear Pin Break (J/kg.K)

        self.pressure_force = 0.0 # Force on the piston due to pressure differential (N)
        self.drag_force = 0.0     # Force on the piston due to drag (N)
        self.net_force = 0.0      # Net force on the piston (N)

        #
        # ODE State
        # 
        self.time = time_offset
        self._state = np.zeros(4)
        self._state_derivative = np.zeros(4)

        self._shear_pins_sheared = False        # Shear pins have sheared
        self._target_velocity_met = False       # The target velocity has been met
        self._velocity_negative = False         # Piston velocity reversed
        self._is_stagnant = False               # Tank is not receiving further mass flux
        self._shear_pins_shear_time = float("inf") # Time at which the shear pins shear (s)

        # Need to initialise with a minimum amount of mass in the chamber to prevent the Coolprop solver
        # from failing on a         
                
        self._state = np.array([0.0, 0.0, initial_mass, initial_energy])
        
        self._state_derivative = self._system(self._state, self.time)
        self._delta_time = initial_delta_time
        self._end_simulation_time = end_simulation_time # Stop after this time if we havn't already (s)
        self._minimum_run_time = minimum_run_time       # Require this much time elapsed before exit (s)
        self.is_complete = False
        self._iterations = 0

        self.displacement = self._state[0]
        self.velocity = self._state[1]
        self.mass = self._state[2]
        self.energy = self._state[3]

        #
        # Logging
        #
        self._force_log = False # Must log this iteration
        self._low_frequency_logging = low_frequency_logging
        self._high_frequency_logging = high_frequency_logging
        self._last_log_time = self.time
        self._logger_period = 1.0 / low_frequency_logging # Start off with the low frequency logger
        self.logger = Logger(reserve_logger_size)

        # Log initial state
        self.logger.log(self)

        print(f"Simulation initialised successfully")

    #
    # Public API
    #
    
    def tick(self):
        """ Advance the state of the simulation by one tick """

        # Update discrete simulation state
        self._update_discrete_state()

        # Advance the time state, supports adaptive integration
        self._state, delta_time = self._integrator.march(self._system, self._state, self.time, self._delta_time)

        # Incriment time
        self.time += self._delta_time

        # Set the new time step
        self._delta_time = delta_time

        # Incriment iteration counter
        self._iterations += 1

        # Check for a simulation exit
        if (self._check_end_simulation_conditions() == True):
            self.is_complete = True
            self._force_log = True

        # Log the current state
        if ((self.time - self._last_log_time >= self._logger_period) or (self._force_log == True)):
            self.logger.log(self)
            self._last_update_time = self.time

    def end(self):
        """ Clean up any remaining simulation state and exit the simulation """

        # Truncate logs to actual size
        self.logger.resize(self.logger.index)

        print(f"Simulation ended at t={self.time:.5g} after {self._iterations} iterations")

    def analyse(self):
        """ Analyse the results of the simulation """

        # Post process some other interesting fields
        inflow_velocity = self.logger.mass_flow_in / (self._regulator_output_density * self._cross_area_orifice)

        boundary_work = -self.logger.pressure_force * self.logger.velocity
        viscous_loss = -self.logger.drag_force * self.logger.velocity
        enthalpy_in = self.logger.mass_flow_in * (self._regulator_output_static_enthalpy
                                                   + 0.5 * inflow_velocity * inflow_velocity)

        static_friction_force = np.vectorize(self._calculate_static_friction_force)(self.logger.net_force,
                                                                                    self.logger.velocity)

        temperature = np.zeros_like(inflow_velocity)
        specific_entropy = np.zeros_like(inflow_velocity)

        for i, (p, d) in enumerate(zip(self.logger.pressure, self.logger.density)):
            self._fluid.update(CP.DmassP_INPUTS, d, p)
            temperature[i] = self._fluid.T()
            specific_entropy[i] = self._fluid.smass()

        external_heat_transfer = np.vectorize(self._calculate_external_heat_transfer)(self.logger.displacement,
                                                                                      temperature)

        volume = self._volume_tank + self._cross_area_piston * self.logger.displacement

        if (self._shear_pins_shear_time == float("inf")):
            shear_index = None
        else:
            shear_index = np.absolute(self.logger.time - self._shear_pins_shear_time).argmin()

        fig, ax = plt.subplots()
        ax.plot(self.logger.time, self.logger.displacement, label="piston")
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Displacement (m)')

        if (shear_index != None):
            ax.axvline(x=self._shear_pins_shear_time, label="Shear Pins Break", linestyle="--", color="black")
            
        ax.legend()
        ax.grid(True)

        fig2, ax2 = plt.subplots()
        ax2.plot(self.logger.time, self.logger.velocity, label="Piston")
        ax2.plot(self.logger.time, inflow_velocity, label="Inflow")
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Velocity (m/s)')

        if (shear_index != None):
            ax2.axvline(x=self._shear_pins_shear_time, label="Shear Pins Break", linestyle="--", color="black")
            
        ax2.legend()
        ax2.grid(True)

        fig3, ax3 = plt.subplots()
        ax3.plot(self.logger.time, self.logger.pressure / 1.0E3, label="chamber")
        ax3.axvline(x=self._shear_pins_shear_time, label="Shear Pins Break", linestyle="--", color="black")
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Pressure (kPa)')
        ax3.legend()
        ax3.grid(True)

        fig4, ax4 = plt.subplots()
        ax4.plot(self.logger.time, self.logger.mass_flow_in, label="Valve")
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Mass Flow Rate (kg/s)')

        if (shear_index != None):
            ax4.axvline(x=self._shear_pins_shear_time, label="Shear Pins Break", linestyle="--", color="black")
            
        ax4.legend()
        ax4.grid(True)

        fig5, ax5 = plt.subplots()
        ax5.plot(self.logger.time, temperature, label="Chamber")

        if (shear_index != None):
            ax5.axvline(x=self._shear_pins_shear_time, label="Shear Pins Break", linestyle="--", color="black")
            
        ax5.legend()
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Temperature (K)')
        ax5.grid(True)

        fig6, ax6 = plt.subplots()
        ax6.plot(self.logger.time, self.logger.mass)

        if (shear_index != None):
            ax6.axvline(x=self._shear_pins_shear_time, label="Shear Pins Break", linestyle="--", color="black")

        ax6.set_xlabel("Time (s)")
        ax6.set_ylabel("Chamber Mass (kg)")
        ax6.grid(True)

        fig7, ax7 = plt.subplots()
        ax7.plot(self.logger.time, self.logger.pressure_force, label="Pressure Force")
        ax7.plot(self.logger.time, -self.logger.drag_force, label="Dynamic Friction")
        ax7.plot(self.logger.time, -static_friction_force, label="Static Friction")
        ax7.plot(self.logger.time, np.ones_like(temperature) * self._body_acceleration * self._driven_mass,
                 label="Inertial")

        if (shear_index != None):
            ax7.axvline(x=self._shear_pins_shear_time, label="Shear Pins Break", linestyle="--", color="black")

        ax7.set_xlabel('Time (s)')
        ax7.set_ylabel('Forces (N)')
        ax7.grid(True)
        ax7.legend()

        fig9, ax9 = plt.subplots()
        ax9.plot(self.logger.displacement, self.logger.velocity)
        ax9.set_xlabel("Piston Displacement (m)")
        ax9.set_ylabel("Piston Velocity (m/s)")
        ax9.grid(True)

        fig10, ax10 = plt.subplots()
        ax10.plot(self.logger.time, enthalpy_in, label="Enthalpy In")
        ax10.plot(self.logger.time, boundary_work, label="Boundary Work")
        ax10.plot(self.logger.time, viscous_loss, label="Viscous Losses")
        ax10.plot(self.logger.time, external_heat_transfer, label="External Heat Transfer")

        if (shear_index != None):
            ax10.axvline(x=self._shear_pins_shear_time, label="Shear Pins Break", linestyle="--", color="black")

        ax10.set_xlabel("Time (s)")
        ax10.set_ylabel("Energy Rates (W)")
        ax10.grid()
        ax10.legend()

        fig11, (ax11a, ax11b) = plt.subplots(1,2)

        # P - V
        ax11a.plot(volume, self.logger.pressure / 1.0E3)
        ax11a.plot([volume[0]], [self.logger.pressure[0] / 1.0E3],
                   linestyle="",
                   marker="o",
                   label=f"t={self.logger.time[0]:.3g}")
        
        if (shear_index != None):
            ax11a.plot([volume[shear_index]], [self.logger.pressure[shear_index] / 1.0E3],
                       linestyle="",
                       marker="o",
                       label=f"Pins Shear")
            
        ax11a.plot([volume[-1]], [self.logger.pressure[-1] / 1.0E3],
                   linestyle="",
                   marker="o",
                   label=f"t={self.logger.time[-1]:.3g}")
        ax11a.set_xlabel("Volume (m3)")
        ax11a.set_ylabel("Pressure (kPa)")
        ax11a.legend()
        ax11a.grid()

        # T - s
        ax11b.plot(specific_entropy, temperature)
        ax11b.plot([specific_entropy[0]], [temperature[0]],
                   linestyle="",
                   marker="o",
                   label=f"t={self.logger.time[0]:.3g}")
        
        if (shear_index != None):
            ax11b.plot([specific_entropy[shear_index]], [temperature[shear_index]],
                       linestyle="",
                       marker="o",
                       label=f"Pins Shear")
            
        ax11b.plot([specific_entropy[-1]], [temperature[-1]],
                   linestyle="",
                   marker="o",
                   label=f"t={self.logger.time[-1]:.3g}")
        
        ax11b.set_xlabel("Specific Entropy (J/kg.K)")
        ax11b.set_ylabel("Temperature (K)")
        ax11b.grid()

        fig11.tight_layout()

    #
    # Private Methods
    #

    def _system(self, state, time):
        """
        ODE's describing the physics of the system

        state = [x, v, m, E]
        x : displacement (m)
        v : velocity (m/s)
        m : mass (kg)
        E : internal energy (J)

        Returns: [dx/dt, dv/dt, dm/dt, dE/dt]

        Total fluid energy is E = m (e + 1/2 v^2), i.e static + kinetic internal energy

        Assumes the incoming flow stagnates into the base of the chamber, so that the average fluid
        velocity is 1/2 v_piston
        """

        x, v, m, E = state

        self.displacement = x
        self.velocity = v
        self.mass = m
        self.energy = E

        # Update the volume and mass within the chamber
        self.volume_piston =  self._cross_area_piston * x
        self.density = m / (self.volume_piston + self._volume_tank)

        # Update the internal fluid state using density, specific internal energy inputs
        specific_internal_energy = E / m - 0.125 * v * v
        self._fluid.update(CP.DmassUmass_INPUTS, self.density, specific_internal_energy)
        self.pressure = self._fluid.p()

        # Forces
        self.pressure_force = (self.pressure - self._external_pressure) * self._cross_area_piston
        self.drag_force = 0.5 * self.density * self._cross_area_piston * self._friction_factor * v * abs(v)

        #
        # Populate state derivative
        #
        
        # Calculate the net force on the driven mass (parachute + piston)
        self.net_force = self.pressure_force - self.drag_force - self._driven_mass * self._body_acceleration

        # Apply static friction
        self.net_force -= self._calculate_static_friction_force(self.net_force, v)

        dxdt = v
        dvdt = self.net_force / self._driven_mass

        # If at the maximum compressed displacement and the shear pins have not burst, then do not apply
        # acceleration upwards and zero velocity
        if ((x >= self._maximum_compression) and (self._shear_pins_sheared == False)):
            dvdt = min(0.0, dvdt)
            dxdt = min(0.0, dxdt)

        # Mass flow into the chamber
        pressure_ratio = self.pressure / self._regulator_output_pressure
        self.mass_flow_in = self._fill_function(pressure_ratio) * self._regulator_output_density
        
        self.inflow_velocity = self.mass_flow_in / (self._regulator_output_density * self._cross_area_orifice)
        total_inflow_enthalpy = self._regulator_output_static_enthalpy + 0.5 * self.inflow_velocity ** 2

        # External heat transfer
        external_heat_transfer = self._calculate_external_heat_transfer(x, self._fluid.T())

        # Conservation of energy dot(E) = dot(m)in * Hin - dot(W) - dot(W)loss - dot(Q)loss
        dEdt = (self.mass_flow_in * total_inflow_enthalpy - self.pressure_force * v - self.drag_force * v
                + external_heat_transfer)

        self._state_derivative[0] = dxdt
        self._state_derivative[1] = dvdt
        self._state_derivative[2] = self.mass_flow_in
        self._state_derivative[3] = dEdt

        return self._state_derivative

    def _update_discrete_state(self):
        """ Updates discrete events of the system """

        self._force_log = False

        # Check if the shear pins should shear
        if ((self._shear_pins_sheared == False) and\
            (self.net_force > self._shear_pin_force_limit) and\
            (self.displacement >= self._maximum_compression)):
            
            self._shear_pins_sheared = True
            self._shear_pins_shear_time = self.time
            self._force_log = True
            self._logger_period = 1.0 / self._high_frequency_logging # Switch to high frequency logging
            print(f"Shear Pins burst at t={self.time:.5g} s with net force = {self.net_force:.5g} N")

        # Check if we have reached the target velocity
        if ((self._target_velocity_met == False) and (self.velocity >= self._target_velocity)):
            print ((f"Target Velocity v={self.velocity:.5g} m/s reached at t={self.time:.5g} s, "
                    f" x={self.displacement:.5g} m"))
            self._target_velocity_met = True

        # Checks if the piston velocity is negative
        if ((self._velocity_negative == False) and (self._shear_pins_sheared == True) and (self.velocity < 0)):
            print((f"Velocity is negative at t={self.time:.5g}, x = {self.displacement:.5g}"
                   " simulation is ill posed or not energetic enough to eject payload"))
            self._velocity_negative = True

        # Checks if we have failed to shear the pins
        if ((self._shear_pins_sheared == False) and (self._is_stagnant == False) and (self.mass_flow_in <= 0)):
            print((f"Mass flow into accumulator is 0 but shear pins have not sheared at time={self.time:.5g}"
                   f". Net force is {self.net_force:.5g} N, required shear force "
                   f"{self._shear_pin_force_limit:.5g} N"))
            self._is_stagnant = True

        # Check for overshoot of the maximum compression, and forcefully zero motion if required
        if (self._shear_pins_sheared == False):
            if ((self.displacement > self._maximum_compression)) or\
               ((self.displacement == self._maximum_compression) and (self.velocity > 0.0)):

                self._state[0] = self._maximum_compression
                self._state[1] = 0.0

                # Recalculate state
                self._state_derivative = self._system(self._state, self.time)

                self._force_log = True

    def _check_end_simulation_conditions(self):
        """ Checks if this simulation is complete """

        # Target velocity reached
        if ((self._target_velocity_met == True) and (self.time > self._minimum_run_time)):
            return True

        # Target velocity not possible, or simulation error
        if ((self._velocity_negative == True) and (self.time > self._minimum_run_time)):
            return True

        # Internal pressure not enough to burst
        if ((self._is_stagnant == True) and (self.time > self._minimum_run_time)):
            return True

        # Timeout
        if (self.time > self._end_simulation_time):
            print(f"Simulation end time {self._end_simulation_time:.5g} exceeded.")
            return True

        # Continue
        return False

    def _calculate_static_friction_force(self, net_force, velocity):
        "Calculates the force of static friction on the drien mass"
        if (velocity == 0):
            if (net_force > 0):
                return min(net_force, self._static_friction_force)
            elif (self.net_force < 0):
                return -min(-net_force, self._static_friction_force)

        return 0.0
        
    def _calculate_external_heat_transfer(self, displacement, temperature):
        "Calculates heat transfer from the mortar to the chamber gas"
        heat_transfer_area = 2.0 * np.pi * self._radius_piston * (displacement + self._height_tank)
        return -heat_transfer_area * self._heat_transfer_coefficient * (temperature - self._mortar_temperature)

