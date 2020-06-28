import CoolProp.CoolProp as CP
import numpy as np

class RK4:
    """
    Classic Fourth Order Runge Kutta Integration
    """

    def __init__(self):
        print("RK4 Integrator initialised successfully")

    def march(self, odes, state, time, delta_time):
        """
        Classic Fixed Step Algoritm
        
        Calculates a new state X[t + dt] and a new dt (for adaptive methods only)

        Inputs:
        odes: System of ordinary differential equations
        state: state at previous iteration X[t]
        time: time at previous iteration
        delta_time: dt at previous iteration

        Returns:
        state at next time step: X[t + dt]
        dt: For compatibile interface with adaptive step solvers
        """

        k1 = odes(state, time) * delta_time
        k2 = odes(state + 0.5 * k1, time + 0.5 * delta_time) * delta_time
        k3 = odes(state + 0.5 * k2, time + 0.5 * delta_time) * delta_time
        k4 = odes(state + k3, time + delta_time) * delta_time

        new_state = state + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        return new_state, delta_time

class IdealGasEOS:
    """
    Ideal Gas Equation of state calculator which mimics the interface of
    CoolProp.AbstractState allowing it be passed as an input to MortarSimulation

    Inputs:
    gas_constant: Specific gas constant (J/kg.K)
    gamma       : Ratio of specific heats
    """

    def __init__(self, gas_constant, gamma):
        self._gamma = gamma
        self._gas_constant = gas_constant
        self._cvmass = gas_constant / (gamma - 1.0)
        self._cpmass = gas_constant + self._cvmass
        self._pressure = 0.0
        self._density = 0.0
        self._temperature = 0.0
        self._internal_energy = 0.0

    #
    # Public API
    # 

    def update(self, update_key, input1, input2):
        """
        Update method which mimics the CoolProp interface
        """

        if (update_key == CP.PT_INPUTS):
            self._update_from_pT(input1, input2)
        elif (update_key == CP.DmassP_INPUTS):
            self._update_from_dp(input1, input2)
        elif (update_key == CP.DmassUmass_INPUTS):
            self._update_from_du(input1, input2)
        else:
            raise Exception("Unrecognised update key" + update_key)

    def p(self) : return self._pressure
    def T(self) : return self._temperature
    def rhomass(self) : return self._density
    def umass(self) : return self._internal_energy
    def hmass(self) : return self._cpmass * self._temperature
    def smass(self) : raise Exception("Entropy has not been defined for IdealEOS object")

    #
    # Private Methods
    #
    def _update_from_pT(self, pressure, temperature):
        self._pressure = pressure
        self._temperature = temperature
        self._density = self._pressure / (self._gas_constant * self._temperature)
        self._internal_energy = self._cvmass * self._temperature

    def _update_from_dp(self, density, pressure):
        self._density = density
        self._pressure = pressure
        self._temperature = pressure / (self._gas_constant * density)
        self._internal_energy = self._cvmass * self._temperature

    def _update_from_du(self, density, internal_energy):
        self._internal_energy = internal_energy
        self._density = density
        self._temperature = internal_energy / self._cvmass
        self._pressure = self._density * self._gas_constant * self._temperature

def generate_orifice_function(C, b, p1, T):
    """
    Produces a function defining volumetric flow rate through an orifice

    Assumes source pressure and temperature are constant

    Input:
    C  : Maximum flow capacity of any pneumatic component with an open flow path (L/s.bar)
    b  : Critical pressure ratio between output and input pressure when the flow is choked (-)
    p1 : source pressure (Pa)
    T  : source temperature (K)
    
    https://fluidpowerjournal.com/sonic-conductance-for-everyone/
    """

    p1 /= 1.0E6 # Convert to MPa
    sonic_value = 600.0 * C * p1 * np.sqrt(293.0 / T) * 1.66667E-5

    def result (pressure_ratio):
        
        if (pressure_ratio >= 1):
            return 0.0
        elif (pressure_ratio < C):
            return sonic_value
        else:
            return sonic_value * abs(1.0 - ( pressure_ratio - b) / (1.0 - b) )

    return result
