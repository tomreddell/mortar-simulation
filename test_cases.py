import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from utils import generate_orifice_function, IdealGasEOS, RK4
from mortar_simulation import MortarSimulation

RIGID_PACK_STROKE = 0.227 # m
RIGID_PACK_EJECTION_VELOCITY = 33.2 # m/s

RIGID_PACK_PRESSURE_TIME_DATA = np.array([
    8.82293826520008E-05,
    0.007694457662437,
    0.008359785542131,
    0.009571238870403,
    0.010661016046547,
    0.011780993233342,
    0.013020378994673,
    0.013925657812234,
    0.014618248158882,
    0.015310786969675,
    0.015823104897796,
    0.01664020586856,
    0.017186073637865,
    0.01791546058452,
    0.018765596037924,
    0.019464937581491,
    0.020133924506845,
    0.021408715400116,
    0.022744267065669,
    0.024413358780593,
    0.025447734913308,
    0.026693974943268,
    0.030057977836147])
    
RIGID_PACK_PRESSURE_DATA = np.array([
    0.106116446980772,
    0.099476567505957,
    0.152492531595773,
    0.178940732049632,
    0.223764556731527,
    0.27266795964399,
    0.392991810952743,
    0.543950635522977,
    0.717372908288511,
    0.892836000886416,
    1.00504604728586,
    1.04784760504296,
    1.03148394056885,
    0.947760853021461,
    0.882396898228712,
    0.78847177295364,
    0.696589528945112,
    0.614870155414958,
    0.527024199519342,
    0.430992288518327,
    0.269697432999095,
    0.118592246602499,
    0.104077688582575]) * 1.0E6

RIGID_PACK_LOAD_TIME_DATA = np.array([0.0,
                                      0.010135917950921,
                                      0.010389861998604,
                                      0.011711582451807,
                                      0.012594490683563,
                                      0.013321398270955,
                                      0.014078977608334,
                                      0.014645567309241,
                                      0.015242442141438,
                                      0.015681318799334,
                                      0.016432261182409,
                                      0.017402609676207,
                                      0.018685023895183,
                                      0.020032454491758,
                                      0.021410621274768,
                                      0.023165225796059,
                                      0.023727304945498,
                                      0.024602545239757,
                                      0.030031187241583])
    
RIGID_PACK_LOAD_DATA = np.array([82.26386726091,
                                 -0.558449229451071,
                                 1846.44794071846,
                                 4222.95011544864,
                                 6929.7964882135,
                                 10212.2321860066,
                                 13042.7965419105,
                                 14149.2992536111,
                                 14557.5256403372,
                                 14268.2489394834,
                                 12868.8610857542,
                                 11304.3011330076,
                                 8629.63002738549,
                                 7392.06357729689,
                                 5743.69328250013,
                                 4011.64151855233,
                                 2243.41942758954,
                                 63.2336358266621,
                                 -82.2638672609173])

COMPR_PACK_PRESSURE_TIME_DATA = np.array([5.13511778676458E-06,
                                          0.00762848492629,
                                          0.008957357126046,
                                          0.010749299270385,
                                          0.012821693733017,
                                          0.014246688918844,
                                          0.015209790957914,
                                          0.015834242677108,
                                          0.016551393970516,
                                          0.01729459528853,
                                          0.018377302779383,
                                          0.019058026830991,
                                          0.019737360121531,
                                          0.020290455099814,
                                          0.021152619979888,
                                          0.022075443438817,
                                          0.022967081755355,
                                          0.024413633737724,
                                          0.025644885208721,
                                          0.026599535699767,
                                          0.027182639022616,
                                          0.027884919870766,
                                          0.028773134775446,
                                          0.029912114598712,
                                          0.032597727710379])

COMPR_PACK_PRESSURE_DATA = np.array([0.099826689774697,
                                     0.097746967071057,
                                     0.131022530329289,
                                     0.166377816291161,
                                     0.253726169844021,
                                     0.355632582322357,
                                     0.478336221837088,
                                     0.617677642980936,
                                     0.759098786828423,
                                     0.806932409012132,
                                     0.85476603119584,
                                     0.888041594454073,
                                     0.894280762564991,
                                     0.846447140381282,
                                     0.806932409012132,
                                     0.746620450606586,
                                     0.680069324090121,
                                     0.60103986135182,
                                     0.536568457538995,
                                     0.494974003466204,
                                     0.430502599653379,
                                     0.282842287694974,
                                     0.149740034662045,
                                     0.091507798960139,
                                     0.099826689774697]) * 1.0E6

COMPR_PACK_LOAD_TIME_DATA = np.array([0,
                                      0.011756756756757,
                                      0.012006237006237,
                                      0.012006237006237,
                                      0.012006237006237,
                                      0.013347193347193,
                                      0.014282744282744,
                                      0.01506237006237,
                                      0.015717255717256,
                                      0.016995841995842,
                                      0.017900207900208,
                                      0.018274428274428,
                                      0.019022869022869,
                                      0.02039501039501,
                                      0.021798336798337,
                                      0.023638253638254,
                                      0.02529106029106,
                                      0.026039501039501,
                                      0.026881496881497])

COMPR_PACK_LOAD_DATA = np.array([40.9836065573763,
                                 40.9836065573763,
                                 1065.5737704918,
                                 1065.5737704918,
                                 1065.5737704918,
                                 3442.62295081967,
                                 6065.5737704918,
                                 8852.45901639344,
                                 9672.13114754098,
                                 10450.8196721311,
                                 11106.5573770492,
                                 10860.6557377049,
                                 9877.04918032787,
                                 8893.44262295082,
                                 7295.08196721311,
                                 5491.80327868852,
                                 4303.27868852459,
                                 2500,
                                 40.9836065573763])


def rk4_test():
    """
    Tests for the correct functioning of the RK4 algorithm
    """

    # Initialise an integrator
    integrator = RK4()

    INITIAL_STATE = np.array([500.0, 300.0])
    T_INF = 0.5 * (INITIAL_STATE[0] + INITIAL_STATE[1])
    C2 = 0.5 * (INITIAL_STATE[0] - INITIAL_STATE[1])
    AREA = 1.0E-4
    LENGTH = 1.0E-2
    VOLUME = 1.0E-6
    CONDUCTIVITY = 388.0
    DENSITY = 8960.0
    SPECIFIC_HEAT = 385.0
    BETA = CONDUCTIVITY / (DENSITY * SPECIFIC_HEAT * VOLUME) * AREA / LENGTH

    # Define a test system, two body heat exchange
    def system(state, time):

        T1, T2 = state

        return np.array([-BETA * (T1 - T2), BETA * (T1 - T2)])

    # Analytical solution
    def analytical(time):
        return np.array([T_INF + C2 * np.exp(-2.0 * BETA * time), T_INF - C2 * np.exp(-2.0 * BETA * time)])
                        
    t = 0.0
    dt = 1.0E-3
    end_time = 8.0 # s
    state = np.copy(INITIAL_STATE)

    while (t < end_time):

        state, dt_new = integrator.march(system, state, t, dt)

        dt = dt_new
        t += dt

        exact = analytical(t)

        assert abs(exact[0] - state[0]) < 1.0E-4, f"Time = {t}, {exact[0]}, {state[0]}"
        assert abs(exact[1] - state[1]) < 1.0E-4, f"Time = {t}, {exact[1]}, {state[1]}"

def rigid_pack_test():

    regulator_output_temperature = 1700.0
    regulator_output_pressure = 8.6E6 # Fitted to match measurements

    # These were arbitrarily chosen
    C = 0.4
    b = 0.5
    
    fill_function = generate_orifice_function(C,
                                              b,
                                              regulator_output_pressure,
                                              regulator_output_temperature)

    #
    # Initialise simulation
    # 
    simulation = MortarSimulation(
            
        # Model inputs
        fill_function = fill_function,
        regulator_output_pressure = regulator_output_pressure,
        regulator_output_temperature = regulator_output_temperature,
        gas_model = IdealGasEOS(378.0, 1.2559),
        cross_area_orifice = 6.54E-6,          
        volume_tank = 0.000714,
        radius_piston = 0.0725,
        external_pressure = 101325.0,          
        shear_pin_force_limit = 3000.0,        
        piston_mass = 2.757,                   
        parachute_mass = 0.293,
        friction_factor = 675.0,
        body_acceleration = 0.0,
        static_friction_force = 100.0,
        maximum_compression = 0.001,
        target_velocity = 33.2,                
        initial_density = 1.225,               
        initial_pressure = 101325.0,
        mortar_temperature = 273.0,
        heat_transfer_coefficient = 0.0,

        # Simulation control and logging
        time_offset = 0.012,
        initial_delta_time = 2.5E-6,   
        end_simulation_time = 0.05,
        minimum_run_time = 0.0,
        low_frequency_logging = 10000,
        high_frequency_logging = 10000.0,
        integrator = RK4())

    #
    # Incriment simulation by one tick until termination conditions are met
    # 
    while (simulation.is_complete == False):
        simulation.tick()

    #
    # Clean up
    # 
    simulation.end()

    data = simulation.logger

    #
    # Make Some Plots
    #
    fig, ax = plt.subplots()
    ax.plot(RIGID_PACK_PRESSURE_TIME_DATA, RIGID_PACK_PRESSURE_DATA / 1.0E6,
            label="Pressure (Measured)", linestyle="", marker="*")
    ax.plot(data.time, data.pressure / 1.0E6, label="Pressure (Simulated)")
    ax.legend()
    ax.set_title("Rigid Pack Pressure")
    ax.grid(True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (MPa)")

    fig2, ax2 = plt.subplots()
    ax2.plot(RIGID_PACK_LOAD_TIME_DATA, RIGID_PACK_LOAD_DATA / 1.0E3,
             label="Reaction Force (Measured)", linestyle="", marker="*")
    ax2.plot(data.time, data.net_force / 1.0E3, label="Reaction Force (Simulated)")
    
    ax2.set_title("Rigid Pack Reaction Force")
    ax2.grid(True)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Reaction Force (kN)")
    ax2.legend()

def compressible_pack_test():

    regulator_output_temperature = 1700.0
    regulator_output_pressure = 8.6E6 # Fitted to match measurements

    # These were arbitrarily chosen
    C = 0.4
    b = 0.5
    
    fill_function = generate_orifice_function(C,
                                              b,
                                              regulator_output_pressure,
                                              regulator_output_temperature)

    #
    # Initialise simulation
    # 
    simulation = MortarSimulation(
            
        # Model inputs
        fill_function = fill_function,
        regulator_output_pressure = regulator_output_pressure,
        regulator_output_temperature = regulator_output_temperature,
        gas_model = IdealGasEOS(378.0, 1.2559),
        cross_area_orifice = 6.54E-6,          
        volume_tank = 0.000714,
        radius_piston = 0.0725,
        external_pressure = 101325.0,          
        shear_pin_force_limit = 3000.0,        
        piston_mass = 2.725,                   
        parachute_mass = 0.293,
        friction_factor = 300.0, # 175.0
        body_acceleration = 0.0,
        static_friction_force = 100.0,
        maximum_compression = 0.02,
        target_velocity = 36.9,                
        initial_density = 1.225,               
        initial_pressure = 101325.0,
        mortar_temperature = 273.0,
        heat_transfer_coefficient = 0.0,

        # Simulation control and logging
        time_offset = 0.0135,
        initial_delta_time = 2.5E-6,   
        end_simulation_time = 0.05,
        minimum_run_time = 0.0,
        low_frequency_logging = 10000,
        high_frequency_logging = 10000.0,
        integrator = RK4())

    #
    # Incriment simulation by one tick until termination conditions are met
    # 
    while (simulation.is_complete == False):
        simulation.tick()

    #
    # Clean up
    # 
    simulation.end()

    data = simulation.logger

    #
    # Make Some Plots
    #
    fig, ax = plt.subplots()
    ax.plot(COMPR_PACK_PRESSURE_TIME_DATA, COMPR_PACK_PRESSURE_DATA / 1.0E6,
            label="Pressure (Measured)", linestyle="", marker="*")
    ax.plot(data.time, data.pressure / 1.0E6, label="Pressure (Simulated)")
    ax.legend()
    ax.set_title("Compressible Pack Pressure")
    ax.grid(True)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (MPa)")

    fig2, ax2 = plt.subplots()
    ax2.plot(COMPR_PACK_LOAD_TIME_DATA, COMPR_PACK_LOAD_DATA / 1.0E3,
             label="Reaction Force (Measured)", linestyle="", marker="*")
    ax2.plot(data.time, data.net_force / 1.0E3, label="Reaction Force (Simulated)")
    
    ax2.set_title("Compressible Pack Reaction Force")
    ax2.grid(True)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Reaction Force (kN)")
    ax2.legend()

if __name__ == "__main__":

    COMMAND_LINE_EXECUTION = False

    # Run all tests

    rk4_test()
    rigid_pack_test()
    compressible_pack_test()

    if (COMMAND_LINE_EXECUTION == True):
        plt.show()
