## Overview
Simple simulation of a mortar fire process. Simulates the process of an accumulator fill, shear pin burst, and piston acceleration. Terminates the simulation once the piston reaches the top of the chamber.

## Features
* Solves transient energy and mass balance equations. Does not require isentropic assumption
* Calculates ejection velocity, losses, and transient dynamics
* Highly modular
* 22 tunable parameters
* Leverages CoolProp thermofluids library for non ideal gas support
* Automatic logging and time stepping control

## Limitations
* Current gas model can not adequately approximate a gas generation based pressurisation

## Requirements
* Python version 3.7 or greater
* Scientific python libraries including numpy, scipy, matplotlib, and CoolProp
