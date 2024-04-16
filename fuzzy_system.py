import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import matplotlib.pyplot as plt
from datetime import datetime

"""
Set up the path
"""
GRAPH_DIR = "graph"
os.makedirs(GRAPH_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join(GRAPH_DIR, timestamp)
os.makedirs(LOG_DIR, exist_ok=True)
"""
Set up the graph size
"""
GRAPH_WIDTH = 8
GRAPH_HEIGHT = 4

"""
Step 1: Define Fuzzy Variables and Membership Functions
"""
# Input variables
x = np.linspace(0, 6, 1000)
asset_values = ctrl.Antecedent(x, "asset_values")
threat_levels = ctrl.Antecedent(x, "threat_levels")
vulnerability_levels = ctrl.Antecedent(x, "vulnerability_levels")
# Output variable
y = np.linspace(0, 40, 1000)
risk = ctrl.Consequent(y, "risk")

"""
Step 2: Define Membership Functions
"""
# Define Membership Functions
LEFT_ZERO_DIFF = 0.334
LEFT_ONE_DIFF = 0.1577
RIGHT_ONE_DIFF = 0.53
RIGHT_ZERO_DIFF = 0.94
# Asset values
asset_values["low"] = fuzz.trapmf(
    asset_values.universe,
    [1 - LEFT_ZERO_DIFF, 1 - LEFT_ONE_DIFF, 1 + RIGHT_ONE_DIFF, 1 + RIGHT_ZERO_DIFF],
)
asset_values["medium"] = fuzz.trapmf(
    asset_values.universe,
    [2 - LEFT_ZERO_DIFF, 2 - LEFT_ONE_DIFF, 2 + RIGHT_ONE_DIFF, 2 + RIGHT_ZERO_DIFF],
)
asset_values["high"] = fuzz.trapmf(
    asset_values.universe,
    [3 - LEFT_ZERO_DIFF, 3 - LEFT_ONE_DIFF, 3 + RIGHT_ONE_DIFF, 3 + RIGHT_ZERO_DIFF],
)
asset_values["very_high"] = fuzz.trapmf(
    asset_values.universe,
    [4 - LEFT_ZERO_DIFF, 4 - LEFT_ONE_DIFF, 4 + RIGHT_ONE_DIFF, 4 + RIGHT_ZERO_DIFF],
)
LEFT_ZERO_DIFF = LEFT_ZERO_DIFF * (3 / 4)
LEFT_ONE_DIFF = LEFT_ONE_DIFF * (3 / 4)
RIGHT_ONE_DIFF = RIGHT_ONE_DIFF * (3 / 4)
RIGHT_ZERO_DIFF = RIGHT_ZERO_DIFF * (3 / 4)
# Threat levels
threat_levels["low"] = fuzz.trapmf(
    threat_levels.universe,
    [1 - LEFT_ZERO_DIFF, 1 - LEFT_ONE_DIFF, 1 + RIGHT_ONE_DIFF, 1 + RIGHT_ZERO_DIFF],
)
threat_levels["medium"] = fuzz.trapmf(
    threat_levels.universe,
    [2 - LEFT_ZERO_DIFF, 2 - LEFT_ONE_DIFF, 2 + RIGHT_ONE_DIFF, 2 + RIGHT_ZERO_DIFF],
)
threat_levels["high"] = fuzz.trapmf(
    threat_levels.universe,
    [3 - LEFT_ZERO_DIFF, 3 - LEFT_ONE_DIFF, 3 + RIGHT_ONE_DIFF, 3 + RIGHT_ZERO_DIFF],
)
# Vulnerability levels
vulnerability_levels["low"] = fuzz.trapmf(
    vulnerability_levels.universe,
    [1 - LEFT_ZERO_DIFF, 1 - LEFT_ONE_DIFF, 1 + RIGHT_ONE_DIFF, 1 + RIGHT_ZERO_DIFF],
)
vulnerability_levels["medium"] = fuzz.trapmf(
    vulnerability_levels.universe,
    [2 - LEFT_ZERO_DIFF, 2 - LEFT_ONE_DIFF, 2 + RIGHT_ONE_DIFF, 2 + RIGHT_ZERO_DIFF],
)
vulnerability_levels["high"] = fuzz.trapmf(
    vulnerability_levels.universe,
    [3 - LEFT_ZERO_DIFF, 3 - LEFT_ONE_DIFF, 3 + RIGHT_ONE_DIFF, 3 + RIGHT_ZERO_DIFF],
)
# Risk
LEFT_ZERO_DIFF = LEFT_ZERO_DIFF * (36 / 3)
LEFT_ONE_DIFF = LEFT_ONE_DIFF * (36 / 3)
RIGHT_ONE_DIFF = RIGHT_ONE_DIFF * (36 / 3)
RIGHT_ZERO_DIFF = RIGHT_ZERO_DIFF * (36 / 3)
UNIT = (36 - 1) / 10
risk["very_low"] = fuzz.trapmf(
    risk.universe,
    [
        (1 + UNIT) - LEFT_ZERO_DIFF,
        (1 + UNIT) - LEFT_ONE_DIFF,
        (1 + UNIT) + RIGHT_ONE_DIFF,
        (1 + UNIT) + RIGHT_ZERO_DIFF,
    ],
)
risk["low"] = fuzz.trapmf(
    risk.universe,
    [
        (1 + (1 * 2 + 1) * UNIT) - LEFT_ZERO_DIFF,
        (1 + (1 * 2 + 1) * UNIT) - LEFT_ONE_DIFF,
        (1 + (1 * 2 + 1) * UNIT) + RIGHT_ONE_DIFF,
        (1 + (1 * 2 + 1) * UNIT) + RIGHT_ZERO_DIFF,
    ],
)
risk["medium"] = fuzz.trapmf(
    risk.universe,
    [
        (1 + (2 * 2 + 1) * UNIT) - LEFT_ZERO_DIFF,
        (1 + (2 * 2 + 1) * UNIT) - LEFT_ONE_DIFF,
        (1 + (2 * 2 + 1) * UNIT) + RIGHT_ONE_DIFF,
        (1 + (2 * 2 + 1) * UNIT) + RIGHT_ZERO_DIFF,
    ],
)
risk["high"] = fuzz.trapmf(
    risk.universe,
    [
        (1 + (3 * 2 + 1) * UNIT) - LEFT_ZERO_DIFF,
        (1 + (3 * 2 + 1) * UNIT) - LEFT_ONE_DIFF,
        (1 + (3 * 2 + 1) * UNIT) + RIGHT_ONE_DIFF,
        (1 + (3 * 2 + 1) * UNIT) + RIGHT_ZERO_DIFF,
    ],
)
risk["very_high"] = fuzz.trapmf(
    risk.universe,
    [
        (1 + (4 * 2 + 1) * UNIT) - LEFT_ZERO_DIFF,
        (1 + (4 * 2 + 1) * UNIT) - LEFT_ONE_DIFF,
        (1 + (4 * 2 + 1) * UNIT) + RIGHT_ONE_DIFF,
        (1 + (4 * 2 + 1) * UNIT) + RIGHT_ZERO_DIFF,
    ],
)


"""
Save the membership functions image
"""
# Asset Values
plt.figure(figsize=(GRAPH_WIDTH, GRAPH_HEIGHT))
plt.plot(asset_values.universe, asset_values["low"].mf, "b", linewidth=1.5, label="Low")
plt.plot(
    asset_values.universe, asset_values["medium"].mf, "g", linewidth=1.5, label="Medium"
)
plt.plot(
    asset_values.universe, asset_values["high"].mf, "r", linewidth=1.5, label="High"
)
plt.plot(
    asset_values.universe,
    asset_values["very_high"].mf,
    "k",
    linewidth=1.5,
    label="very_high",
)
plt.title("Membership Functions for Asset Values")
plt.ylabel("Membership degree")
plt.xlabel("Asset Value")
plt.legend()
plt.savefig(os.path.join(LOG_DIR, "asset_values.png"))
plt.close()

# Threat Levels
plt.figure(figsize=(GRAPH_WIDTH, GRAPH_HEIGHT))
plt.plot(
    threat_levels.universe, threat_levels["low"].mf, "b", linewidth=1.5, label="Low"
)
plt.plot(
    threat_levels.universe,
    threat_levels["medium"].mf,
    "g",
    linewidth=1.5,
    label="Medium",
)
plt.plot(
    threat_levels.universe, threat_levels["high"].mf, "r", linewidth=1.5, label="High"
)
plt.title("Membership Functions for Threat Levels")
plt.ylabel("Membership degree")
plt.xlabel("Threat Level")
plt.legend()
plt.savefig(os.path.join(LOG_DIR, "threat_levels.png"))
plt.close()

# Vulnerability levels
plt.figure(figsize=(GRAPH_WIDTH, GRAPH_HEIGHT))
plt.plot(
    vulnerability_levels.universe,
    vulnerability_levels["low"].mf,
    "b",
    linewidth=1.5,
    label="Low",
)
plt.plot(
    vulnerability_levels.universe,
    vulnerability_levels["medium"].mf,
    "g",
    linewidth=1.5,
    label="Medium",
)
plt.plot(
    vulnerability_levels.universe,
    vulnerability_levels["high"].mf,
    "r",
    linewidth=1.5,
    label="High",
)
plt.title("Membership Functions for Vulnerability Levels")
plt.ylabel("Membership degree")
plt.xlabel("Value")
plt.legend()
plt.savefig(os.path.join(LOG_DIR, "vulnerability_levels.png"))
plt.close()

# Risk
plt.figure(figsize=(GRAPH_WIDTH, GRAPH_HEIGHT))
plt.plot(risk.universe, risk["very_low"].mf, "b", linewidth=1.5, label="very_low")
plt.plot(risk.universe, risk["low"].mf, "g", linewidth=1.5, label="Low")
plt.plot(risk.universe, risk["medium"].mf, "r", linewidth=1.5, label="Medium")
plt.plot(risk.universe, risk["high"].mf, "k", linewidth=1.5, label="High")
plt.plot(risk.universe, risk["very_high"].mf, "y", linewidth=1.5, label="very_high")
plt.title("Membership Functions for Risk")
plt.ylabel("Membership degree")
plt.xlabel("Value")
plt.legend()

plt.savefig(os.path.join(LOG_DIR, "risk.png"))
plt.close()


"""
Step 3: Define Fuzzy Rules
"""
# Create the fuzzy control system

# Define the rules
rules = [
    # Dependent on the highest value
    ctrl.Rule(
        asset_values["low"] & threat_levels["low"] & vulnerability_levels["low"],
        risk["very_low"],
    ),
    ctrl.Rule(
        asset_values["medium"] & threat_levels["low"] & vulnerability_levels["low"],
        risk["very_low"],
    ),
    ctrl.Rule(
        asset_values["high"] & threat_levels["low"] & vulnerability_levels["low"],
        risk["very_low"],
    ),
    ctrl.Rule(
        asset_values["very_high"] & threat_levels["low"] & vulnerability_levels["low"],
        risk["very_low"],
    ),
    ctrl.Rule(
        asset_values["low"] & threat_levels["low"] & vulnerability_levels["medium"],
        risk["very_low"],
    ),
    ctrl.Rule(
        asset_values["medium"] & threat_levels["low"] & vulnerability_levels["medium"],
        risk["very_low"],
    ),
    ctrl.Rule(
        asset_values["high"] & threat_levels["low"] & vulnerability_levels["medium"],
        risk["low"],
    ),
    ctrl.Rule(
        asset_values["very_high"]
        & threat_levels["low"]
        & vulnerability_levels["medium"],
        risk["low"],
    ),
    ctrl.Rule(
        asset_values["low"] & threat_levels["low"] & vulnerability_levels["high"],
        risk["very_low"],
    ),
    ctrl.Rule(
        asset_values["medium"] & threat_levels["low"] & vulnerability_levels["high"],
        risk["low"],
    ),
    ctrl.Rule(
        asset_values["high"] & threat_levels["low"] & vulnerability_levels["high"],
        risk["medium"],
    ),
    ctrl.Rule(
        asset_values["very_high"] & threat_levels["low"] & vulnerability_levels["high"],
        risk["medium"],
    ),
    ctrl.Rule(
        asset_values["low"] & threat_levels["medium"] & vulnerability_levels["low"],
        risk["very_low"],
    ),
    ctrl.Rule(
        asset_values["medium"] & threat_levels["medium"] & vulnerability_levels["low"],
        risk["very_low"],
    ),
    ctrl.Rule(
        asset_values["high"] & threat_levels["medium"] & vulnerability_levels["low"],
        risk["low"],
    ),
    ctrl.Rule(
        asset_values["very_high"]
        & threat_levels["medium"]
        & vulnerability_levels["low"],
        risk["low"],
    ),
    ctrl.Rule(
        asset_values["low"] & threat_levels["medium"] & vulnerability_levels["medium"],
        risk["very_low"],
    ),
    ctrl.Rule(
        asset_values["medium"]
        & threat_levels["medium"]
        & vulnerability_levels["medium"],
        risk["low"],
    ),
    ctrl.Rule(
        asset_values["high"] & threat_levels["medium"] & vulnerability_levels["medium"],
        risk["medium"],
    ),
    ctrl.Rule(
        asset_values["very_high"]
        & threat_levels["medium"]
        & vulnerability_levels["medium"],
        risk["high"],
    ),
    ctrl.Rule(
        asset_values["low"] & threat_levels["medium"] & vulnerability_levels["high"],
        risk["low"],
    ),
    ctrl.Rule(
        asset_values["medium"] & threat_levels["medium"] & vulnerability_levels["high"],
        risk["medium"],
    ),
    ctrl.Rule(
        asset_values["high"] & threat_levels["medium"] & vulnerability_levels["high"],
        risk["high"],
    ),
    ctrl.Rule(
        asset_values["very_high"]
        & threat_levels["medium"]
        & vulnerability_levels["high"],
        risk["very_high"],
    ),
    ctrl.Rule(
        asset_values["low"] & threat_levels["high"] & vulnerability_levels["low"],
        risk["very_low"],
    ),
    ctrl.Rule(
        asset_values["medium"] & threat_levels["high"] & vulnerability_levels["low"],
        risk["low"],
    ),
    ctrl.Rule(
        asset_values["high"] & threat_levels["high"] & vulnerability_levels["low"],
        risk["medium"],
    ),
    ctrl.Rule(
        asset_values["very_high"] & threat_levels["high"] & vulnerability_levels["low"],
        risk["medium"],
    ),
    ctrl.Rule(
        asset_values["low"] & threat_levels["high"] & vulnerability_levels["medium"],
        risk["low"],
    ),
    ctrl.Rule(
        asset_values["medium"] & threat_levels["high"] & vulnerability_levels["medium"],
        risk["medium"],
    ),
    ctrl.Rule(
        asset_values["high"] & threat_levels["high"] & vulnerability_levels["medium"],
        risk["high"],
    ),
    ctrl.Rule(
        asset_values["very_high"]
        & threat_levels["high"]
        & vulnerability_levels["medium"],
        risk["very_high"],
    ),
    ctrl.Rule(
        asset_values["low"] & threat_levels["high"] & vulnerability_levels["high"],
        risk["medium"],
    ),
    ctrl.Rule(
        asset_values["medium"] & threat_levels["high"] & vulnerability_levels["high"],
        risk["high"],
    ),
    ctrl.Rule(
        asset_values["high"] & threat_levels["high"] & vulnerability_levels["high"],
        risk["very_high"],
    ),
    ctrl.Rule(
        asset_values["very_high"]
        & threat_levels["high"]
        & vulnerability_levels["high"],
        risk["very_high"],
    ),
]

"""
Step 4: Create the Fuzzy Control System
"""
risk_assessment_ctrl_sys = ctrl.ControlSystem(rules)
risk_assessment_ctrl_sim = ctrl.ControlSystemSimulation(risk_assessment_ctrl_sys)

"""
Step 5: Calculate the Risk
"""
# For user input
while True:
    try:
        # Input values
        asset_value = float(input("Enter asset value (1-4): "))
        threat_level = float(input("Enter threat level (1-3): "))
        vulnerability_level = float(input("Enter vulnerability level (1-3): "))

        risk_assessment_ctrl_sim.input["asset_values"] = asset_value
        risk_assessment_ctrl_sim.input["threat_levels"] = threat_level
        risk_assessment_ctrl_sim.input["vulnerability_levels"] = vulnerability_level

        # Calculate the risk
        risk_assessment_ctrl_sim.compute()

        # Print the risk
        print("Risk:", risk_assessment_ctrl_sim.output["risk"])
        risk.view(sim=risk_assessment_ctrl_sim)
        time = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(
            os.path.join(
                LOG_DIR,
                f"risk_assessment_{asset_value}_{threat_level}_{vulnerability_level}_{time}.png",
            )
        )

    except ValueError:
        print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        break
