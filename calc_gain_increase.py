import numpy as np

leg_length = 0.3
new_leg_r = 0.091
og_leg_r = 0.065

arm_length = 0.25
new_arm_r = 0.063
og_arm_r = 0.035


def get_mass_scale(L, old_r, new_r):
    old_volume = np.pi * old_r ** 2 * L + 4 / 3 * np.pi * old_r ** 3
    new_volume = np.pi * new_r ** 2 * L + 4 / 3 * np.pi * new_r ** 3

    return new_volume / old_volume

print(f"Leg: {get_mass_scale(leg_length, og_leg_r, new_leg_r)}")
print(f"Arm: {get_mass_scale(arm_length, og_arm_r, new_arm_r)}")