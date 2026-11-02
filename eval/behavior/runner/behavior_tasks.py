"""Task table shared by the BEHAVIOR runner and program backend: five ASPIRE-style navigate-and-pick-up tasks on
BEHAVIOR-1K R1Pro (2025-challenge instances; seed = instance id). Success = an object of the target category held in the
controller's current gripper and lifted (ASPIRE's own predicate for trash/radio, the same rule generalised for the new tasks)."""
TASKS5 = {
    "trash": {"config": "env_configs/r1pro/r1pro_pick_up_trash_oracle.yaml", "activity": "picking_up_trash", "scene": "house_double_floor_lower",
              "object": "blue can of soda", "surface": "floor", "categories": ["can_of_soda"],
              "instruction": "Pick up the blue can of soda on the floor.",
              "language": "Pick up the blue can of soda from the floor and hold it in the gripper."},
    "radio": {"config": "env_configs/r1pro/r1pro_pick_up_radio_oracle.yaml", "activity": "turning_on_radio", "scene": "house_double_floor_lower",
              "object": "red radio", "surface": "table", "categories": ["radio"],
              "instruction": "Pick up the red radio on the table.",
              "language": "Pick up the red radio from the table and hold it in the gripper."},
    "teddy": {"config": "env_configs/r1pro/r1pro_pick_up_teddy_oracle.yaml", "activity": "collecting_childrens_toys", "scene": "house_single_floor",
              "object": "teddy bear", "surface": "floor", "categories": ["teddy_bear"],
              "instruction": "Pick up the teddy bear on the floor.",
              "language": "Pick up the teddy bear from the floor and hold it in the gripper."},
    "drill": {"config": "env_configs/r1pro/r1pro_pick_up_drill_oracle.yaml", "activity": "outfit_a_basic_toolbox", "scene": "house_single_floor",
              "object": "power drill", "surface": "workbench table", "categories": ["drill"],
              "instruction": "Pick up the power drill on the workbench table.",
              "language": "Pick up the power drill from the workbench table and hold it in the gripper."},
    "candle": {"config": "env_configs/r1pro/r1pro_pick_up_candle_oracle.yaml", "activity": "assembling_gift_baskets", "scene": "house_double_floor_lower",
               "object": "pillar candle", "surface": "table", "categories": ["pillar_candle"],
               "instruction": "Pick up a pillar candle on the table.",
               "language": "Pick up a pillar candle from the table and hold it in the gripper."},

    "shoe": {"config": "env_configs/r1pro/r1pro_pick_up_shoe_oracle.yaml", "activity": "putting_shoes_on_rack", "scene": "house_double_floor_lower",
             "object": "gym shoe", "surface": "floor", "categories": ["gym_shoe"],
             "instruction": "Pick up a gym shoe on the floor.",
             "language": "Pick up a gym shoe from the floor and hold it in the gripper."},
    "camera": {"config": "env_configs/r1pro/r1pro_pick_up_camera_oracle.yaml", "activity": "attach_a_camera_to_a_tripod", "scene": "house_double_floor_upper",
               "object": "digital camera", "surface": "floor", "categories": ["digital_camera"],
               "instruction": "Pick up the digital camera on the floor.",
               "language": "Pick up the digital camera from the floor and hold it in the gripper."},
}
# TASKS5_ACTIVE is the five-task set. "oil" (storing_food) and "candle" (assembling_gift_baskets) are listed but inactive: curobo's
# warm-up can hang in CUDA-graph capture for those object-rich scenes; "shoe" is available as a spare.
TASKS5_ACTIVE = ("trash", "radio", "teddy", "drill", "camera")
CONFIGS = {k: v["config"] for k, v in TASKS5.items()}
INSTRUCTIONS = {k: v["instruction"] for k, v in TASKS5.items()}
CRITERION = {k: (f"The instruction is complete when the {v['object']} is held in one of the robot's grippers, lifted off the {v['surface']}. "
                 "It does NOT have to be placed anywhere; holding it up counts as done.") for k, v in TASKS5.items()}
