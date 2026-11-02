# BEHAVIOR task prompts

The online agents and the program backend read a task prompt from `<task>.txt` in this directory
(or from the directory named by `REA_BEHAVIOR_PROMPTS`). REA does not redistribute task prompts.
For the ASPIRE setting, copy the `prompt` field of your ASPIRE install's
`env_configs/r1pro/r1pro_pick_up_<task>_aspire.yaml` into `<task>.txt`, or write your own.

Without a prompt file the agent receives a one-line goal built from the task instruction.
