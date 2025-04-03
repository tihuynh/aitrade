import json
with open("states.json") as f:
    data = json.load(f)
for state in data["states"]:
    # print(state)
    # print (state["name"])
    # print(state["abbreviation"])
    # print(state["area_codes"])
    del state["area_codes"]
with open("states_copy.json","w+") as wf:
    json.dump(data,wf, indent=2)