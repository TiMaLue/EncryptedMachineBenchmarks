import json

with open("protos1.json", "r") as fp:
    protos = json.load(fp)

# print([p["name"] for p in protos])

print("\\textbf{Name}   & \\textbf{\\#Parties}  & \\textbf{\\#Corruptions} & \\textbf{Adversarial Behavior} & \\textbf{Computational Model}\\\\ \\hline")

for p in protos:
    name = p["name"]
    party_num = p["party_num"]
    if party_num == -1:
        parties = "\\(\\geq 2\\)"
    elif party_num == -3:
        parties = "\\(\\geq 3\\)"
    else:
        parties = f"\\({party_num}\\)"
    corruptions = "\\(< n\\)"
    if not p["dishonest_maj"]:
        corruptions = "\\(< n/2\\)"
    adv = "malicious"
    if  p["is_covert"]:
        adv = "covert"
    elif p["is_semi"]:
        adv = "passive"
    comp_model = "Arithmetic (field)"
    if p["is_ring"]:
        comp_model = "Arithmetic (ring)"
    if p["is_binary"]:
        comp_model = "Binary"
    print(
        f"{name} &"
        f"{parties} &"
        f"{corruptions} &"
        f"{adv} &"
        f"{comp_model}"
        " \\\\ \\hline")
