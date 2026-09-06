"""Write CLIP text descriptions for the 31 Toyota Smarthome classes.

The ETRI pipeline reads an xlsx with an ``action_description`` (short) and a
``global_description`` (a sentence). Smarthome ships neither, so the sentences
are written here and exported in the same two-column shape, which lets the rest
of the pipeline - including the eight-way prompt ensembling - run unchanged.

Wording follows the ETRI descriptions: third person, present tense, one clause
naming the objects involved, since those are what the object stream keys on.

Usage:
  cd /workspace/VPOCLIP_plus
  python tools/build_smarthome_descriptions.py
"""

import argparse
import json
from pathlib import Path

# action name -> (short phrase, full sentence)
DESCRIPTIONS = {
    "Cook.Cleandishes": ("cleaning dishes", "A person cleans dishes at the kitchen sink, scrubbing plates and cups under running water."),
    "Cook.Cleanup": ("cleaning up the kitchen", "A person tidies the kitchen after cooking, wiping surfaces and putting utensils away."),
    "Cook.Cut": ("cutting food", "A person cuts food on a chopping board with a knife in the kitchen."),
    "Cook.Stir": ("stirring food in a pot", "A person stirs food in a pot on the stove, moving a spoon in circles."),
    "Cook.Usestove": ("using the stove", "A person operates a stove in the kitchen, adjusting the burner while cooking."),
    "Cutbread": ("cutting bread", "A person slices a loaf of bread with a knife on the kitchen counter."),
    "Drink.Frombottle": ("drinking from a bottle", "A person lifts a bottle to the mouth and drinks from it."),
    "Drink.Fromcan": ("drinking from a can", "A person raises a can to the mouth and drinks from it."),
    "Drink.Fromcup": ("drinking from a cup", "A person picks up a cup, brings it to the mouth and drinks."),
    "Drink.Fromglass": ("drinking from a glass", "A person lifts a glass to the mouth and drinks from it."),
    "Eat.Attable": ("eating at a table", "A person sits at a table and eats a meal with cutlery."),
    "Eat.Snack": ("eating a snack", "A person eats a small snack with the hands while standing or sitting."),
    "Enter": ("entering the room", "A person walks through a doorway and enters the room."),
    "Getup": ("getting up", "A person rises from a seat or a bed and stands upright."),
    "Laydown": ("lying down", "A person lies down on a sofa or a bed and settles into a resting position."),
    "Leave": ("leaving the room", "A person walks toward a doorway and leaves the room."),
    "Makecoffee.Pourgrains": ("pouring coffee grains", "A person pours coffee grains into a coffee machine while preparing coffee."),
    "Makecoffee.Pourwater": ("pouring water for coffee", "A person pours water into a coffee machine while preparing coffee."),
    "Maketea.Boilwater": ("boiling water for tea", "A person boils water in a kettle while preparing tea."),
    "Maketea.Insertteabag": ("inserting a tea bag", "A person places a tea bag into a cup while preparing tea."),
    "Pour.Frombottle": ("pouring from a bottle", "A person tilts a bottle and pours liquid into a container."),
    "Pour.Fromcan": ("pouring from a can", "A person tilts a can and pours liquid into a container."),
    "Pour.Fromkettle": ("pouring from a kettle", "A person tilts a kettle and pours hot water into a cup."),
    "Readbook": ("reading a book", "A person holds an open book and reads it while seated."),
    "Sitdown": ("sitting down", "A person lowers the body onto a chair or a sofa and sits down."),
    "Takepills": ("taking pills", "A person takes medicine pills and swallows them, often with a drink."),
    "Uselaptop": ("using a laptop", "A person works on a laptop, typing on the keyboard while seated."),
    "Usetablet": ("using a tablet", "A person holds a tablet computer and taps its screen."),
    "Usetelephone": ("using a telephone", "A person holds a telephone to the ear and talks."),
    "Walk": ("walking", "A person walks across the room."),
    "WatchTV": ("watching television", "A person sits and watches television, facing the screen."),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Smarthome CLIP descriptions.")
    parser.add_argument("--split-dir", default="data/smarthome_splits")
    parser.add_argument("--output", default="data/smarthome_splits/descriptions.json")
    return parser.parse_args()


def main():
    args = parse_args()
    metadata = json.loads((Path(args.split_dir) / "metadata.json").read_text())
    actions = metadata["actions"]

    missing = [a for a in actions if a not in DESCRIPTIONS]
    if missing:
        raise ValueError(f"No description written for: {missing}")

    records = []
    for action in actions:
        short, full = DESCRIPTIONS[action]
        records.append({
            "ID": metadata["action_to_label"][action] + 1,   # xlsx convention: 1-based
            "action": action,
            "action_description": short,
            "global_description": full,
        })

    Path(args.output).write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"{len(records)} descriptions written to {args.output}")
    for record in records[:3]:
        print(f"  {record['ID']:2d} {record['action']:24s} {record['global_description']}")


if __name__ == "__main__":
    main()
