"""mito_menu_architect.py

A prototype script that bridges the 33-state Digital Twin with 
real-world meal planning. 

Features:
1. Bio-state integration (Oura/CGM proxies)
2. Source-aware inventory (Homestead, Essex Farm, Co-op)
3. Target-mapped recipe database
"""

import random

# ── 1. THE MITO-RECIPE DATABASE ──────────────────────────────────────────────
# Mapped to specific 33-state variables
RECIPES = [
    {
        "name": "Scholar's Brain Bowl",
        "targets": ["SS", "ATP", "M"], # Synaptic Strength, Energy, Gut
        "ingredients": ["2 Home Eggs", "Essex Farm Sauteed Greens", "Co-op Olive Oil", "Essex Farm Kimchi"],
        "logic": "High Choline for synapses, Fiber for gut, Polyphenols for inflammation."
    },
    {
        "name": "Orchard Senolytic Crumble",
        "targets": ["sen_fraction", "ROS"], # Cellular Cleaning, ROS reduction
        "ingredients": ["2 Orchard Apples (frozen/fresh)", "Co-op Walnuts", "Cinnamon", "Essex Farm Butter"],
        "logic": "Quercetin pulse from apple skins + high Omega-3 from walnuts."
    },
    {
        "name": "Keto-Guard Beef Stew",
        "targets": ["IS", "BDNF", "L"], # Insulin Sensitivity, Muscle/BDNF, Liver
        "ingredients": ["Essex Farm Grass-fed Beef", "Root Vegetables (limited)", "Bone Broth", "Rosemary"],
        "logic": "Ketogenic fuel + Carnosine for muscle signaling + Liver support."
    },
    {
        "name": "Glymphatic Flush Salmon",
        "targets": ["sleep_quality", "Ab"], # Sleep, Amyloid clearance
        "ingredients": ["Wild Salmon (Tops)", "Asparagus", "Co-op Pumpkin Seeds (Magnesium)"],
        "logic": "High Magnesium for GABA/Deep Sleep + DHA for amyloid clearance."
    }
]

# ── 2. ARCHITECT LOGIC ───────────────────────────────────────────────────────

def generate_weekly_plan(bio_stress_level=0.5, insulin_resistance=0.2):
    """
    Generates a meal plan based on current bio-data.
    """
    available_recipes = list(RECIPES)
    if insulin_resistance > 0.5:
        available_recipes.sort(key=lambda x: "IS" in x["targets"], reverse=True)
    elif bio_stress_level > 0.5:
        available_recipes.sort(key=lambda x: "sleep_quality" in x["targets"], reverse=True)
    return available_recipes

# ── 3. THE TRIPLE-THREAT SHOPPING LIST ───────────────────────────────────────

def generate_shopping_list(meal_plan):
    """Categorizes ingredients by sourcing tier."""
    homestead = set()
    essex_farm = set()
    co_op = set()
    tops = set()
    for recipe in meal_plan:
        for ing in recipe["ingredients"]:
            if "Home" in ing or "Orchard" in ing: homestead.add(ing)
            elif "Essex Farm" in ing: essex_farm.add(ing)
            elif "Co-op" in ing: co_op.add(ing)
            else: tops.add(ing)
    return {
        "HOMESTEAD (Inventory Check)": homestead,
        "ESSEX FARM (Primary Fuel)": essex_farm,
        "MIDDLEBURY CO-OP (Shields & Bulk)": co_op,
        "TOPS (Staples)": tops
    }

# ── 4. RUN PROTOTYPE ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Mito-Menu Architect Prototype")
    print("-" * 30)
    print("Bio-State Input: High Stress (0.8), Moderate Insulin Spike (0.4)")
    plan = generate_weekly_plan(bio_stress_level=0.8, insulin_resistance=0.4)
    print("\nPROPOSED MEAL PLAN (Targeted to your 33-state model):")
    for r in plan:
        print(f"- {r['name']}: {r['logic']}")
    list_obj = generate_shopping_list(plan)
    print("\nTRIPLE-THREAT SHOPPING LIST:")
    for tier, items in list_obj.items():
        print(f"\n[{tier}]")
        for item in items:
            print(f"  - {item}")
