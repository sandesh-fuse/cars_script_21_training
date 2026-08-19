# Worst-case feature correlation report ($100-2.5K band)


## ALL notably-bad rows (worst (dollar-top-N union pct-top-N) vs rest of band)

| field | signal |
|---|---|
| clean_title | true: worst=94.8% vs rest=96.3% (lift -1.6pp) |
| is_title_clear | true: worst=44.2% vs rest=46.9% (lift -2.7pp) |
| true_mileage_unknown | true: worst=20.3% vs rest=12.7% (lift +7.6pp) |
| is_confirm_nameontitle | true: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| days_in_title_issues | mean: worst=230.9 vs rest=233.9; missing: worst=91.3% vs rest=92.8% |
| title_start_date | missing: worst=91.3% vs rest=92.6% (lift -1.3pp) |
| title_end_date | missing: worst=91.3% vs rest=92.8% (lift -1.5pp) |
| title_keys_poc | missing: worst=78.5% vs rest=83.8% (lift -5.3pp) |
| title_and_keyspoc | missing: worst=91.9% vs rest=94.5% (lift -2.6pp) |
| ids_on_title | missing: worst=14.5% vs rest=15.6% (lift -1.1pp) |
| name_on_title | missing: worst=0.0% vs rest=0.0% (lift -0.0pp) |
| nav_condition | unknown: worst=1.2% vs rest=1.2%; top-lift categories: Runs & Drives: worst=65.7% vs rest=53.1%; Doesn’t Run / Can be Moved: worst=12.2% vs rest=24.0%; Doesn’t Run / Doesn’t Move: worst=2.9% vs rest=5.2% |
| bodypaintcondition | unknown: worst=0.6% vs rest=0.7%; top-lift categories: Normal Wear & Tear (all body panels intact & attached): worst=80.2% vs rest=75.4%; Some Mirrors, Glass, or Lights are Cracked/Missing: worst=3.5% vs rest=6.1%; Baseball-sized or Larger Damage*: worst=11.0% vs rest=12.9% |
| enginecondition | unknown: worst=10.5% vs rest=16.5%; top-lift categories: Operational: worst=68.6% vs rest=54.1%; Unknown: worst=10.5% vs rest=16.5%; Major Malfunction / Still Installed: worst=8.7% vs rest=13.4% |
| transmissioncondition | unknown: worst=11.0% vs rest=17.7%; top-lift categories: Operational: worst=77.9% vs rest=67.1%; Unknown: worst=11.0% vs rest=17.7%; Minor Issues / Still Functional: worst=5.2% vs rest=7.3% |
| tirecondition | unknown: worst=2.9% vs rest=1.4%; top-lift categories: 1 or More Tires are Flat*: worst=2.9% vs rest=6.8%; All Wheels Mounted & Tires Inflated: worst=94.2% vs rest=91.1%; Unknown: worst=2.9% vs rest=1.4% |
| interiorcondition | unknown: worst=1.7% vs rest=1.7%; top-lift categories: Damaged or Removed Parts (notes required): worst=4.1% vs rest=8.6%; Normal Wear & Tear (all interior intact & attached): worst=94.2% vs rest=89.7%; Unknown: worst=1.7% vs rest=1.7% |
| other_damages | has-any: worst=6.4% vs rest=15.0%; top-lift tokens: Other*: worst=45.5% vs rest=35.1%; Severe Undercarriage Rust: worst=9.1% vs rest=16.2%; Mold: worst=18.2% vs rest=12.8%; Air Bag(s) Deployed: worst=0.0% vs rest=4.5%; Flood Damage: worst=0.0% vs rest=2.0% |
| gvm_range | missing: worst=100.0% vs rest=99.9% (lift +0.1pp) |
| tonnage | mean: worst=n/a vs rest=n/a; missing: worst=100.0% vs rest=100.0% |
| engine_type | missing: worst=100.0% vs rest=99.8% (lift +0.2pp) |

## OVER-predicted (model too high) (overpredicted worst rows vs rest of band)

| field | signal |
|---|---|
| clean_title | true: worst=94.6% vs rest=96.3% (lift -1.7pp) |
| is_title_clear | true: worst=44.3% vs rest=46.9% (lift -2.6pp) |
| true_mileage_unknown | true: worst=19.2% vs rest=12.8% (lift +6.4pp) |
| is_confirm_nameontitle | true: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| days_in_title_issues | mean: worst=230.9 vs rest=233.9; missing: worst=91.0% vs rest=92.8% |
| title_start_date | missing: worst=91.0% vs rest=92.6% (lift -1.5pp) |
| title_end_date | missing: worst=91.0% vs rest=92.8% (lift -1.8pp) |
| title_keys_poc | missing: worst=78.4% vs rest=83.8% (lift -5.3pp) |
| title_and_keyspoc | missing: worst=92.2% vs rest=94.5% (lift -2.3pp) |
| ids_on_title | missing: worst=15.0% vs rest=15.6% (lift -0.6pp) |
| name_on_title | missing: worst=0.0% vs rest=0.0% (lift -0.0pp) |
| nav_condition | unknown: worst=1.2% vs rest=1.2%; top-lift categories: Runs & Drives: worst=65.3% vs rest=53.1%; Doesn’t Run / Can be Moved: worst=12.0% vs rest=24.0%; Doesn’t Run / Doesn’t Move: worst=3.0% vs rest=5.1% |
| bodypaintcondition | unknown: worst=0.6% vs rest=0.7%; top-lift categories: Normal Wear & Tear (all body panels intact & attached): worst=79.6% vs rest=75.4%; Some Mirrors, Glass, or Lights are Cracked/Missing: worst=3.6% vs rest=6.1%; Baseball-sized or Larger Damage*: worst=11.4% vs rest=12.9% |
| enginecondition | unknown: worst=10.2% vs rest=16.5%; top-lift categories: Operational: worst=70.1% vs rest=54.1%; Unknown: worst=10.2% vs rest=16.5%; Major Malfunction / Still Installed: worst=8.4% vs rest=13.4% |
| transmissioncondition | unknown: worst=9.6% vs rest=17.7%; top-lift categories: Operational: worst=79.0% vs rest=67.0%; Unknown: worst=9.6% vs rest=17.7%; Minor Issues / Still Functional: worst=5.4% vs rest=7.3% |
| tirecondition | unknown: worst=2.4% vs rest=1.4%; top-lift categories: 1 or More Tires are Flat*: worst=3.0% vs rest=6.8%; All Wheels Mounted & Tires Inflated: worst=94.6% vs rest=91.1%; Unknown: worst=2.4% vs rest=1.4% |
| interiorcondition | unknown: worst=1.8% vs rest=1.7%; top-lift categories: Damaged or Removed Parts (notes required): worst=4.2% vs rest=8.6%; Normal Wear & Tear (all interior intact & attached): worst=94.0% vs rest=89.7%; Unknown: worst=1.8% vs rest=1.7% |
| other_damages | has-any: worst=6.6% vs rest=15.0%; top-lift tokens: Other*: worst=45.5% vs rest=35.1%; Severe Undercarriage Rust: worst=9.1% vs rest=16.2%; Mold: worst=18.2% vs rest=12.8%; Air Bag(s) Deployed: worst=0.0% vs rest=4.5%; Flood Damage: worst=0.0% vs rest=2.0% |
| gvm_range | missing: worst=100.0% vs rest=99.9% (lift +0.1pp) |
| tonnage | mean: worst=n/a vs rest=n/a; missing: worst=100.0% vs rest=100.0% |
| engine_type | missing: worst=100.0% vs rest=99.8% (lift +0.2pp) |

## UNDER-predicted (model too low) (underpredicted worst rows vs rest of band)

| field | signal |
|---|---|
| clean_title | true: worst=100.0% vs rest=96.3% (lift +3.7pp) |
| is_title_clear | true: worst=40.0% vs rest=46.9% (lift -6.9pp) |
| true_mileage_unknown | true: worst=60.0% vs rest=12.8% (lift +47.2pp) |
| is_confirm_nameontitle | true: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| days_in_title_issues | mean: worst=n/a vs rest=233.9; missing: worst=100.0% vs rest=92.8% |
| title_start_date | missing: worst=100.0% vs rest=92.5% (lift +7.5pp) |
| title_end_date | missing: worst=100.0% vs rest=92.8% (lift +7.2pp) |
| title_keys_poc | missing: worst=80.0% vs rest=83.7% (lift -3.7pp) |
| title_and_keyspoc | missing: worst=80.0% vs rest=94.5% (lift -14.5pp) |
| ids_on_title | missing: worst=0.0% vs rest=15.6% (lift -15.6pp) |
| name_on_title | missing: worst=0.0% vs rest=0.0% (lift -0.0pp) |
| nav_condition | unknown: worst=0.0% vs rest=1.2%; top-lift categories: Runs & Drives: worst=80.0% vs rest=53.3%; Runs & Moves / Don’t Drive: worst=0.0% vs rest=10.5%; Doesn’t Run / Doesn’t Move: worst=0.0% vs rest=5.1% |
| bodypaintcondition | unknown: worst=0.0% vs rest=0.7%; top-lift categories: Normal Wear & Tear (all body panels intact & attached): worst=100.0% vs rest=75.4%; Baseball-sized or Larger Damage*: worst=0.0% vs rest=12.9%; Some Mirrors, Glass, or Lights are Cracked/Missing: worst=0.0% vs rest=6.0% |
| enginecondition | unknown: worst=20.0% vs rest=16.4%; top-lift categories: Operational: worst=20.0% vs rest=54.3%; Rebuilt/Replaced: worst=20.0% vs rest=1.0%; Major Malfunction / Still Installed: worst=20.0% vs rest=13.3% |
| transmissioncondition | unknown: worst=60.0% vs rest=17.6%; top-lift categories: Unknown: worst=60.0% vs rest=17.6%; Operational: worst=40.0% vs rest=67.2%; Minor Issues / Still Functional: worst=0.0% vs rest=7.2% |
| tirecondition | unknown: worst=20.0% vs rest=1.4%; top-lift categories: Unknown: worst=20.0% vs rest=1.4%; All Wheels Mounted & Tires Inflated: worst=80.0% vs rest=91.1%; 1 or More Tires are Flat*: worst=0.0% vs rest=6.8% |
| interiorcondition | unknown: worst=0.0% vs rest=1.7%; top-lift categories: Normal Wear & Tear (all interior intact & attached): worst=100.0% vs rest=89.8%; Damaged or Removed Parts (notes required): worst=0.0% vs rest=8.6%; Unknown: worst=0.0% vs rest=1.7% |
| other_damages | has-any: worst=0.0% vs rest=14.9%; top-lift tokens: Other*: worst=0.0% vs rest=35.1%; Won’t Pass Smog/State Inspection: worst=0.0% vs rest=28.6%; Severe Undercarriage Rust: worst=0.0% vs rest=16.1%; Mold: worst=0.0% vs rest=12.8%; Air Bag(s) Deployed: worst=0.0% vs rest=4.4% |
| gvm_range | missing: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| tonnage | mean: worst=n/a vs rest=n/a; missing: worst=100.0% vs rest=100.0% |
| engine_type | missing: worst=100.0% vs rest=99.8% (lift +0.2pp) |