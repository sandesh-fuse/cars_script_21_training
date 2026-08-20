# Worst-case feature correlation report ($100-2.5K band)


## ALL notably-bad rows (worst (dollar-top-N union pct-top-N) vs rest of band)

| field | signal |
|---|---|
| clean_title | true: worst=94.4% vs rest=96.3% (lift -2.0pp) |
| is_title_clear | true: worst=45.8% vs rest=46.7% (lift -0.9pp) |
| true_mileage_unknown | true: worst=17.7% vs rest=12.9% (lift +4.8pp) |
| is_confirm_nameontitle | true: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| days_in_title_issues | mean: worst=233.5 vs rest=233.9; missing: worst=89.6% vs rest=92.9% |
| title_start_date | missing: worst=89.2% vs rest=92.6% (lift -3.5pp) |
| title_end_date | missing: worst=89.6% vs rest=92.8% (lift -3.3pp) |
| title_keys_poc | missing: worst=81.1% vs rest=83.6% (lift -2.4pp) |
| title_and_keyspoc | missing: worst=92.8% vs rest=94.4% (lift -1.7pp) |
| ids_on_title | missing: worst=16.1% vs rest=15.7% (lift +0.4pp) |
| name_on_title | missing: worst=0.0% vs rest=0.0% (lift -0.0pp) |
| nav_condition | unknown: worst=1.2% vs rest=1.2%; top-lift categories: Runs & Drives: worst=66.3% vs rest=52.1%; Doesn’t Run / Can be Moved: worst=14.1% vs rest=24.6%; Doesn’t Run / Doesn’t Move: worst=2.0% vs rest=5.3% |
| bodypaintcondition | unknown: worst=0.4% vs rest=0.7%; top-lift categories: Normal Wear & Tear (all body panels intact & attached): worst=80.7% vs rest=75.0%; Baseball-sized or Larger Damage*: worst=10.4% vs rest=13.0%; Some Mirrors, Glass, or Lights are Cracked/Missing: worst=3.6% vs rest=6.1% |
| enginecondition | unknown: worst=9.6% vs rest=17.0%; top-lift categories: Operational: worst=70.7% vs rest=53.1%; Unknown: worst=9.6% vs rest=17.0%; Major Malfunction / Still Installed: worst=8.0% vs rest=13.7% |
| transmissioncondition | unknown: worst=9.6% vs rest=18.3%; top-lift categories: Operational: worst=79.1% vs rest=66.5%; Unknown: worst=9.6% vs rest=18.3%; Major Malfunction / Still Installed: worst=4.8% vs rest=6.4% |
| tirecondition | unknown: worst=2.0% vs rest=1.4%; top-lift categories: 1 or More Tires are Flat*: worst=3.6% vs rest=7.0%; All Wheels Mounted & Tires Inflated: worst=94.0% vs rest=90.8%; Unknown: worst=2.0% vs rest=1.4% |
| interiorcondition | unknown: worst=1.6% vs rest=1.7%; top-lift categories: Normal Wear & Tear (all interior intact & attached): worst=93.2% vs rest=89.6%; Damaged or Removed Parts (notes required): worst=5.2% vs rest=8.7%; Unknown: worst=1.6% vs rest=1.7% |
| other_damages | has-any: worst=8.4% vs rest=15.2%; top-lift tokens: Mold: worst=28.6% vs rest=13.2%; Severe Undercarriage Rust: worst=9.5% vs rest=16.0%; Won’t Pass Smog/State Inspection: worst=23.8% vs rest=28.2%; Other*: worst=33.3% vs rest=35.4%; Flood Damage: worst=0.0% vs rest=1.6% |
| gvm_range | missing: worst=100.0% vs rest=99.9% (lift +0.1pp) |
| tonnage | mean: worst=n/a vs rest=n/a; missing: worst=100.0% vs rest=100.0% |
| engine_type | missing: worst=100.0% vs rest=99.8% (lift +0.2pp) |

## OVER-predicted (model too high) (overpredicted worst rows vs rest of band)

| field | signal |
|---|---|
| clean_title | true: worst=94.4% vs rest=96.3% (lift -2.0pp) |
| is_title_clear | true: worst=45.6% vs rest=46.7% (lift -1.1pp) |
| true_mileage_unknown | true: worst=17.7% vs rest=12.9% (lift +4.8pp) |
| is_confirm_nameontitle | true: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| days_in_title_issues | mean: worst=233.5 vs rest=233.9; missing: worst=89.5% vs rest=92.9% |
| title_start_date | missing: worst=89.1% vs rest=92.6% (lift -3.5pp) |
| title_end_date | missing: worst=89.5% vs rest=92.8% (lift -3.3pp) |
| title_keys_poc | missing: worst=81.0% vs rest=83.6% (lift -2.5pp) |
| title_and_keyspoc | missing: worst=92.7% vs rest=94.4% (lift -1.7pp) |
| ids_on_title | missing: worst=16.1% vs rest=15.7% (lift +0.4pp) |
| name_on_title | missing: worst=0.0% vs rest=0.0% (lift -0.0pp) |
| nav_condition | unknown: worst=1.2% vs rest=1.2%; top-lift categories: Runs & Drives: worst=66.5% vs rest=52.1%; Doesn’t Run / Can be Moved: worst=13.7% vs rest=24.7%; Doesn’t Run / Doesn’t Move: worst=2.0% vs rest=5.3% |
| bodypaintcondition | unknown: worst=0.4% vs rest=0.7%; top-lift categories: Normal Wear & Tear (all body panels intact & attached): worst=80.6% vs rest=75.0%; Baseball-sized or Larger Damage*: worst=10.5% vs rest=13.0%; Some Mirrors, Glass, or Lights are Cracked/Missing: worst=3.6% vs rest=6.1% |
| enginecondition | unknown: worst=9.3% vs rest=17.0%; top-lift categories: Operational: worst=71.0% vs rest=53.1%; Unknown: worst=9.3% vs rest=17.0%; Major Malfunction / Still Installed: worst=8.1% vs rest=13.7% |
| transmissioncondition | unknown: worst=9.3% vs rest=18.3%; top-lift categories: Operational: worst=79.4% vs rest=66.5%; Unknown: worst=9.3% vs rest=18.3%; Major Malfunction / Still Installed: worst=4.8% vs rest=6.4% |
| tirecondition | unknown: worst=2.0% vs rest=1.4%; top-lift categories: 1 or More Tires are Flat*: worst=3.6% vs rest=7.0%; All Wheels Mounted & Tires Inflated: worst=94.0% vs rest=90.8%; Unknown: worst=2.0% vs rest=1.4% |
| interiorcondition | unknown: worst=1.6% vs rest=1.7%; top-lift categories: Normal Wear & Tear (all interior intact & attached): worst=93.1% vs rest=89.6%; Damaged or Removed Parts (notes required): worst=5.2% vs rest=8.7%; Unknown: worst=1.6% vs rest=1.7% |
| other_damages | has-any: worst=8.5% vs rest=15.2%; top-lift tokens: Mold: worst=28.6% vs rest=13.2%; Severe Undercarriage Rust: worst=9.5% vs rest=16.0%; Won’t Pass Smog/State Inspection: worst=23.8% vs rest=28.2%; Other*: worst=33.3% vs rest=35.4%; Flood Damage: worst=0.0% vs rest=1.6% |
| gvm_range | missing: worst=100.0% vs rest=99.9% (lift +0.1pp) |
| tonnage | mean: worst=n/a vs rest=n/a; missing: worst=100.0% vs rest=100.0% |
| engine_type | missing: worst=100.0% vs rest=99.8% (lift +0.2pp) |

## UNDER-predicted (model too low) (underpredicted worst rows vs rest of band)

| field | signal |
|---|---|
| clean_title | true: worst=100.0% vs rest=96.3% (lift +3.7pp) |
| is_title_clear | true: worst=100.0% vs rest=46.7% (lift +53.3pp) |
| true_mileage_unknown | true: worst=0.0% vs rest=13.0% (lift -13.0pp) |
| is_confirm_nameontitle | true: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| days_in_title_issues | mean: worst=n/a vs rest=233.9; missing: worst=100.0% vs rest=92.8% |
| title_start_date | missing: worst=100.0% vs rest=92.5% (lift +7.5pp) |
| title_end_date | missing: worst=100.0% vs rest=92.8% (lift +7.2pp) |
| title_keys_poc | missing: worst=100.0% vs rest=83.5% (lift +16.5pp) |
| title_and_keyspoc | missing: worst=100.0% vs rest=94.4% (lift +5.6pp) |
| ids_on_title | missing: worst=0.0% vs rest=15.7% (lift -15.7pp) |
| name_on_title | missing: worst=0.0% vs rest=0.0% (lift -0.0pp) |
| nav_condition | unknown: worst=0.0% vs rest=1.2%; top-lift categories: Doesn’t Run / Can be Moved: worst=100.0% vs rest=24.4%; Runs & Drives: worst=0.0% vs rest=52.4%; Runs & Moves / Don’t Drive: worst=0.0% vs rest=10.5% |
| bodypaintcondition | unknown: worst=0.0% vs rest=0.7%; top-lift categories: Normal Wear & Tear (all body panels intact & attached): worst=100.0% vs rest=75.1%; Baseball-sized or Larger Damage*: worst=0.0% vs rest=13.0%; Some Mirrors, Glass, or Lights are Cracked/Missing: worst=0.0% vs rest=6.1% |
| enginecondition | unknown: worst=100.0% vs rest=16.9%; top-lift categories: Unknown: worst=100.0% vs rest=16.9%; Operational: worst=0.0% vs rest=53.5%; Minor Issues / Still Functional: worst=0.0% vs rest=14.9% |
| transmissioncondition | unknown: worst=100.0% vs rest=18.1%; top-lift categories: Unknown: worst=100.0% vs rest=18.1%; Operational: worst=0.0% vs rest=66.7%; Minor Issues / Still Functional: worst=0.0% vs rest=7.3% |
| tirecondition | unknown: worst=0.0% vs rest=1.4%; top-lift categories: All Wheels Mounted & Tires Inflated: worst=100.0% vs rest=90.9%; 1 or More Tires are Flat*: worst=0.0% vs rest=7.0%; Unknown: worst=0.0% vs rest=1.4% |
| interiorcondition | unknown: worst=0.0% vs rest=1.7%; top-lift categories: Normal Wear & Tear (all interior intact & attached): worst=100.0% vs rest=89.6%; Damaged or Removed Parts (notes required): worst=0.0% vs rest=8.6%; Unknown: worst=0.0% vs rest=1.7% |
| other_damages | has-any: worst=0.0% vs rest=15.0%; top-lift tokens: Other*: worst=0.0% vs rest=35.3%; Won’t Pass Smog/State Inspection: worst=0.0% vs rest=28.1%; Severe Undercarriage Rust: worst=0.0% vs rest=16.0%; Mold: worst=0.0% vs rest=13.4%; Air Bag(s) Deployed: worst=0.0% vs rest=4.6% |
| gvm_range | missing: worst=100.0% vs rest=99.9% (lift +0.1pp) |
| tonnage | mean: worst=n/a vs rest=n/a; missing: worst=100.0% vs rest=100.0% |
| engine_type | missing: worst=100.0% vs rest=99.8% (lift +0.2pp) |