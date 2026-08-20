# Worst-case feature correlation report ($100-2.5K band)


## ALL notably-bad rows (worst (dollar-top-N union pct-top-N) vs rest of band)

| field | signal |
|---|---|
| clean_title | true: worst=99.4% vs rest=97.0% (lift +2.3pp) |
| is_title_clear | true: worst=43.2% vs rest=52.1% (lift -8.8pp) |
| true_mileage_unknown | true: worst=30.3% vs rest=8.4% (lift +21.9pp) |
| is_confirm_nameontitle | true: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| days_in_title_issues | mean: worst=228.7 vs rest=235.2; missing: worst=91.6% vs rest=92.9% |
| title_start_date | missing: worst=91.6% vs rest=92.8% (lift -1.1pp) |
| title_end_date | missing: worst=91.6% vs rest=92.9% (lift -1.3pp) |
| title_keys_poc | missing: worst=81.9% vs rest=86.3% (lift -4.4pp) |
| title_and_keyspoc | missing: worst=96.8% vs rest=94.5% (lift +2.2pp) |
| ids_on_title | missing: worst=15.5% vs rest=9.1% (lift +6.3pp) |
| name_on_title | missing: worst=0.0% vs rest=0.0% (lift +0.0pp) |
| nav_condition | unknown: worst=1.3% vs rest=0.4%; top-lift categories: Runs & Drives: worst=48.4% vs rest=68.2%; Doesn’t Run / Can be Moved: worst=31.0% vs rest=15.2%; Doesn’t Run / Doesn’t Move: worst=6.5% vs rest=1.8% |
| bodypaintcondition | unknown: worst=1.9% vs rest=0.1%; top-lift categories: Baseball-sized or Larger Damage*: worst=8.4% vs rest=10.2%; Unknown: worst=1.9% vs rest=0.1%; Major Damage*: worst=0.6% vs rest=1.5% |
| enginecondition | unknown: worst=27.7% vs rest=8.2%; top-lift categories: Unknown: worst=27.7% vs rest=8.2%; Operational: worst=49.7% vs rest=68.8%; Major Malfunction / Still Installed: worst=12.3% vs rest=9.5% |
| transmissioncondition | unknown: worst=29.7% vs rest=9.0%; top-lift categories: Operational: worst=58.1% vs rest=80.8%; Unknown: worst=29.7% vs rest=9.0%; Major Malfunction / Still Installed: worst=7.7% vs rest=3.8% |
| tirecondition | unknown: worst=3.9% vs rest=0.5%; top-lift categories: All Wheels Mounted & Tires Inflated: worst=89.7% vs rest=95.6%; Unknown: worst=3.9% vs rest=0.5%; 1 or More Tires are Flat*: worst=5.8% vs rest=3.4% |
| interiorcondition | unknown: worst=1.9% vs rest=0.4%; top-lift categories: Normal Wear & Tear (all interior intact & attached): worst=90.3% vs rest=94.3%; Damaged or Removed Parts (notes required): worst=7.7% vs rest=5.3%; Unknown: worst=1.9% vs rest=0.4% |
| other_damages | has-any: worst=11.6% vs rest=12.2%; top-lift tokens: Other*: worst=16.7% vs rest=47.9%; Severe Undercarriage Rust: worst=38.9% vs rest=12.0%; Won’t Pass Smog/State Inspection: worst=22.2% vs rest=32.3%; Air Bag(s) Deployed: worst=11.1% vs rest=3.1%; Mold: worst=11.1% vs rest=3.6% |
| gvm_range | missing: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| tonnage | mean: worst=n/a vs rest=n/a; missing: worst=100.0% vs rest=100.0% |
| engine_type | missing: worst=100.0% vs rest=99.9% (lift +0.1pp) |

## OVER-predicted (model too high) (overpredicted worst rows vs rest of band)

| field | signal |
|---|---|
| clean_title | true: worst=100.0% vs rest=97.2% (lift +2.8pp) |
| is_title_clear | true: worst=50.0% vs rest=51.3% (lift -1.3pp) |
| true_mileage_unknown | true: worst=25.0% vs rest=10.2% (lift +14.8pp) |
| is_confirm_nameontitle | true: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| days_in_title_issues | mean: worst=230.0 vs rest=234.5; missing: worst=91.7% vs rest=92.8% |
| title_start_date | missing: worst=91.7% vs rest=92.7% (lift -1.0pp) |
| title_end_date | missing: worst=91.7% vs rest=92.8% (lift -1.1pp) |
| title_keys_poc | missing: worst=83.3% vs rest=85.9% (lift -2.6pp) |
| title_and_keyspoc | missing: worst=100.0% vs rest=94.7% (lift +5.3pp) |
| ids_on_title | missing: worst=0.0% vs rest=9.8% (lift -9.8pp) |
| name_on_title | missing: worst=0.0% vs rest=0.0% (lift +0.0pp) |
| nav_condition | unknown: worst=0.0% vs rest=0.5%; top-lift categories: Runs & Drives: worst=75.0% vs rest=66.4%; Runs / Doesn’t Move: worst=0.0% vs rest=2.4%; Doesn’t Run / Doesn’t Move: worst=0.0% vs rest=2.3% |
| bodypaintcondition | unknown: worst=0.0% vs rest=0.3%; top-lift categories: Normal Wear & Tear (all body panels intact & attached): worst=91.7% vs rest=82.3%; Some Mirrors, Glass, or Lights are Cracked/Missing: worst=0.0% vs rest=3.8%; Loose or Missing Panels*: worst=0.0% vs rest=2.2% |
| enginecondition | unknown: worst=0.0% vs rest=10.0%; top-lift categories: Operational: worst=83.3% vs rest=66.9%; Unknown: worst=0.0% vs rest=10.0%; Minor Issues / Still Functional: worst=8.3% vs rest=12.5% |
| transmissioncondition | unknown: worst=0.0% vs rest=10.9%; top-lift categories: Operational: worst=100.0% vs rest=78.6%; Unknown: worst=0.0% vs rest=10.9%; Minor Issues / Still Functional: worst=0.0% vs rest=5.2% |
| tirecondition | unknown: worst=0.0% vs rest=0.8%; top-lift categories: All Wheels Mounted & Tires Inflated: worst=100.0% vs rest=95.0%; 1 or More Tires are Flat*: worst=0.0% vs rest=3.7%; Unknown: worst=0.0% vs rest=0.8% |
| interiorcondition | unknown: worst=0.0% vs rest=0.6%; top-lift categories: Normal Wear & Tear (all interior intact & attached): worst=100.0% vs rest=93.9%; Damaged or Removed Parts (notes required): worst=0.0% vs rest=5.5%; Unknown: worst=0.0% vs rest=0.6% |
| other_damages | has-any: worst=8.3% vs rest=12.2%; top-lift tokens: Severe Undercarriage Rust: worst=100.0% vs rest=13.9%; Other*: worst=0.0% vs rest=45.5%; Won’t Pass Smog/State Inspection: worst=0.0% vs rest=31.6%; Mold: worst=0.0% vs rest=4.3%; Air Bag(s) Deployed: worst=0.0% vs rest=3.8% |
| gvm_range | missing: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| tonnage | mean: worst=n/a vs rest=n/a; missing: worst=100.0% vs rest=100.0% |
| engine_type | missing: worst=100.0% vs rest=99.9% (lift +0.1pp) |

## UNDER-predicted (model too low) (underpredicted worst rows vs rest of band)

| field | signal |
|---|---|
| clean_title | true: worst=99.3% vs rest=97.0% (lift +2.3pp) |
| is_title_clear | true: worst=42.7% vs rest=52.0% (lift -9.4pp) |
| true_mileage_unknown | true: worst=30.8% vs rest=8.5% (lift +22.3pp) |
| is_confirm_nameontitle | true: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| days_in_title_issues | mean: worst=228.6 vs rest=235.1; missing: worst=91.6% vs rest=92.9% |
| title_start_date | missing: worst=91.6% vs rest=92.8% (lift -1.1pp) |
| title_end_date | missing: worst=91.6% vs rest=92.9% (lift -1.3pp) |
| title_keys_poc | missing: worst=81.8% vs rest=86.3% (lift -4.4pp) |
| title_and_keyspoc | missing: worst=96.5% vs rest=94.6% (lift +1.9pp) |
| ids_on_title | missing: worst=16.8% vs rest=9.1% (lift +7.7pp) |
| name_on_title | missing: worst=0.0% vs rest=0.0% (lift +0.0pp) |
| nav_condition | unknown: worst=1.4% vs rest=0.4%; top-lift categories: Runs & Drives: worst=46.2% vs rest=68.2%; Doesn’t Run / Can be Moved: worst=32.2% vs rest=15.2%; Doesn’t Run / Doesn’t Move: worst=7.0% vs rest=1.8% |
| bodypaintcondition | unknown: worst=2.1% vs rest=0.1%; top-lift categories: Unknown: worst=2.1% vs rest=0.1%; Baseball-sized or Larger Damage*: worst=8.4% vs rest=10.2%; Major Damage*: worst=0.7% vs rest=1.4% |
| enginecondition | unknown: worst=30.1% vs rest=8.1%; top-lift categories: Operational: worst=46.9% vs rest=68.9%; Unknown: worst=30.1% vs rest=8.1%; Major Malfunction / Still Installed: worst=12.6% vs rest=9.5% |
| transmissioncondition | unknown: worst=32.2% vs rest=8.9%; top-lift categories: Operational: worst=54.5% vs rest=81.0%; Unknown: worst=32.2% vs rest=8.9%; Major Malfunction / Still Installed: worst=8.4% vs rest=3.8% |
| tirecondition | unknown: worst=4.2% vs rest=0.5%; top-lift categories: All Wheels Mounted & Tires Inflated: worst=88.8% vs rest=95.6%; Unknown: worst=4.2% vs rest=0.5%; 1 or More Tires are Flat*: worst=6.3% vs rest=3.4% |
| interiorcondition | unknown: worst=2.1% vs rest=0.4%; top-lift categories: Normal Wear & Tear (all interior intact & attached): worst=89.5% vs rest=94.3%; Damaged or Removed Parts (notes required): worst=8.4% vs rest=5.2%; Unknown: worst=2.1% vs rest=0.4% |
| other_damages | has-any: worst=11.9% vs rest=12.2%; top-lift tokens: Other*: worst=17.6% vs rest=47.7%; Severe Undercarriage Rust: worst=35.3% vs rest=12.4%; Air Bag(s) Deployed: worst=11.8% vs rest=3.1%; Won’t Pass Smog/State Inspection: worst=23.5% vs rest=32.1%; Mold: worst=11.8% vs rest=3.6% |
| gvm_range | missing: worst=100.0% vs rest=100.0% (lift +0.0pp) |
| tonnage | mean: worst=n/a vs rest=n/a; missing: worst=100.0% vs rest=100.0% |
| engine_type | missing: worst=100.0% vs rest=99.9% (lift +0.1pp) |