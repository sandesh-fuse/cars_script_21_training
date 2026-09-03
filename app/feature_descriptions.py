"""
feature_descriptions.py
========================
Maps preprocessor feature names to human-readable labels used in the
SHAP response and the Granite LLM prompt.

Unknown features fall back to a heuristic that prettifies the column name.
"""

# Direct mapping for features whose machine names don't read well.
HUMAN_READABLE = {
    # Core vehicle
    'make':                       'Make',
    'model':                      'Model',
    'year':                       'Model year',
    'trim':                       'Trim level',
    'vehicle_type':               'Vehicle type',
    'body_type':                  'Body type',
    'body_subtype':               'Body subtype',
    'drive_type':                 'Drive type (FWD/AWD/4WD)',
    'engine_name':                'Engine',
    'transmission_name':          'Transmission',
    'mileage':                    'Mileage (miles driven)',
    'mileage_bucket':             'Mileage bucket',
    'miles_per_year':             'Average miles driven per year',
    'doors':                      'Number of doors',
    'oem_doors':                  'OEM door count',
    'oem_body_style':             'OEM body style',
    'us_style_name':              'US trim/style name',
    'model_number':               'Manufacturer model number',
    'msrp':                       'MSRP (original sticker price)',
    'nav_color':                  'Vehicle color',
    'age':                        'Vehicle age (years)',
    'age_sq':                     'Vehicle age squared',
    'age_bucket':                 'Vehicle age bucket',

    # Conditions
    'nav_condition':              'Driveability condition',
    'bodypaintcondition':         'Body/paint condition',
    'enginecondition':            'Engine condition',
    'transmissioncondition':      'Transmission condition',
    'tirecondition':              'Tire/wheel condition',
    'interiorcondition':          'Interior condition',
    'nav_severity':               'Driveability severity (higher = worse)',
    'body_severity':              'Body damage severity',
    'engine_severity':            'Engine issue severity',
    'trans_severity':             'Transmission issue severity',
    'tire_severity':              'Tire/wheel issue severity',
    'interior_severity':          'Interior damage severity',
    'mechanical_severity_sum':    'Total mechanical severity',
    'mechanical_severity_mean':   'Average mechanical severity',
    'mechanical_severity_max':    'Worst mechanical severity',
    'runs_flag':                  'Vehicle runs (yes/no)',
    'n_unknowns':                 'Number of unknown condition fields',
    'all_unknown':                'All condition fields unknown',
    'any_unknown':                'Any condition field unknown',
    'old_and_unknown':            'Old vehicle with unknown conditions',

    # Engine specs (parsed from engine_name)
    'displacementl':              'Engine displacement (liters)',
    'enginecylinders':            'Number of engine cylinders',
    'engineconfiguration':        'Engine layout (V/In-line/Boxer/W/Rotary)',
    'engineconfiguration_freq':   'Engine layout (frequency)',
    'enginehp':                   'Engine horsepower',
    'enginehp_bucket':            'Engine horsepower (bucketed)',
    'valvetraindesign':           'Valvetrain design (DOHC/SOHC/OHV)',
    'valvetraindesign_freq':      'Valvetrain design (frequency)',
    'displacementl_x_age':        'Engine displacement × age',
    'enginecylinders_x_make':     'Cylinders × make',
    'enginecylinders_x_make_freq':'Cylinders × make (frequency)',
    'enginehp_bkt_x_age_bkt':     'HP bucket × age bucket',
    'enginehp_bkt_x_age_bkt_freq':'HP bucket × age bucket (frequency)',
    'engineconfig_x_make':        'Engine layout × make',
    'engineconfig_x_make_freq':   'Engine layout × make (frequency)',

    # Temporal (from record_creation_date)
    'dow':                        'Day of week (0=Mon ... 6=Sun)',
    'month':                      'Month of sale (1-12)',
    'quarter':                    'Quarter of sale (1-4)',
    'day_of_month':               'Day of month (1-31)',
    'day_of_year':                'Day of year (1-366)',
    'dow_sin':                    'Day of week (cyclical sin)',
    'dow_cos':                    'Day of week (cyclical cos)',
    'month_sin':                  'Month of sale (cyclical sin)',
    'month_cos':                  'Month of sale (cyclical cos)',
    'day_of_month_sin':           'Day of month (cyclical sin)',
    'day_of_month_cos':           'Day of month (cyclical cos)',
    'day_of_year_sin':            'Day of year (cyclical sin)',
    'day_of_year_cos':            'Day of year (cyclical cos)',

    # Interactions (user-requested)
    'make_x_age':                 'Make × age bucket interaction',
    'month_x_age':                'Month × age bucket interaction',
    'mileage_x_age':              'Mileage × age interaction',
    'year_x_dow':                 'Year × day-of-week interaction',
    'zip_region_x_mileage_bkt':   'ZIP region × mileage bucket',
    'quarter_x_make':             'Quarter × make interaction',

    # Vehicle-profile cluster (target-free k-means on vehicle attributes)
    'vehicle_profile_cluster_id':   'Vehicle profile cluster (1 of 12 groups)',
    'cluster_x_age_bkt':            'Cluster × age bucket',
    'cluster_x_age_bkt_freq':       'Cluster × age bucket (frequency)',
    'cluster_x_mileage_bkt':        'Cluster × mileage bucket',
    'cluster_x_mileage_bkt_freq':   'Cluster × mileage bucket (frequency)',

    # Geo
    'vazipcode':                  'ZIP code',
    'zip_region':                 'ZIP region (first digit)',
    'zip_first2':                 'ZIP first 2 digits (regional)',
    'zip_first3':                 'ZIP first 3 digits (state-area)',
    'zip_first2_freq':            'ZIP regional (frequency of 2-digit prefix)',
    'zip_first3_freq':            'ZIP state-area (frequency of 3-digit prefix)',
    'zip_full_freq':              'ZIP full (frequency of 5-digit ZIP)',
    'zip_lat':                    'Latitude',
    'zip_lon':                    'Longitude',
    'state_province_of_title':    'State of title',

    # Macro
    'cpi_at_sale':                'Consumer Price Index at sale',
    'manheim_at_sale':            'Manheim used-vehicle index at sale',
    'loan_at_sale':               'Auto loan rate at sale',

    # Cult
    'is_cult':                    'Cult/collectible vehicle',
    'cult_tier_num':              'Cult tier (A/B/C)',
    'cult_enthusiast_score':      'Cult enthusiast score',
    'cult_uplift_low':            'Cult price uplift (low)',
    'cult_uplift_mid':            'Cult price uplift (midpoint)',
    'cult_uplift_high':           'Cult price uplift (high)',
    'cult_uplift_range':          'Cult price uplift range',
    'cult_last_of_kind':          'Cult: last-of-kind model',
    'cult_liquidity_num':         'Cult market liquidity',
    'cult_volatility_num':        'Cult price volatility',
    'cult_origsens_num':          'Cult originality sensitivity',
    'cult_match_strictness':      'Cult match confidence',

    # Interactions
    'cult_x_age':                 'Cult interaction with age',
    'cult_x_mileage_bkt':         'Cult interaction with mileage',
    'culttier_x_age':             'Cult tier interaction with age',
    'origsens_x_runs':            'Originality interaction with runs',

    # Other damages (parsed from other_damages JSON/text field)
    'has_other_damage':           'Reported damage flagged',
    'n_other_damages':            'Number of damage types reported',
    'other_damages_normalized':   'Damage types (normalized)',
    'other_damages_normalized_freq':'Damage signature frequency',
    'has_mold':                   'Has mold damage',
    'has_undercarriage_rust':     'Has severe undercarriage rust',
    'has_smog_fail':              "Won't pass smog/state inspection",
    'damage_x_age':               'Damage × age interaction',
    'damage_x_mileage_bkt':       'Damage × mileage bucket',
    'damage_x_mileage_bkt_freq':  'Damage × mileage bucket (frequency)',

    # New raw categoricals (NEW_FEATURE_COLS; gated by --enable-new-features)
    'true_mileage_unknown':       'Odometer reading may be inaccurate',
    'clean_title':                'Clean title',
    'gvm_range':                  'Gross vehicle weight (GVM) range',
    'tonnage':                    'Tonnage/weight class',
    'engine_type':                'Engine type',

    # Mileage-trust interactions
    'mileage_unknown_x_age':            'Odometer trust × age',
    'mileage_unknown_x_mileage_bucket': 'Odometer trust × mileage bucket',
    'mileage_unknown_x_make':           'Odometer trust × make',
    'mileage_unknown_x_n_unknowns':     'Odometer trust × unknown-condition count',

    # Clean-title interactions
    'clean_title_x_age':            'Clean title × age',
    'clean_title_x_mileage_bucket': 'Clean title × mileage bucket',

    # Worst-dollar-error-tier interactions (WORST_TIER_FEATURE_COLS)
    'unknowns_x_mileage_bkt':       'Unknown-condition count × mileage bucket',
    'unknowns_x_age_bkt':           'Unknown-condition count × age bucket',
    'mech_severity_x_mileage_bkt':  'Mechanical severity × mileage bucket',
    'cult_x_n_unknowns':            'Cult status × unknown-condition count',
    'vtype_x_mileage_bkt':          'Vehicle type × mileage bucket',

    # Condition-x-make interactions (CONDITION_MAKE_FEATURE_COLS)
    'runs_x_make':                  'Driveability × make',
    'mech_severity_x_make':         'Mechanical severity × make',
    'all_cond_combo_x_make':        'Overall condition combo × make',

    # Original pass-through columns missing a label (always present)
    'Specialty Item':               'Flagged as a specialty item (RV/boat/heavy equipment)',
    'all_clean_notes':              'Donor/pickup notes',
    # Plain "State" on purpose: this is the vehicle's own state
    # (state_picklist_id), distinct from state_province_of_title's "State
    # of title" (state_title_picklist). The old label mentioned title and
    # registration, which read as the same thing as that other group --
    # they coincide on most vehicles but are genuinely separate inputs.
    'vstate_name':                  'State',
    'accessiblefortwotruck':        'Accessible for tow truck',
    'locatedatdonationca':          'Located at donation center',
}

# Suffix-based fallbacks
SUFFIX_HINTS = {
    '_freq':    'frequency in training data',
    '_tgt_enc': 'historical price for this group',
}


def humanize_feature(name: str) -> str:
    if name in HUMAN_READABLE:
        return HUMAN_READABLE[name]

    base = name
    suffix_label = None
    for suf, label in SUFFIX_HINTS.items():
        if name.endswith(suf):
            base = name[: -len(suf)]
            suffix_label = label
            break

    base_label = HUMAN_READABLE.get(base, base.replace('_', ' ').strip())

    if suffix_label:
        return f"{base_label} — {suffix_label}"
    return base_label
