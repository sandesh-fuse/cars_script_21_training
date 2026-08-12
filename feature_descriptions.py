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
