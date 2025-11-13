# Mapping Informasi Penyakit untuk sistem CNN + LLM

disease_info = {
    "Grape___Black_rot": {
        "general_symptoms": [
            "Large brown spots with target-like concentric rings on leaves",
            "Berries turn black and shrivel with firm brown lesions",
            "Canes may show elongated brown lesions"
        ],
        "distinguishing_features": [
            "Concentric ring (target) pattern on leaves",
            "Mummified, hard black berries"
        ],
        "early_actions": [
            "Sanitation: remove infected tissues",
            "Apply preventive fungicide if needed"
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "general_symptoms": [
            "Leaves show chlorotic (yellow) spots with tiger-stripe banding",
            "Berries may shrivel or display sunburn-like symptoms",
            "Associated with fungal trunk disease (wood infection)"
        ],
        "distinguishing_features": [
            "Characteristic 'tiger-stripe' pattern on leaves",
            "Occurs more often in older vines"
        ],
        "early_actions": [
            "Manage trunk wounds to prevent infection",
            "Prune affected wood and improve vineyard sanitation"
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "general_symptoms": [
            "Irregular brown lesions on leaves, often starting at margins",
            "Lesions may coalesce, causing blighted areas",
            "Can result in premature leaf drop"
        ],
        "distinguishing_features": [
            "Lesions often lack concentric target rings",
            "Tends to affect leaf edges first"
        ],
        "early_actions": [
            "Monitor canopy and remove heavily infected leaves",
            "Ensure pruning and canopy management to reduce humidity"
        ]
    },
    "Grape___healthy": {
        "general_symptoms": [
            "Uniform green leaves without necrotic spots",
            "No fungal growth or pustules",
            "No chlorosis or blighted areas"
        ],
        "distinguishing_features": [
            "Absence of disease-specific lesions or discoloration"
        ],
        "early_actions": [
            "Keep monitoring and maintain good cultural practices",
            "Avoid excessive leaf wetness and maintain proper spacing"
        ]
    },
    "Apple___Apple_scab": {
        "general_symptoms": [
            "Olive-green to brown velvety lesions on leaves and fruit",
            "Leaves may become distorted or curled",
            "Early defoliation can occur under severe infection"
        ],
        "distinguishing_features": [
            "Velvety olive-brown spots that may turn darker and corky",
            "Commonly seen on young leaves and developing fruits"
        ],
        "early_actions": [
            "Remove or bury fallen leaves to reduce inoculum",
            "Use resistant cultivars and follow recommended fungicide schedules"
        ]
    },
    "Apple___Black_rot": {
        "general_symptoms": [
            "Leaf spots that start purple and turn brown with dark borders",
            "Fruit rot with concentric rings ('frog-eye' pattern)",
            "Cankers on twigs, branches, and trunks"
        ],
        "distinguishing_features": [
            "Distinct 'frog-eye' lesions on fruit",
            "Cankers with concentric ring pattern on bark"
        ],
        "early_actions": [
            "Prune out dead wood and cankers",
            "Destroy mummified fruits and fallen debris"
        ]
    },
    "Apple___Cedar_apple_rust": {
        "general_symptoms": [
            "Bright yellow-orange spots on upper leaf surfaces",
            "Lesions may develop concentric rings or 'halo'",
            "On the underside of leaves, orange, horn-like structures may form"
        ],
        "distinguishing_features": [
            "Association with juniper/cedar alternate hosts",
            "Gelatinous orange telial horns on juniper galls in spring"
        ],
        "early_actions": [
            "Remove nearby juniper hosts if practical",
            "Follow local guidelines for fungicide applications where necessary"
        ]
    },
    "Apple___healthy": {
        "general_symptoms": [
            "Uniformly green leaves without spots or lesions",
            "No visible signs of fungal or bacterial infection",
            "Normal leaf shape and texture"
        ],
        "distinguishing_features": [
            "Absence of scab, rust, or rot lesions"
        ],
        "early_actions": [
            "Maintain balanced fertilization and irrigation",
            "Monitor regularly for early symptom appearance"
        ]
    },
    "Potato___Early_blight": {
        "general_symptoms": [
            "Small dark spots on older leaves, often with concentric ring pattern",
            "Lesions expand and may cause leaf yellowing and defoliation",
            "Can also infect stems and tubers"
        ],
        "distinguishing_features": [
            "Target-like concentric rings in leaf lesions",
            "Primarily affects older foliage first"
        ],
        "early_actions": [
            "Remove and destroy infected plant debris",
            "Avoid overhead irrigation that prolongs leaf wetness"
        ]
    },
    "Potato___Late_blight": {
        "general_symptoms": [
            "Water-soaked, pale green lesions that rapidly turn dark brown to black",
            "Lesions often begin at leaf tips or edges and progress rapidly",
            "White fungal growth on the undersides of leaves in humid conditions",
            "Can infect stems and tubers; progresses very quickly"
        ],
        "distinguishing_features": [
            "Very rapid, water-soaked lesion expansion",
            "White mycelium along leaf undersides when humid"
        ],
        "early_actions": [
            "Remove infected tissue; follow local fungicide recommendations if needed",
            "Avoid prolonged leaf wetness and standing water"
        ]
    },
    "Potato___healthy": {
        "general_symptoms": [
            "Uniform green leaves without necrotic spots",
            "No fungal growth or pustules",
            "No concentric rings or elongated lesions"
        ],
        "distinguishing_features": [
            "Absence of all disease-specific cues"
        ],
        "early_actions": [
            "Continue good cultural practices",
            "Monitor regularly for early detection"
        ]
    }
}

