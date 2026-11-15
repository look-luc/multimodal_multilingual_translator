'''
AUDIO TO TEXT DATASETS
'''
WAXAL_LANG_CODE = {
    "Acholi": "ach",
    "Luganda": "lug",
    "Masaaba": "mas",
    "Nyankole": "nyn",
    "Soga": "sog",
    "Akan": "aka",
    "Dagbani": "dag",
    "Dagaare": "dga",
    "Ikposo": "kpo",
    "Fula": "ful",
    "Lingala": "lin",
    "Shona": "sna",
    "Malagasy": "mlg",
    "Amharic": "amh",
    "Oromo": "orm",
    "Sidama": "sid",
    "Tigrinya": "tir",
    "Wolaytta": "wal",
}
WAXAL_LANG_CODE_TTS ={
    "Acholi": "ach",
    "Luganda": "lug",
    "Kiswahili": "swa",
    "Nyankole": "nyn",
    "Akan": "aka",
    "Baoule": "bau",
    "Ewe": "ewe",
    "Fula": "ful",
    "Igbo": "ibo",
    "Hausa": "hau",
    "Yoruba": "yor",
    "Nigerian Pidgin": "pcm",
    "Kikuyu": "kik",
    "Luo": "luo",
    "Swahili": "swa",
    "Bambara": "bam",
    "Pular": "fat",
    "Wolof": "wol"
}

FLEURS_LANG_CODES = {
    "Afrikaans": "af_za",
    "Kamba": "kam_ke",
    "Northern Sotho": "nso_za",
    "Nyanja": "ny_mw",
    "Somali": "so_so",
    "Umbundu": "umb_ao",
    "Xhosa": "xh_za",
    "Zulu": "zu_za",
    "Asturian": "ast_es",
    "Bosnian": "bs_ba",
    "Catalan": "ca_es",
    "Croatian": "hr_hr",
    "Danish": "da_dk",
    "Dutch": "nl_nl",
    "Finnish": "fi_fi",
    "Galician": "gl_es",
    "Greek": "el_gr",
    "Hungarian": "hu_hu",
    "Icelandic": "is_is",
    "Irish": "ga_ie",
    "Kabuverdianu": "kea_cv",
    "Luxembourgish": "lb_lu",
    "Maltese": "mt_mt",
    "Norwegian Bokmål": "nb_no",
    "Occitan": "oc_fr",
    "Portuguese": "pt_br",
    "Swedish": "sv_se",
    "Welsh": "cy_gb",
    "Armenian": "hy_am",
    "Belarusian": "be_by",
    "Bulgarian": "bg_bg",
    "Czech": "cs_cz",
    "Estonian": "et_ee",
    "Georgian": "ka_ge",
    "Latvian": "lv_lv",
    "Lithuanian": "lt_lt",
    "Macedonian": "mk_mk",
    "Polish": "pl_pl",
    "Romanian": "ro_ro",
    "Russian": "ru_ru",
    "Serbian": "sr_rs",
    "Slovak": "sk_sk",
    "Slovenian": "sl_si",
    "Ukrainian": "uk_ua",
    "Azerbaijani": "az_az",
    "Kazakh": "kk_kz",
    "Kyrgyz": "ky_kg",
    "Mongolian": "mn_mn",
    "Pashto": "ps_af",
    "Persian": "fa_ir",
    "Sorani Kurdish": "ckb_iq",
    "Tajik": "tg_tj",
    "Turkish": "tr_tr",
    "Uzbek": "uz_uz",
    "Assamese": "as_in",
    "Bengali": "bn_in",
    "Gujarati": "gu_in",
    "Hindi": "hi_in",
    "Kannada": "kn_in",
    "Malayalam": "ml_in",
    "Marathi": "mr_in",
    "Nepali": "ne_np",
    "Odia": "or_in",
    "Punjabi": "pa_in",
    "Sindhi": "sd_in",
    "Tamil": "ta_in",
    "Telugu": "te_in",
    "Urdu": "ur_pk",
    "Burmese": "my_mm",
    "Cebuano": "ceb_ph",
    "Filipino": "fil_ph",
    "Indonesian": "id_id",
    "Javanese": "jv_id",
    "Khmer": "km_kh",
    "Lao": "lo_la",
    "Malay": "ms_my",
    "Māori": "mi_nz",
    "Thai": "th_th",
    "Vietnamese": "vi_vn",
    "Cantonese": "yue_hant_hk",
    "Korean": "ko_kr"
}

IVANGTORRE_LANG_CODE = {
    "Quechua": "qu",
    "Aymara": "ay",
    "Guaraní": "gn",
    "Nahuatl": "nah",
}

QCRI_QASR_LANGUAGES = {
    "Arabic": "ar",   # Primary language: Covers Modern Standard Arabic (MSA) and regional dialects
    "English": "en"   # Secondary language: Present in broadcast code-switching and foreign terms
}

CASABLANCA_CONFIGS = {
    "Algerian Arabic": "Algeria",
    "Egyptian Arabic": "Egypt",
    "Jordanian Arabic": "Jordan",
    "Mauritanian Arabic": "Mauritania",
    "Moroccan Arabic": "Morocco",
    "Palestinian Arabic": "Palestine",
    "Emirati Arabic": "UAE",
    "Yemeni Arabic": "Yemen"
}

'''
TEXT TO TEXT DATASETS
'''
AFRIQA_LANG_CODE = {
    "Bemba": "bem",
    "Fon": "fon",
    "Hausa": "hau",
    "Igbo": "ibo",
    "Kinyarwanda": "kin",
    "Swahili": "swą",
    "Twi": "twi",
    "Wolof": "wol",
    "Yorùbá": "yor",
    "Zulu": "zul",
}

AMERICAS_LANG_CODE = {
    "Bribri": "bribri",
    "Guaraní": "guarani",
    "Kotiria": "kotiria",
    "Wa'ikhana": "waikhana",
    "Quechua": "quechua"
}

SEACROWD_PARALLEL_ASIAN_TREEBANK_LANGS = {
    "Bengali": "ben",
    "Filipino": "fil",
    "Hindi": "hin",
    "Indonesian": "ind",
    "Khmer": "khm",
    "Lao": "lao",
    "Malay": "zlm",       # Or msa
    "Myanmar (Burmese)": "mya",
    "Thai": "tha",
    "Vietnamese": "vie"
}

FLORES_PLUS_CODES = {
    # --- Major Global & Regional Languages ---
    "Korean (Hangul)": "kor_Hang",
    "Russian (Cyrillic)": "rus_Cyrl",
    "Hindi (Devanagari)": "hin_Deva",

    # --- Germanic Languages ---
    "Dutch": "nld_Latn",
    "Danish": "dan_Latn",
    "Swedish": "swe_Latn",
    "Norwegian Bokmål": "nob_Latn",
    "Norwegian Nynorsk": "nno_Latn",
    "Icelandic": "isl_Latn",
    "Faroese": "fao_Latn",
    "Luxembourgish": "ltz_Latn",
    "Limburgish": "lim_Latn",

    # --- Romance Languages ---
    "Portuguese": "por_Latn",
    "Romanian": "ron_Latn",
    "Catalan": "cat_Latn",
    "Galician": "glg_Latn",
    "Occitan": "oci_Latn",
    "Asturian": "ast_Latn",
    "Aragonese": "arg_Latn",
    "Sardinian": "srd_Latn",
    "Sicilian": "scn_Latn",
    "Friulian": "fur_Latn",
    "Ligurian": "lij_Latn",
    "Lombard": "lmo_Latn",

    # --- Slavic Languages ---
    "Russian": "rus_Cyrl",
    "Ukrainian": "ukr_Cyrl",
    "Polish": "pol_Latn",
    "Czech": "ces_Latn",
    "Slovak": "slk_Latn",
    "Bulgarian": "bul_Cyrl",
    "Serbian": "srp_Cyrl",
    "Croatian": "hrv_Latn",
    "Bosnian": "bos_Latn",
    "Slovenian": "slv_Latn",
    "Macedonian": "mkd_Cyrl",
    "Belarusian": "bel_Cyrl",
    "Silesian": "szl_Latn",

    # --- Finno-Ugric & Uralic Languages ---
    "Finnish": "fin_Latn",
    "Estonian": "est_Latn",
    "Hungarian": "hun_Latn",

    # --- Celtic Languages ---
    "Welsh": "cym_Latn",
    "Irish": "gle_Latn",
    "Scottish Gaelic": "gla_Latn",

    # --- Baltic Languages ---
    "Lithuanian": "lit_Latn",
    "Latvian": "lvs_Latn",
    "Latgalian": "ltg_Latn",

    # --- Other European Language Families & Isolates ---
    "Greek": "ell_Grek",
    "Albanian (Tosk)": "als_Latn",
    "Basque": "eus_Latn",
    "Maltese": "mlt_Latn",
    "Armenian": "hye_Armn",
    "Turkish": "tur_Latn",
    "Crimean Tatar": "crh_Latn",

    # --- Arabic Dialects & Regional Varieties ---
    "Mesopotamian Arabic": "acm_Arab",
    "Ta’izzi-Adeni Arabic": "acq_Arab",
    "Tunisian Arabic": "aeb_Arab",
    "Levantine Arabic (North)": "apc_Arab",
    "Sudanese Arabic": "apd_Arab",
    "Najdi Arabic": "ars_Arab",
    "Moroccan Arabic": "ary_Arab",
    "Egyptian Arabic": "arz_Arab",

    # --- Acehnese Scripts ---
    "Acehnese (Jawi script)": "ace_Arab",
    "Acehnese (Latin script)": "ace_Latn",

    # --- African & Indigenous Varieties ---
    "Afrikaans": "afr_Latn",
    "Amharic": "amh_Ethi",
    "Bambara": "bam_Latn",
    "West Central Oromo": "gaz_Latn",
    "Nigerian Fulfulde": "fuv_Latn",
}

'''
Sign languages
'''
HOW2SIGN = {
    "American Sign Language (ASL)": "ase"
}

SIGNVERSE_LANG_CODES = {
    "Argentine Sign Language": "aed",
    "American Sign Language": "ase",
    "Australian Sign Language": "asf",
    "British Sign Language": "bfi",
    "Brazilian Sign Language": "bzs",
    "Catalan Sign Language": "csc",
    "Czech Sign Language": "cse",
    "Chilean Sign Language": "csg",
    "Colombian Sign Language": "csn",
    "Dutch Sign Language": "dse",
    "Danish Sign Language": "dsl",
    "Estonian Sign Language": "eso",
    "Quebec Sign Language": "fcs",
    "Finnish Sign Language": "fse",
    "French Sign Language": "fsl",
    "Finland-Swedish Sign Language": "fss",
    "German Sign Language": "gsg",
    "Greek Sign Language": "gss",
    "Hong Kong Sign Language": "hks",
    "Hungarian Sign Language": "hsh",
    "Icelandic Sign Language": "icl",
    "International Sign": "ils",
    "Indonesian Sign Language": "inl",
    "Indian Sign Language": "ins",
    "Italian Sign Language": "ise",
    "Irish Sign Language": "isg",
    "Jordanian Sign Language": "jos",
    "Korean Sign Language": "kvk",
    "Slovenian Sign Language": "ysl"
}
