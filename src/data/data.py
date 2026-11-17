from datasets import load_dataset

from src.data.codes import (
    AFRIQA_LANG_CODE,
    AMERICAS_LANG_CODE,
    CASABLANCA_CONFIGS,
    FLEURS_LANG_CODES,
    FLORES_PLUS_CODES,
    HOW2SIGN,
    IVANGTORRE_LANG_CODE,
    QCRI_QASR_LANGUAGES,
    SEACROWD_PARALLEL_ASIAN_TREEBANK_LANGS,
    SIGNVERSE_LANG_CODES,
    WAXAL_LANG_CODE,
    WAXAL_LANG_CODE_TTS,
)


def load_all_datasets(streaming: bool = False):
    loaded_datasets = {}

    print("--- Loading Audio-to-Text & Speech Datasets ---")

    loaded_datasets["waxal_asr"] = {}
    for lang_name, code in WAXAL_LANG_CODE.items():
        try:
            print(f"Loading WAXAL ASR for {lang_name} ({code})...")
            loaded_datasets["waxal_asr"][lang_name] = load_dataset(
                "google/WaxalNLP", code, streaming=streaming
            )
        except Exception as e:
            print(f"Skipping WAXAL ASR {lang_name}: {e}")

    loaded_datasets["waxal_tts"] = {}
    for lang_name, code in WAXAL_LANG_CODE_TTS.items():
        try:
            print(f"Loading WAXAL TTS for {lang_name} ({code})...")
            loaded_datasets["waxal_tts"][lang_name] = load_dataset(
                "google/WaxalNLP", f"tts_{code}", streaming=streaming
            )
        except Exception as e:
            print(f"Skipping WAXAL TTS {lang_name}: {e}")

    loaded_datasets["fleurs"] = {}
    for lang_name, code in FLEURS_LANG_CODES.items():
        try:
            print(f"Loading FLEURS for {lang_name} ({code})...")
            loaded_datasets["fleurs"][lang_name] = load_dataset(
                "google/fleurs", code, streaming=streaming
            )
        except Exception as e:
            print(f"Skipping FLEURS {lang_name}: {e}")

    loaded_datasets["ivangtorre_americas"] = {}
    for lang_name, code in IVANGTORRE_LANG_CODE.items():
        try:
            print(f"Loading Americas NLP / Mapudungun for {lang_name} ({code})...")
            loaded_datasets["ivangtorre_americas"][lang_name] = load_dataset(
                "ivangtorre/second_americas_nlp_2022", code, streaming=streaming
            )
        except Exception as e:
            print(f"Skipping Americas NLP {lang_name}: {e}")

    loaded_datasets["qcri_qasr"] = {}
    for lang_name, code in QCRI_QASR_LANGUAGES.items():
        try:
            print(f"Loading QCRI QASR for {lang_name} ({code})...")
            loaded_datasets["qcri_qasr"][lang_name] = load_dataset(
                "QCRI/QASR", code, streaming=streaming
            )
        except Exception as e:
            print(f"Skipping QCRI QASR {lang_name}: {e}")

    loaded_datasets["casablanca"] = {}
    for dialect_name, config in CASABLANCA_CONFIGS.items():
        try:
            print(f"Loading Casablanca ASR for {dialect_name} ({config})...")
            loaded_datasets["casablanca"][dialect_name] = load_dataset(
                "Casablanca", config, streaming=streaming
            )
        except Exception as e:
            print(f"Skipping Casablanca {dialect_name}: {e}")

    print("\n--- Loading Text-to-Text Datasets ---")

    loaded_datasets["afriqa"] = {}
    for lang_name, code in AFRIQA_LANG_CODE.items():
        try:
            print(f"Loading AfriQA for {lang_name} ({code})...")
            loaded_datasets["afriqa"][lang_name] = load_dataset(
                "African-Languages-Lab/multi-open", code, streaming=streaming
            )
        except Exception as e:
            print(f"Skipping AfriQA {lang_name}: {e}")

    loaded_datasets["americas_text"] = {}
    for lang_name, code in AMERICAS_LANG_CODE.items():
        try:
            print(f"Loading Americas text corpus for {lang_name} ({code})...")
            loaded_datasets["americas_text"][lang_name] = load_dataset(
                "ivangtorre/second_americas_nlp_2022", code, streaming=streaming
            )
        except Exception as e:
            print(f"Skipping Americas text {lang_name}: {e}")

    loaded_datasets["seacrowd_asian"] = {}
    for lang_name, code in SEACROWD_PARALLEL_ASIAN_TREEBANK_LANGS.items():
        try:
            print(f"Loading SEACrowd Parallel Asian Treebank for {lang_name} ({code})...")
            loaded_datasets["seacrowd_asian"][lang_name] = load_dataset(
                "SEACrowd/parallel_asian_treebank", code, streaming=streaming
            )
        except Exception as e:
            print(f"Skipping SEACrowd Asian {lang_name}: {e}")

    loaded_datasets["flores_plus"] = {}
    for lang_name, code in FLORES_PLUS_CODES.items():
        try:
            print(f"Loading FLORES-Plus for {lang_name} ({code})...")
            loaded_datasets["flores_plus"][lang_name] = load_dataset(
                "facebook/flores", code, streaming=streaming
            )
        except Exception as e:
            print(f"Skipping FLORES-Plus {lang_name}: {e}")

    print("\n--- Loading Sign Language Datasets ---")

    loaded_datasets["how2sign"] = {}
    for lang_name, code in HOW2SIGN.items():
        try:
            print(f"Loading How2Sign for {lang_name} ({code})...")
            loaded_datasets["how2sign"][lang_name] = load_dataset(
                "How2Sign", code, streaming=streaming
            )
        except Exception as e:
            print(f"Skipping How2Sign {lang_name}: {e}")

    loaded_datasets["signverse"] = {}
    for lang_name, code in SIGNVERSE_LANG_CODES.items():
        try:
            print(f"Loading SignVerse dataset config for {lang_name} ({code})...")
            loaded_datasets["signverse"][lang_name] = load_dataset(
                "SignVerse", code, streaming=streaming
            )
        except Exception as e:
            print(f"Skipping SignVerse {lang_name}: {e}")

    return loaded_datasets
