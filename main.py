import re
import requests
from datetime import datetime, timedelta
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# # Load predictions
# predictions = {
#     "2025-04-30": 58908.17,
#     "2025-05-01": 59049.48,
#     "2025-05-02": 59124.85,
#     "2025-05-03": 59197.16,
#     "2025-05-04": 59266.38,
#     "2025-05-05": 59332.35,
#     "2025-05-06": 59394.97,
#     "2025-05-07": 59454.09
# }
# df = pd.DataFrame(list(predictions.items()), columns=["date", "price"])
# df["date"] = pd.to_datetime(df["date"])
# df.set_index("date", inplace=True)

base_url = 'http://127.0.0.1:5000'

# Ambil prediksi dari endpoint Flask
def fetch_predictions_from_extracted_info(extracted: dict):
    commodity = extracted.get("PRD")
    num_days = extracted.get("days")

    if commodity is None or num_days is None:
        print("Gagal: Komoditas atau jumlah hari tidak valid.")
        return pd.DataFrame(columns=["date", "price"])

    try:
        url = f"{base_url}/lstm/predict?comodity={commodity}&num_days={num_days}"
        res = requests.get(url)
        data = res.json()["predictions"]
        df = pd.DataFrame(list(data.items()), columns=["date", "price"])
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)
        return df
    except Exception as e:
        print("Gagal mengambil data prediksi:", e)
        return pd.DataFrame(columns=["date", "price"])
    
# Load model dan tokenizer
# model_name = "cahya/bert-base-indonesian-NER"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModelForTokenClassification.from_pretrained(model_name)

# # Inisialisasi pipeline NER
# ner_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# def extract_entities(text):
#     ner_results = ner_pipeline(text)
#     entities = {}
#     for entity in ner_results:
#         label = entity['entity_group']
#         word = entity['word'].lower()
#         if label in entities:
#             entities[label].append(word)
#         else:
#             entities[label] = [word]
#     return entities

def extract_info(text: str) -> dict:
    result = {"PRD": None, "QTY": None, "days": None}
    text = text.lower()

    # === 1. Produk ===
    if "cabai" in text:
        result["PRD"] = "cabai"
    elif "bawang merah" in text:
        result["PRD"] = "bawang_merah"
    elif "bawang putih" in text:
        result["PRD"] = "bawang_putih"
    elif "bawang" in text and "putih" in text:
        result["PRD"] = "bawang_putih"
    elif "bawang" in text and "merah" in text:
        result["PRD"] = "bawang_merah"

    # === 2. Kuantitas (durasi) ===
    qty_patterns = [
        r"\d+\s*hari", r"\d+\s*minggu", r"\d+\s*bulan", r"\d+\s*tahun",
        r"(sehari|seminggu|sebulan|setahun)",
        r"(satu|dua|tiga|empat|lima|enam|tujuh|delapan|sembilan|sepuluh)\s*(hari|minggu|bulan|tahun)"
    ]
    
    matches = []
    for pattern in qty_patterns:
        matches += re.findall(pattern, text)

    durations = []
    for m in matches:
        qty = m if isinstance(m, str) else " ".join(m)
        days = convert_to_days(qty)
        if days:
            durations.append((qty, days))

    if durations:
        # Ambil durasi dengan nilai hari terbesar
        max_duration = max(durations, key=lambda x: x[1])
        result["QTY"] = max_duration[0]
        result["days"] = max_duration[1]

    return result


def convert_to_days(duration: str) -> int | None:
    duration = duration.lower()
    mapping = {
        "hari": 1, "minggu": 7, "bulan": 30, "tahun": 365,
        "sehari": 1, "seminggu": 7, "sebulan": 30, "setahun": 365
    }

    if duration in mapping:
        return mapping[duration]

    for key in mapping:
        if key in duration:
            num = extract_number(duration)
            return num * mapping[key]
    return None

def extract_number(text: str) -> int:
    number_map = {
        "satu": 1, "dua": 2, "tiga": 3, "empat": 4,
        "lima": 5, "enam": 6, "tujuh": 7, "delapan": 8,
        "sembilan": 9, "sepuluh": 10
    }
    for word, num in number_map.items():
        if word in text:
            return num
    match = re.search(r"\d+", text)
    return int(match.group()) if match else 1

def build_context(df, commodity):
    commodity_map = {
        "cabai": "cabai",
        "bawang_merah": "bawang merah",
        "bawang_putih": "bawang putih"
    }
    commodity_name = commodity_map.get(commodity, commodity)  # 
    today = datetime.today().strftime('%d %B %Y')
    context = f"Hari ini adalah {today}.\nBerikut data prediksi harga {commodity_name}:\n"
    for date, row in df.iterrows():
        context += f"{date.strftime('%d %B %Y')}: Rp{int(row['price']):,}\n"
    return context


def ask_llama(question, context, commodity):
    system_prompt = f"Gunakan konteks untuk menjawab pertanyaan tentang harga {commodity} dari hari ini."
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.1",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}"},
                {"role": "user", "content": question}
            ],
            "stream": False
        }
    )

    if response.status_code == 200:
        return response.json()["message"]["content"]
    else:
        return f"Error: {response.status_code} - {response.text}"


# MAIN

test_inputs = [
    "berapa harga cabai untuk 3 hari ke depan",
    "tolong tampilkan harga cabai selama satu bulan",
    "aku pengen tau harga cabai buat 2 tahun ke depan",
    "berapa sih harga cabai kalau dihitung 30 hari dari sekarang?",
    "bisa kasih tau harga bawang putih dalam 15 hari ke depan?",
    "harga cabai kalau saya pantau selama 10 hari gimana?",
    "prediksi harga tomat selama seminggu ke depan dong",
    "berapa harga cabe rawit dalam rentang 2 minggu?",
    "saya ingin melihat harga bawang merah selama satu tahun penuh",
    "berapa sih perkiraan harga sayur kol dalam 20 hari mendatang?",
    "coba lihat harga cabe ijo selama 5 hari ya",
    "berapa kira-kira harga cabai selama tiga bulan?",
    "perlu info harga bawang merah untuk 14 hari ke depan",
    "apa harga cabai akan naik selama 40 hari mendatang?",
    "kasih saya data harga bawang putih untuk sebulan ke depan dong"
]

# for input_text in test_inputs:
#     comodity, num_days = extract_comodity_and_days(input_text)
#     print(f"Input: {input_text} => Commodity: {comodity}, Num_days: {num_days}")

print("-----------------------------------------------")

# for input_text in test_inputs:
#     entities = extract_info(input_text)
#     print(f"Input: {input_text} => {entities} ")

input = "tampilkan data untuk komoditas cabai seminggu dan sebulan kedepan?"

def run_prediction_pipeline(user_input):
    extracted = extract_info(user_input)
    print(f"Ekstraksi: {extracted}")

    df = fetch_predictions_from_extracted_info(extracted)
    if not df.empty:
        print("\n📈 Hasil Prediksi Harga:")
        print(df)
    else:
        print("❌ Tidak ada data yang bisa ditampilkan.")

# run_prediction_pipeline(input("enter query : "))
# run_prediction_pipeline(input)

def chatbot_test(user_input): 
    extracted = extract_info(user_input)
    print(f"Ekstraksi: {extracted}")
        
    df = fetch_predictions_from_extracted_info(extracted)
    if df.empty:
            print("Tidak ada data prediksi tersedia.")
    else:
        context = build_context(df, extracted["PRD"])
        # Tanyakan ke Llama
        print(user_input)
        print("🟢 Jawaban untuk pertanyaan pengguna:")
        print(ask_llama(user_input, context, extracted["PRD"]))

# chatbot_test(input("enter query : "))
chatbot_test(input)