from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import random

class TitleGenerator:
    def __init__(self, model_name="cahya/t5-base-indonesian-summarization-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def clean_generated_text(self, text):
    # Coba deteksi dan bersihin token rusak
        text = re.sub(r"[^a-zA-Z0-9\s\-.,()]", "", text)
        
        stop_phrases = ["bab", "daftar isi", "kesimpulan", "pendahuluan"]
        for phrase in stop_phrases:
            match = re.search(rf"{phrase}", text, re.IGNORECASE)
            if match:
                text = text[:match.start()]
        
        return text.strip().split("\n")[0].strip()


    def is_invalid_output(self, text):
        lower = text.lower()
        return not text or len(text) < 10 or any(x in lower for x in ['http', 'skripsi ini', 'universitas', 'contoh'])

    def fallback_title(self, input_topic):
        opsi_awal = ["Rancang Bangun", "Analisis", "Pengembangan", "Perancangan Sistem"]
        teknik = ["Machine Learning", "Deep Learning", "SVM", "KNN", "Decision Tree"]
        return f"{random.choice(opsi_awal)} Sistem {input_topic.title()} Menggunakan {random.choice(teknik)}"

    # def generate_title(self, context_titles, input_topic=None, max_new_tokens=30, retry_limit=3):
    #     context = "\n".join(f"- {title}" for title in context_titles)

    #     # Gunakan format T5-friendly (instruksi + konteks + jawaban)
    #     prompt = (
    #         f"Berikut adalah beberapa contoh judul skripsi:\n"
    #         f"{context}\n\n"
    #         f"Tugas Anda adalah membuat satu judul skripsi baru yang:\n"
    #         f"- Relevan dengan daftar judul di atas\n"
    #         f"- Menggunakan Bahasa Indonesia akademik\n"
    #         f"- Hanya terdiri dari satu kalimat\n"
    #         f"- Tidak menyebutkan penulis, kampus, tautan, atau metadata lainnya\n\n"
    #         f"Judul:"
    #     )


    #     for attempt in range(retry_limit):
    #         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512, padding=True)
    #         outputs = self.model.generate(
    #             input_ids=inputs["input_ids"],
    #             attention_mask=inputs["attention_mask"],
    #             max_new_tokens=50,
    #             do_sample=True,
    #             top_p=0.92,
    #             temperature=0.9,
    #             num_return_sequences=1
    #         )

    #         decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    #         raw = decoded.split("Judul:")[-1].strip()
    #         cleaned = self.clean_generated_text(raw)

    #         if not self.is_invalid_output(cleaned):
    #             return cleaned

    #     return self.fallback_title(input_topic or "Teknologi")
    # def generate_title(self, titles: list[str]) -> str:
    #     """
    #     Menghasilkan judul skripsi baru menggunakan pendekatan generate-only.
    #     Hanya memberi pola judul sebelumnya, lalu model akan meneruskan daftar.
    #     """
    #     # Siapkan prompt berbasis pola list judul
    #     prompt = "Berikut adalah beberapa judul skripsi:\n"
    #     for title in titles:
    #         prompt += f"- {title}\n"
    #     prompt += "-"

    #     # Tokenisasi input
    #     inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True).to(self.model.device)

    #     # Generate tanpa instruksi, hanya meneruskan daftar
    #     outputs = self.model.generate(
    #         input_ids=inputs["input_ids"],
    #         attention_mask=inputs["attention_mask"],
    #         max_new_tokens=50,
    #         do_sample=True,
    #         top_p=0.9,
    #         temperature=0.85,
    #         num_return_sequences=1
    #     )

    #     # Decode hasil
    #     decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    #     # Ambil hanya hasil lanjutan setelah prompt terakhir (setelah "-")
    #     lines = decoded.strip().split("\n")
    #     for line in lines:
    #         if line.startswith("-"):
    #             candidate = line[1:].strip()
    #             if candidate and candidate not in titles:
    #                 return candidate

    #     # Jika gagal, fallback
    #     return "Judul baru tidak berhasil dihasilkan."

    def generate_title(self, titles: list[str]) -> str:
        # Gabungkan semua judul jadi 1 input string
        combined_titles = " | ".join(titles)

        # Tambahkan prefix task khusus T5
        prompt = f"buat_judul: {combined_titles}"

        # Generate dengan model
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=64,
            do_sample=True,         # sampling instead of greedy
            top_k=50,                # pick from top 50 candidates
            top_p=0.95,              # nucleus sampling
            temperature=0.9          # more creative outputs
        )


        # Decode hasilnya
        generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return generated.strip() if generated.strip() else "Judul baru tidak berhasil dihasilkan."



