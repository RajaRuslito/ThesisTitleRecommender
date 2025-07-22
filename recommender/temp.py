import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import random

class TitleGenerator:
    def __init__(self, model_name="cahya/gpt2-small-indonesian-522M"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        # Pad token harus diset untuk model GPT2
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Kata-kata tabu
        self.bad_words = ['http', 'youtube', 'skripsi ini', 'contoh karya', 'universitas']
        self.bad_words_ids = [self.tokenizer.encode(w, add_special_tokens=False) for w in self.bad_words]

    def clean_generated_text(self, text):
        # Hentikan kalau muncul bagian non-judul
        stop_phrases = ["bab", "daftar isi", "kesimpulan", "pendahuluan"]
        for phrase in stop_phrases:
            match = re.search(rf"{phrase}", text, re.IGNORECASE)
            if match:
                text = text[:match.start()]
        # Ambil satu kalimat pertama
        cleaned = text.strip().split("\n")[0].strip()
        return cleaned

    def is_invalid_output(self, text):
        lower = text.lower()
        if not text or len(text) < 10:
            return True
        if any(bad in lower for bad in self.bad_words):
            return True
        if re.search(r'https?://', text):
            return True
        return False

    def fallback_title(self, input_topic):
        opsi_awal = ["Rancang Bangun", "Analisis", "Pengembangan", "Perancangan Sistem"]
        teknik = ["Machine Learning", "Deep Learning", "SVM", "KNN", "Decision Tree"]
        return f"{random.choice(opsi_awal)} Sistem {input_topic.title()} Menggunakan {random.choice(teknik)}"

    def generate_title(self, context_titles, input_topic=None, max_new_tokens=30, retry_limit=3):
        # Format ulang prompt jadi lebih "guided"
        context = "\n".join(f"- {title}" for title in context_titles)
        input_topic = input_topic or "teknologi"

        prompt = (
            f"Topik: {input_topic.title()}\n\n"
            f"Berikut adalah beberapa contoh judul skripsi:\n"
            f"{context}\n\n"
            f"Tugas Anda adalah membuat satu judul skripsi baru yang:\n"
            f"- Relevan dengan daftar judul di atas\n"
            f"- Menggunakan Bahasa Indonesia akademik\n"
            f"- Hanya terdiri dari satu kalimat\n"
            f"- Tidak menyebutkan penulis, kampus, tautan, atau metadata lainnya\n\n"
            f"Judul:"
        )

        for _ in range(retry_limit):
            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
            output = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_k=40,
                temperature=0.85,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                bad_words_ids=self.bad_words_ids
            )

            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            raw_result = decoded.split("Judul:")[-1].strip()
            cleaned = self.clean_generated_text(raw_result)

            if not self.is_invalid_output(cleaned):
                return cleaned

        return self.fallback_title(input_topic)


        #     decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        #     raw_result = decoded.split("Judul:")[-1].strip()
        #     cleaned = self.clean_generated_text(raw_result)

        #     if not self.is_invalid_output(cleaned):
        #         return cleaned

        # # Fallback kalau 3x gagal
        # return self.fallback_title(input_topic or "Teknologi")
