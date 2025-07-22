import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
from recommender.recommender import get_recommendations
from recommender.generator import TitleGenerator

def main():
    parser = argparse.ArgumentParser(description="Skripsi Title Recommender")
    parser.add_argument('--input', type=str, required=True, help='Topik skripsi yang diminati')
    parser.add_argument('--model', type=str, default='sbert', choices=['sbert', 'tfidf'], help='Model embedding yang digunakan')
    parser.add_argument('--top_k', type=int, default=5, help='Jumlah rekomendasi (Top-K)')
    parser.add_argument('--generate-new', action='store_true', help='Aktifkan mode generatif untuk membuat judul baru')
    args = parser.parse_args()

    print(f"Input: {args.input}")
    print(f"Menggunakan model: {args.model}")
    
    rekomendasi = get_recommendations(args.input, top_k=args.top_k, model=args.model)

    print(f"\nTop-{args.top_k} Rekomendasi Judul Skripsi:")
    for i, r in enumerate(rekomendasi, 1):
        print(f"{i}. {r}")

    if args.generate_new:
        print("\n[INFO] Membuat judul skripsi baru secara generatif berdasarkan hasil rekomendasi...")
        generator = TitleGenerator()
        generated_title = generator.generate_title(rekomendasi)
        print("\nJudul Skripsi Baru (Generated):")
        print(f"{generated_title}")
# , args.input
if __name__ == "__main__":
    main()
