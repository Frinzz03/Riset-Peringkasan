import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Pastikan path modul terbaca
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def set_academic_style():
    """Mengatur konfigurasi visual bergaya jurnal ilmiah IEEE / Elsevier."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight'
    })

def generate_all_charts(
    exp_csv_path: str = os.path.join(config.REPORTS_DIR, "hasil_eksperimen_skenario_e1_e4.csv"),
    litm_csv_path: str = os.path.join(config.REPORTS_DIR, "analisis_lost_in_the_middle.csv"),
    err_csv_path: str = os.path.join(config.REPORTS_DIR, "hasil_error_analysis.csv"),
    output_dir: str = os.path.join(config.REPORTS_DIR, "figures")
):
    os.makedirs(output_dir, exist_ok=True)
    set_academic_style()
    print(f"\n[Visualisasi] Menghasilkan grafik ilmiah beresolusi tinggi di: {output_dir}")

    # -------------------------------------------------------------
    # 1. Grafik Batang Perbandingan ROUGE-1, ROUGE-2, ROUGE-L vs Top-K
    # -------------------------------------------------------------
    if os.path.exists(exp_csv_path):
        df_exp = pd.read_csv(exp_csv_path)
        grouped = df_exp.groupby('top_k')[['rouge_1', 'rouge_2', 'rouge_l']].mean()

        fig, ax = plt.subplots(figsize=(8, 5))
        top_k_vals = grouped.index.tolist()
        x = np.arange(len(top_k_vals))
        width = 0.25

        bars1 = ax.bar(x - width, grouped['rouge_1'], width, label='ROUGE-1 (Unigram)', color='#2b5c8f', edgecolor='black', linewidth=0.8)
        bars2 = ax.bar(x, grouped['rouge_2'], width, label='ROUGE-2 (Bigram)', color='#4682b4', edgecolor='black', linewidth=0.8)
        bars3 = ax.bar(x + width, grouped['rouge_l'], width, label='ROUGE-L (LCS)', color='#79a3c7', edgecolor='black', linewidth=0.8)

        ax.set_xlabel('Konfigurasi Retrieval Top-K', fontweight='bold')
        ax.set_ylabel('Rata-rata Skor F1-Measure', fontweight='bold')
        ax.set_title('Perbandingan Kinerja Peringkasan RAG Berdasarkan Nilai Top-K', pad=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"Top-K = {k}" for k in top_k_vals])
        ax.set_ylim(0, max(grouped['rouge_1'].max() * 1.25, 0.45))
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)

        # Label angka di atas batang
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9, fontweight='semibold')

        fig1_path = os.path.join(output_dir, "fig1_rouge_comparison_topk.png")
        plt.savefig(fig1_path)
        plt.close()
        print(f" [OK] Disimpan: {fig1_path}")

        # -------------------------------------------------------------
        # 2. Grafik Latensi Inferensi & Retrieval vs Top-K
        # -------------------------------------------------------------
        lat_grouped = df_exp.groupby('top_k')[['retrieval_sec', 'generation_sec']].mean()
        
        fig, ax1 = plt.subplots(figsize=(8, 5))
        color_gen = '#c0392b'
        color_ret = '#27ae60'

        line1 = ax1.plot(top_k_vals, lat_grouped['generation_sec'], marker='o', color=color_gen, linewidth=2.5, markersize=8, label='Latensi Generasi LLM (detik)')
        ax1.set_xlabel('Konfigurasi Retrieval Top-K', fontweight='bold')
        ax1.set_ylabel('Waktu Generasi LLM (detik)', color=color_gen, fontweight='bold')
        ax1.tick_params(axis='y', labelcolor=color_gen)
        ax1.set_xticks(top_k_vals)
        ax1.grid(True, linestyle=':', alpha=0.6)

        for x_val, y_val in zip(top_k_vals, lat_grouped['generation_sec']):
            ax1.annotate(f'{y_val:.1f}s', xy=(x_val, y_val), xytext=(0, 6), textcoords="offset points",
                         ha='center', fontweight='bold', color=color_gen)

        # Sumbu sekunder untuk retrieval
        ax2 = ax1.twinx()
        line2 = ax2.plot(top_k_vals, lat_grouped['retrieval_sec'] * 1000, marker='s', color=color_ret, linewidth=2, linestyle='--', markersize=7, label='Latensi Retrieval Vektor (ms)')
        ax2.set_ylabel('Waktu Retrieval Vektor (milidetik)', color=color_ret, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor=color_ret)

        for x_val, y_val in zip(top_k_vals, lat_grouped['retrieval_sec'] * 1000):
            ax2.annotate(f'{y_val:.1f} ms', xy=(x_val, y_val), xytext=(0, -14), textcoords="offset points",
                         ha='center', fontweight='bold', color=color_ret)

        plt.title('Trade-Off Skalabilitas: Latensi Komputasi vs Nilai Top-K', pad=15, fontweight='bold')
        fig2_path = os.path.join(output_dir, "fig2_latency_vs_topk.png")
        plt.savefig(fig2_path)
        plt.close()
        print(f" [OK] Disimpan: {fig2_path}")

        # -------------------------------------------------------------
        # 3. Boxplot Sebaran Skor ROUGE per Top-K
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        data_to_plot = [df_exp[df_exp['top_k'] == k]['rouge_1'].values for k in top_k_vals]
        
        bp = ax.boxplot(data_to_plot, patch_artist=True, tick_labels=[f'Top-K={k}' for k in top_k_vals],
                        medianprops=dict(color='black', linewidth=1.5),
                        boxprops=dict(facecolor='#d4e6f1', color='#1b4f72', linewidth=1.2),
                        whiskerprops=dict(color='#1b4f72', linewidth=1.2),
                        capprops=dict(color='#1b4f72', linewidth=1.2),
                        flierprops=dict(marker='o', markerfacecolor='#e74c3c', markersize=6))
        
        ax.set_xlabel('Konfigurasi Top-K', fontweight='bold')
        ax.set_ylabel('Sebaran Skor ROUGE-1 F1', fontweight='bold')
        ax.set_title('Distribusi & Variabilitas Kualitas Ringkasan (ROUGE-1 Boxplot)', pad=15, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        fig5_path = os.path.join(output_dir, "fig5_rouge_boxplot_distribution.png")
        plt.savefig(fig5_path)
        plt.close()
        print(f" [OK] Disimpan: {fig5_path}")

    # -------------------------------------------------------------
    # 4. Grafik Pembuktian Lost in the Middle (Top-K = 10)
    # -------------------------------------------------------------
    if os.path.exists(litm_csv_path):
        df_litm = pd.read_csv(litm_csv_path)
        mean_scores = [
            df_litm['head_recall'].mean(),
            df_litm['middle_recall'].mean(),
            df_litm['tail_recall'].mean()
        ]
        labels = ['Head\n[Dokumen 1-3]', 'Middle\n[Dokumen 4-7]', 'Tail\n[Dokumen 8-10]']
        colors = ['#27ae60', '#e74c3c', '#2980b9']

        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(labels, mean_scores, color=colors, width=0.5, edgecolor='black', linewidth=0.8)
        ax.set_ylabel('Rasio Ketercakupan Informasi (Recall)', fontweight='bold')
        ax.set_title('Uji Bukti Empiris Fenomena "Lost in the Middle" (Top-K = 10)', pad=15, fontweight='bold')
        ax.set_ylim(0, max(mean_scores) * 1.35)
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}\n({height*100:.1f}%)',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Tambahkan anotasi kurva U-Shape
        ax.plot([0, 1, 2], mean_scores, color='#34495e', linestyle='--', linewidth=1.5, marker='o')

        fig3_path = os.path.join(output_dir, "fig3_lost_in_the_middle.png")
        plt.savefig(fig3_path)
        plt.close()
        print(f" [OK] Disimpan: {fig3_path}")

    # -------------------------------------------------------------
    # 5. Grafik Distribusi Error Analysis (5 Kategori)
    # -------------------------------------------------------------
    if os.path.exists(err_csv_path):
        df_err = pd.read_csv(err_csv_path)
        err_counts = {
            'Info Loss': df_err['has_info_loss'].sum(),
            'Redundancy': df_err['has_redundancy'].sum(),
            'Hallucination': df_err['has_hallucination'].sum(),
            'Incoherence': df_err['has_incoherence'].sum(),
            'Length Anomaly': df_err['has_length_anomaly'].sum()
        }
        total_runs = len(df_err)
        err_pcts = [err_counts[k] / total_runs * 100 for k in err_counts]
        cat_names = [
            'Information\nLoss',
            'Redundancy\n(Pengulangan)',
            'Hallucination\n(Halusinasi)',
            'Incoherence\n(Ketidaklogisan)',
            'Length\nAnomaly'
        ]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors_err = ['#e67e22', '#f39c12', '#c0392b', '#8e44ad', '#3498db']
        bars = ax.bar(cat_names, err_pcts, color=colors_err, width=0.55, edgecolor='black', linewidth=0.8)

        ax.set_ylabel('Persentase Kemunculan (%)', fontweight='bold')
        ax.set_title(f'Frekuensi Kategori Error Analysis Pada {total_runs} Uji Peringkasan RAG', pad=15, fontweight='bold')
        ax.set_ylim(0, max(max(err_pcts) * 1.3, 25))
        ax.grid(axis='y', linestyle='--', alpha=0.6)

        for bar, count in zip(bars, err_counts.values()):
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%\n({count}/{total_runs})',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='semibold')

        fig4_path = os.path.join(output_dir, "fig4_error_analysis_distribution.png")
        plt.savefig(fig4_path)
        plt.close()
        print(f" [OK] Disimpan: {fig4_path}")

    print("\n[SELESAI] Seluruh 5 figur ilmiah 300 DPI berhasil digenerasi!")

if __name__ == "__main__":
    generate_all_charts()
