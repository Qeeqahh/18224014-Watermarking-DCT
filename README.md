# Watermarking DCT Domain

Implementasi invisible watermark pada gambar menggunakan **Discrete Cosine Transform (DCT)**.

## Deskripsi

Project ini mengimplementasikan teknik *blind image watermarking* berbasis DCT — teknik yang sama yang digunakan pada kompresi JPEG. Watermark disisipkan di domain frekuensi sehingga tidak terlihat oleh mata manusia namun dapat diekstrak kembali secara komputasional.

## Cara Kerja

1. Gambar dibagi menjadi blok-blok **8×8 pixel**
2. Setiap blok ditransformasi ke domain frekuensi menggunakan **2D DCT**
3. Bit watermark disisipkan pada **koefisien mid-frequency [4][4]**
   - Bit `1` → koefisien dibuat **positif**
   - Bit `0` → koefisien dibuat **negatif**
4. Blok dikembalikan ke domain spasial menggunakan **Inverse DCT**
5. Ekstraksi: cukup cek tanda koefisien DCT[4][4]

## Demo

| Original | Watermarked | Perbedaan (×10) |
|---|---|---|
| ![original](images/original.png) | ![watermarked](images/watermarked.png) | (hampir tidak terlihat) |

## Quick Start

```bash
pip install numpy scipy pillow matplotlib
```

Jalankan notebook:
```bash
jupyter notebook watermarking_dct.ipynb
```

## Parameter

| Parameter | Default | Keterangan |
|---|---|---|
| `alpha` | 25 | Kekuatan watermark (lebih besar = lebih kuat tapi lebih terlihat) |
| `block_size` | 8 | Ukuran blok DCT (standar JPEG = 8) |

## Hasil

- **PSNR > 30 dB** — kualitas gambar tetap sangat baik
- Watermark berhasil diekstrak 100% akurat

## License

MIT
