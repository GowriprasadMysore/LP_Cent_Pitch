# LP_Cent_Pitch

**LP-based Pitch Estimation on the Cent Scale (JASA Express Letters, 2024)**

This repository contains a **lightweight, signal-processing-based implementation for fundamental frequency (f₀) estimation**, proposed in our JASA Express Letters paper:

> **R. Gowriprasad, T. Anand, R. Aravind, H. A. Murthy**  
> *Linear prediction on Cent scale for fundamental frequency analysis*  
> JASA Express Letters, 2024  
> DOI: https://doi.org/10.1121/10.0034516

---

## Why this work matters

Accurate pitch estimation is critical for music, speech, and audio AI systems.  
However, conventional LP and FFT-based methods operate on a **linear frequency scale**, which does not align with **human pitch perception**.

This work introduces:
- **Linear Prediction (LP) modeling on the Cent (log-pitch) scale**
- **Direct f₀ estimation from LP pole locations**
- **High precision with very low model order**
- **Unsupervised, CPU-efficient, real-time capable design**

The method is particularly effective for **music with continuous pitch variations and strong harmonics**, such as Indian Art Music, but is applicable to general audio analysis pipelines.

---

## Key features (Recruiter-focused)

- Signal-processing + perception-aware design (no training data)
- Works with approximate tonic (robust to tuning variations)
- Fewer parameters, lower compute than learning-based models
- Suitable for **real-time systems and embedded pipelines**
- Clean integration point for ML-based downstream models

---

## Method (high-level)

1. Compute short-time power spectrum  
2. Warp frequency axis to **Cent scale**  
3. Linearly resample in Cent domain  
4. Apply **LP all-pole spectral modeling**  
5. Estimate f₀ directly from **pole angles**  

Selective LP over musically relevant octaves yields sharper pitch peaks with fewer coefficients.

---

## Installation

```bash
git clone https://github.com/GowriprasadMysore/LP_Cent_Pitch.git
cd LP_Cent_Pitch
pip install -r requirements.txt
```

---

## Example usage

```bash
python demo_pitch_tracking.py --audio input.wav --tonic 220 --lp_order 6
```

Typical settings:
- Frame length: 100 ms  
- Hop size: 10 ms  
- LP order: 6–8  

---

## Evaluation summary

Evaluated on **Saraga Hindustani**, **Saraga Carnatic**, and **SCMS Synthetic CM** datasets.  
The proposed method **outperforms YIN** while using significantly fewer parameters (reported in the paper).

---

## Citation

```bibtex
@article{gowriprasad2024lpcent,
  title   = {Linear prediction on Cent scale for fundamental frequency analysis},
  author  = {Gowriprasad, R. and Anand, T. and Aravind, R. and Murthy, H. A.},
  journal = {JASA Express Letters},
  year    = {2024},
  doi     = {10.1121/10.0034516}
}
```

---

## License

Released for **research and academic use**.  
Please contact the authors for commercial usage.
