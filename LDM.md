# Latent Diffusion Models
[CompVis Source Code](https://github.com/CompVis/latent-diffusion)

April 2022
- 1.45B latent diffusion LAION model was integrated into Huuggingface Spaces using Gradio.
  - You can try out the web demo.
  - 1.45B model trained on the LAION-400M database.

## How to Create Images beyond $256 \times 256$
For certain inputs, simply ruunning the model in a convolutional fashion on larger features than it was trained on can sometimes result in interesting results.
To try it out, tune the `H` and `W` arguments (which will be integer-divided by `8` in order to calculate the corresponding latent size), run the following command:
```
python scripts/txt2img.py --prompt "a sunset behind a mountain range, vector image" --ddim_eta 1.0 --n_samples 1 --n_iter 1 --H 384 --W 1024 --scale 5.0
```
This creates a sample of size $384 \times 1024$. Note, however, that controllability is reduced compared to the $256 \times 256$ setting.
![image](https://github.com/user-attachments/assets/99d64807-9c8e-4350-bede-7c8325ece012)


## Train Your Own LDMs
