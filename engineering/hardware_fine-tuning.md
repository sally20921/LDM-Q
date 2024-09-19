

## Pre-Training Speedup

For pre-training, the training speed typically increases with increasing batch size. THis is also true for the diffusion model.


- ZeRO and Gemini modules optimizes memory consumption.
- Replace Cross-Attention module with Flash-Attention helps reduce GPU memory. 


- Users can train diffusion models on consumer-level GPUs such as RTX 3080.
- Batch size can be increased to 256 on computing-specific GPUs such as A100. 
- This might speed up training by 6.5 times.

- Normal spec of Stable Diffusion Training
 - Device: A100
 - # GPU: 4
 - Batch size: 32, 64
 - Precision: FP32
 - GPU RAM (GB): 67

## Friendly Fine-Tuning

Stable Diffusion pre-training uses a LAION-5B dataset with 585 billion image-text pairs, which requires 240TB of storage. 

Combined with the complexity of the model, it is inevitable that the cost of pre-training is extremely high.
The Stability team spent over $50 million ofr a supercomputer of 4,000 A100 GPUs. 

A much more practical option for producing AIGC (AI-generated content) is using open-source pre-trained models weights that fine-tune downstream personalization tasks. 

Fine-tuning also needs an RTX 3090 or 4090 top-end consumer graphics card to start. At the same time, many open-source training frameworks do not give a complete training configuration and script at this stage, requiring users to spend extra time on tedious tasks and debugging.


















