# Text_to_Image ML model

<h2>Introduction</h2>


This project is centered around building a text-to-image generation application using machine learning, specifically leveraging generative models. The idea was to develop a system that can take a natural language prompt — such as “a cat playing guitar under the moonlight” — and generate a corresponding visual image. The application was intended to be lightweight, simple to implement, and executable on a MacBook Air M1 with limited storage and compute resources. Therefore, instead of using full-scale models like Stable Diffusion v1.5 (which can exceed 4 GB in size), the project uses stabilityai/sd-turbo, a lightweight and optimized diffusion model available via Hugging Face.

The model works by using diffusion-based techniques to gradually turn noise into coherent images that align with the given textual prompt. This is done via a pipeline built on PyTorch and Hugging Face’s diffusers library. The sd-turbo model is capable of producing reasonable quality images while requiring significantly less memory (around 1.5 GB), making it ideal for local experimentation without a high-end GPU. It is fast and optimized for inference, and can be run with Apple’s Metal Performance Shaders (mps) to take advantage of the MacBook’s M1 GPU.
<h2>Initial configuration</h2>
To begin, a dedicated folder named text_to_image was created to organize the project files. A conda virtual environment named text2img was then created and activated to isolate dependencies from the rest of the system. This was done using the following commands in the terminal:

```
conda create -n text2img python=3.9
conda activate text2img
```
Once inside the environment, the necessary libraries were installed. These included the diffusers library for the diffusion model pipeline, PyTorch along with torchvision and torchaudio for the underlying ML framework, and matplotlib for visualizing the output images in the notebook. The commands used were
<h2>Installing the dependencies</h2>

```
pip install diffusers
pip install torch torchvision torchaudio
pip install matplotlib
```
After setting up the environment, a Hugging Face account was used to authenticate access to the model hub. 
The Jupyter Notebook, launched from within the project folder, contained the core logic. The model was loaded using diffusers and moved to the mps device to utilize the M1 GPU. A helper function named generate_image(prompt) was defined to generate and display an image from any user-supplied prompt. A basic version of the function looked like this:
```python
from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt

pipe = StableDiffusionPipeline.from_pretrained("stabilityai/sd-turbo")
pipe.to("mps")  # for MacBook M1 GPU acceleration

def generate_image(prompt):
    image = pipe(prompt).images[0]
    plt.imshow(image)
    plt.axis("off")
    plt.show()
```

Throughout the development, the focus was on simplicity, offline capability, and lightweight performance. All packages and models were kept minimal in size and locally installed to avoid dependency issues or network interruptions during runtime. The final application ran entirely inside the Jupyter Notebook environment and did not require setting up a web backend or FastAPI server.
<br><br>
<h2>Conclusion</h2>
In summary, this project successfully demonstrates a simplified text-to-image generation ML pipeline using a lightweight model suited for personal machines. It required no specialized GPUs, was fully notebook-based, and served as a practical and hands-on application of generative AI for visual content creation from text. The result is a modular, easy-to-understand implementation that strikes a balance between functionality and feasibility on low-resource systems.
