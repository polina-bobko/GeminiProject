import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path


# 1. Выбор устройства
def setup_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# 2. Загрузка Stable Diffusion
def load_model(device):
    print("Loading Stable Diffusion model...")

    model_id = "stabilityai/sd-turbo"

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32
    )

    pipe = pipe.to(device)

    print("Model loaded successfully!")
    return pipe


# 3. Генерация изображения
def generate_image(pipe, prompt, num_steps=4):
    print(f"\nGenerating: {prompt}")

    image = pipe(
        prompt=prompt,
        num_inference_steps=num_steps,
        guidance_scale=0.0,
        width=512,
        height=512
    ).images[0]

    return image


# 4. Сохранение изображения
def save_image(image, filename):
    image.save(filename)
    print(f"Saved: {filename}")


# 5. Основная программа
def main():
    device = setup_device()
    print(f"Using device: {device}")

    pipe = load_model(device)

    output_dir = Path("generated_images")
    output_dir.mkdir(exist_ok=True)

    # Обычные промпты
    original_prompts = [
        "A golden retriever sitting on green grass",

        "A cozy Scandinavian cafe with large windows, "
        "wooden furniture and green plants",

        "An ancient magical library inside a medieval castle, "
        "floating books and glowing blue crystals"
    ]

    # Улучшенные промпты
    improved_prompts = [
        "A photorealistic golden retriever sitting in a green meadow, "
        "golden hour sunlight, shallow depth of field, "
        "detailed fur, professional photography",

        "A photorealistic Scandinavian cafe interior, "
        "oak furniture, large windows overlooking a rainy street, "
        "warm pendant lights, indoor plants, cozy atmosphere, "
        "wide-angle photography",

        "An ancient magical library inside a medieval castle, "
        "floating leather-bound books, glowing blue crystals, "
        "moonlight through stained glass windows, "
        "dramatic volumetric lighting, intricate details, "
        "fantasy concept art"
    ]

    # Генерация обычных изображений
    for i, prompt in enumerate(original_prompts):
        try:
            image = generate_image(pipe, prompt)

            filename = output_dir / f"original_{i + 1}.png"
            save_image(image, filename)

        except Exception as e:
            print(f"Error generating image: {e}")

    # Генерация улучшенных изображений
    for i, prompt in enumerate(improved_prompts):
        try:
            image = generate_image(pipe, prompt)

            filename = output_dir / f"improved_{i + 1}.png"
            save_image(image, filename)

        except Exception as e:
            print(f"Error generating image: {e}")

    print("\nAll images generated!")


if __name__ == "__main__":
    main()
