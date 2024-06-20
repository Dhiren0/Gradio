import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

# Load the processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    """
    Generate a caption for a given PIL Image object using the BLIP model.
    """
    try:
        inputs = processor(images=image, return_tensors="pt")
        outputs = model.generate(**inputs)
        caption = processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        return f"Error during caption generation: {str(e)}"

def caption_image(image):
    """
    Takes a PIL Image input and returns a caption.
    """
    try:
        caption = generate_caption(image)
        return caption
    except Exception as e:
        return f"An error occurred: {str(e)}"

# Define the Gradio interface
iface = gr.Interface(
    fn=caption_image,
    inputs=gr.inputs.Image(type="pil"),
    outputs="text",
    title="Image Captioning with BLIP",
    description="Upload an image to generate a caption."
)

if __name__ == "__main__":
    iface.launch()
