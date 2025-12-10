from enum import Enum

class ContentType(Enum):
    TEXT = "text"
    IMAGE_URL = "image_url"
    INPUT_IMAGE = "input_url"
    INPUT_AUDIO = "input_audio"
    OUTPUT_AUDIO = "output_audio"