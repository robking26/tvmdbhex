import io

from PIL import Image


def make_poster(bands: list[tuple[tuple[int, int, int], float]], size=(185, 278)) -> Image.Image:
    """Poster made of horizontal colour bands; each band covers `share` of the height."""
    img = Image.new("RGB", size)
    y = 0
    for i, (colour, share) in enumerate(bands):
        h = size[1] - y if i == len(bands) - 1 else round(size[1] * share)
        img.paste(colour, (0, y, size[0], y + h))
        y += h
    return img


def to_jpeg(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()
