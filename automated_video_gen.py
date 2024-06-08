import os
from PIL import Image


def ensure_even_dimensions(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')

    width, height = image.size
    if width % 2 != 0:
        width += 1
    if height % 2 != 0:
        height += 1
    return image.resize((width, height))


def create_video_with_ffmpeg(image_filenames, output_filename, duration_per_image):
    for i, filename in enumerate(image_filenames):
        image = Image.open(filename)
        image = ensure_even_dimensions(image)
        image.save(f"temp_{i}.png")

    # Construct ffmpeg command
    ffmpeg_cmd = (
        f"ffmpeg -y -framerate 1/{duration_per_image} -i temp_%d.png "
        f"-c:v libx264 -r 30 -pix_fmt yuv420p {output_filename}"
    )
    os.system(ffmpeg_cmd)

    # Clean up temporary files
    for i in range(len(image_filenames)):
        os.remove(f"temp_{i}.png")


# Create the video using ffmpeg
if __name__ == "__main__":
    image_filenames = ["sign_dataset/a.png", "sign_dataset/b.png", "sign_dataset/c.png"]
    create_video_with_ffmpeg(image_filenames, "final_video.mp4", duration_per_image=2)
