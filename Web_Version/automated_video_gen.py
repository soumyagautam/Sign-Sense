from moviepy.editor import ImageClip, concatenate_videoclips, AudioFileClip
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


def create_image_clips(image_filenames, duration):
    clips = []
    for filename in image_filenames:
        image = Image.open(filename)
        image = ensure_even_dimensions(image)
        image.save(filename)  # Save the resized image back to the same file
        clip = ImageClip(filename).set_duration(duration)
        clips.append(clip)
    return clips


def concatenate_clips(clips):
    return concatenate_videoclips(clips, method="compose")


def create_final_video(clips, output_filename, audio_filename=None):
    final_clip = concatenate_clips(clips)
    if audio_filename:
        audio = AudioFileClip(audio_filename)
        final_clip = final_clip.set_audio(audio)

    final_clip.write_videofile(
        output_filename,
        codec='libx264',
        audio_codec='aac',
        fps=24,
        preset='slow',
        ffmpeg_params=['-pix_fmt', 'yuv420p']
    )


if __name__ == "__main__":
    image_filenames = ["../sign_dataset/a.png", "../sign_dataset/b.png", "../sign_dataset/c.png"]
    image_clips = create_image_clips(image_filenames, duration=2)
    audio_filename = None
    create_final_video(image_clips, "final_video.mp4", audio_filename)
