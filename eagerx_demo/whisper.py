import whisper
from whisper.tokenizer import get_tokenizer
from eagerx_demo.cliport.tasks import put_block_in_bowl
import torch


if __name__ == "__main__":
    ckpt = "base.en"
    device = torch.device("cpu")
    model = whisper.load_model(ckpt, device=device)
    tokenizer = get_tokenizer(multilingual=False)
    task = put_block_in_bowl.PutBlockInBowlUnseenColors()

    # get all possible language commands and corresponding tokens
    all_color_names = task.get_colors()
    lang_commands = []
    language_tokens = []
    for color_1 in all_color_names:
        for color_2 in all_color_names:
            lang_command = task.lang_template.format(pick=color_1, place=color_2)
            lang_commands.append(lang_command)
            lang_command_tokens = tokenizer.encode(lang_command)
            language_tokens.append(lang_command_tokens)

    from time import time

    start = time()
    result = model.transcribe(
        "delme_rec_unlimited_fu8h3_j2.wav",
        initial_prompt="these are the only possible sentences: " + ", ".join([str(command) for command in lang_commands]),
    )
    print(f"Time: {time() - start}")

    print(f"{result['text']}")
