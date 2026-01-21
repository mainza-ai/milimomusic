from heartlib import HeartMuLaGenPipeline
import argparse
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--version", type=str, default="3B")
    parser.add_argument("--lyrics", type=str, default="./assets/lyrics.txt")
    parser.add_argument("--tags", type=str, default="./assets/tags.txt")
    parser.add_argument("--save_path", type=str, default="./assets/output.mp3")

    parser.add_argument("--max_audio_length_ms", type=int, default=240_000)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    pipe = HeartMuLaGenPipeline.from_pretrained(
        args.model_path,
        device=device,
        dtype=torch.bfloat16,
        version=args.version,
    )
    with torch.no_grad():
        pipe(
            {
                "lyrics": args.lyrics,
                "tags": args.tags,
            },
            max_audio_length_ms=args.max_audio_length_ms,
            save_path=args.save_path,
            topk=args.topk,
            temperature=args.temperature,
            cfg_scale=args.cfg_scale,
        )
    print(f"Generated music saved to {args.save_path}")
