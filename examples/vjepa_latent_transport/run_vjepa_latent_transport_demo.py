from pathlib import Path

from gemla.integrations.vjepa import (
    load_vjepa_embeddings,
    save_sample_vjepa_like_embeddings,
)
from gemla.pipelines import GemlaLatentPipeline


def main() -> None:
    embedding_path = Path("examples/vjepa_latent_transport/sample_vjepa_like_embeddings.npy")

    save_sample_vjepa_like_embeddings(embedding_path)

    embeddings = load_vjepa_embeddings(embedding_path)

    pipe = GemlaLatentPipeline()
    result = pipe.fit_evaluate(embeddings)

    print("GEMLA V-JEPA-Style Latent Transport Demo")
    print("----------------------------------------")
    print("Input embeddings were loaded from:")
    print(f"  {embedding_path}")
    print()
    print(result.summary())


if __name__ == "__main__":
    main()