from gemla.integrations.latent import make_synthetic_latents
from gemla.pipelines import GemlaLatentPipeline


def main() -> None:
    latents = make_synthetic_latents()

    pipe = GemlaLatentPipeline()
    result = pipe.fit_evaluate(latents)

    print(result.summary())


if __name__ == "__main__":
    main()