# Use a t5 model to run summarization on the documents of ECB+
import pickle
import typer
from transformers import AutoModel

app = typer.Typer()


@app.command()
def summarize(mention_map_path, split, men_type="evt", model_name: str = "google-t5/t5-small",):
    """
    Parameters
    ----------
        mention_map_path: str
        split: str
        men_type: str
        model_name: str = "google-t5/t5-small",
    """
    mention_map = pickle.load(open(mention_map_path, "rb"))
    mentions = [men for men in mention_map.values() if men["men_type"] == men_type and men["split"] == split]

    summarizer = AutoModel.from_pretrained(model_name)


if __name__ == "__main__":
    app()