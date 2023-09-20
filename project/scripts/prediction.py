import typer


app = typer.Typer()


def run_llm_lh(dataset: str, split: str, lh_threshold: float):
    # GPT initialze

    # initialize the mention_map

    # call tp, fp, tn, fn = lh(dataset, split, lh_threshold)

    # prediction_pairs = tp + fp

    # similarity_matrix = llm_coref(prediction_pairs, mention_map, template, 'gpt-4', save_folder) -> List[int] = size(predicion_pairs)

    pass


def llm_coref(prediction_pairs: List[Tuple(str, str)],
              mention_map: Dict,
              template: str,
              gpt_version: str,
              save_folder: Path) -> List[int]:
    # use template
    # run prompt
    # parse the response to get 0 or 1
    pass


def evaluate(mention_map, prediction_pairs, similarity_matrix, tmp_folder='/tmp/'):
    # create the key file with gold clusters from mention map

    # run clustering using prediction_pairs and similarity_matrix

    # create a predictions key file

    # Evaluation on gold and prediction key files.
    pass


@app.command()
def run_llm_pipeline(dataset, split, gpt_version, template):
    pass


if __name__=="__main__":
    app()