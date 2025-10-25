from zenml import pipeline

from .digital_data_etl import digital_data_etl
from .feature_engineering import feature_engineering


@pipeline
def end_to_end_data(
    author_links: list[dict[str, str | list[str]]],
) -> None:
    wait_for_ids = []
    for author_data in author_links:
        last_step_invocation_id = digital_data_etl(
            user_full_name=author_data["user_full_name"], links=author_data["links"]
        )

        wait_for_ids.append(last_step_invocation_id)

    author_full_names = [author_data["user_full_name"] for author_data in author_links]
    wait_for_ids = feature_engineering(author_full_names=author_full_names)
