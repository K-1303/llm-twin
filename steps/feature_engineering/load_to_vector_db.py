from loguru import logger
from typing_extensions import Annotated
from zenml import step

from llm_engineering.application import utils
from llm_engineering.domain.base import VectorBaseDocument


@step
def load_to_vector_db(
    documents: Annotated[list, "documents"],
) -> Annotated[bool, "successful"]:
    logger.info(f"Loading {len(documents)} documents into the vector database.")

    grouped_documents = VectorBaseDocument.group_by_class(documents)
    for document_class, documents in grouped_documents.items():
        logger.info(f"Loading documents into {document_class.get_collection_name()}")
        for documents_batch in utils.misc.batch(documents, size=1):
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    if document_class.bulk_insert(documents_batch):
                        break
                    retry_count += 1
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.warning(f"Attempt {retry_count} failed for {document_class.get_collection_name()}: {str(e)}. Retrying...")
                    else:
                        logger.error(f"Failed to insert documents into {document_class.get_collection_name()} after {max_retries} attempts: {str(e)}")
                        return False

    return True
