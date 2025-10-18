import json
from typing import Any, Dict, Optional

from loguru import logger
import google.generativeai as genai

try:
    import boto3
except ModuleNotFoundError:
    logger.warning("Couldn't load AWS or SageMaker imports. Run 'poetry install --with aws' to support AWS.")


from llm_engineering.domain.inference import Inference
from llm_engineering.settings import settings


class LLMInferenceSagemakerEndpoint(Inference):
    """
    Class for performing inference using a SageMaker endpoint for LLM schemas.
    """

    def __init__(
        self,
        endpoint_name: str,
        default_payload: Optional[Dict[str, Any]] = None,
        inference_component_name: Optional[str] = None,
    ) -> None:
        super().__init__()

        genai.configure(api_key=settings.GOOGLE_API_KEY)

        model = genai.GenerativeModel(settings.GOOGLE_GEMINI_MODEL)

        # self.client = boto3.client(
        #     "sagemaker-runtime",
        #     region_name=settings.AWS_REGION,
        #     aws_access_key_id=settings.AWS_ACCESS_KEY,
        #     aws_secret_access_key=settings.AWS_SECRET_KEY,
        # )
        self.client = model
        self.endpoint_name = endpoint_name
        self.payload = default_payload if default_payload else self._default_payload()
        self.inference_component_name = inference_component_name

    def _default_payload(self) -> Dict[str, Any]:
        """
        Generates the default payload for the inference request.

        Returns:
            dict: The default payload.
        """

        return {
            "inputs": "How is the weather?",
            "parameters": {
                "max_new_tokens": settings.MAX_NEW_TOKENS_INFERENCE,
                "top_p": settings.TOP_P_INFERENCE,
                "temperature": settings.TEMPERATURE_INFERENCE,
                "return_full_text": False,
            },
        }

    def set_payload(self, inputs: str, parameters: Optional[Dict[str, Any]] = None) -> None:
        """
        Sets the payload for the inference request.

        Args:
            inputs (str): The input text for the inference.
            parameters (dict, optional): Additional parameters for the inference. Defaults to None.
        """

        self.payload["inputs"] = inputs
        if parameters:
            self.payload["parameters"].update(parameters)

    def inference(self) -> Dict[str, Any]:
        """
        Performs the inference request using the SageMaker endpoint.

        Returns:
            dict: The response from the inference request.
        Raises:
            Exception: If an error occurs during the inference request.
        """

        # try:
        #     logger.info("Inference request sent.")
        #     invoke_args = {
        #         "EndpointName": self.endpoint_name,
        #         "ContentType": "application/json",
        #         "Body": json.dumps(self.payload),
        #     }
        #     if self.inference_component_name not in ["None", None]:
        #         invoke_args["InferenceComponentName"] = self.inference_component_name
        #     response = self.client.invoke_endpoint(**invoke_args)
        #     response_body = response["Body"].read().decode("utf8")

        #     return json.loads(response_body)

        try:
            logger.info("Inference request sent to Gemini.")
            
            # Extract input text and parameters from payload
            input_text = self.payload["inputs"]
            generation_config = {
                "max_output_tokens": self.payload["parameters"].get("max_new_tokens", 1024),
                "temperature": self.payload["parameters"].get("temperature", 0.7),
                "top_p": self.payload["parameters"].get("top_p", 0.95),
            }

            # Generate response using Gemini
            response = self.client.generate_content(
                input_text,
                generation_config=generation_config
            )

            # Extract usage metadata
            usage_metadata = response.usage_metadata

            # Format response to match expected structure
            result = {
                "generated_text": response.text,
                "model": settings.GOOGLE_GEMINI_MODEL,
                "usage": {
                    "prompt_tokens": usage_metadata.prompt_token_count,
                    "completion_tokens": usage_metadata.candidates_token_count,
                    "total_tokens": usage_metadata.total_token_count
                }
            }

            return result

        except Exception:
            logger.exception("SageMaker inference failed.")

            raise
