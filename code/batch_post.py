import os
from openai import OpenAI

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def openai_batch_api(file_path):
    """
    openai batch api 호출 함수
    :param file_path: jsonl 파일 경로
    :return: https://platform.openai.com/batches 여기서 결과 볼 수 있음.
    """
    batch_input_file = client.files.create(
        file=open(file_path, "rb"),
        purpose="batch"
    )

    batch_input_file_id = batch_input_file.id

    client.batches.create(
        input_file_id=batch_input_file_id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"description": "batch_4o_origin_random_fewshot.jsonl"})

openai_batch_api("2025_05_19_data/batch_4o_origin_random_fewshot.jsonl")