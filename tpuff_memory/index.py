import os
import turbopuffer as tpuf
import pandas as pd


TPUF_API_KEY = os.getenv("TPUF_API_KEY")

tpuf.api_key = TPUF_API_KEY
tpuf.api_base_url = "https://gcp-us-central1.turbopuffer.com"



class TurboPufferIndex:

    def __init__(self, name: str):
        self.ns = tpuf.Namespace(name)


    def index_docs(self, docs):
        """
        Index the documents into TurboPuffer.
        """
        self.ns.write(
            upsert_columns={"content": docs,
                            "id": [_id for _id in range(len(docs))]},
            schema={
                "content": {
                    "type": "string",
                    "full_text_search": True,
                }
            }
        )
        count = self.ns.approx_count()
        print(f"Indexed {count} documents into TurboPuffer.")

