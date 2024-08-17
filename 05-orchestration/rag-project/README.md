We create a new pipeline first. IN Data preparation we go to LOAD and Ingest.
We will ad block from the API. we will load https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/01-intro/documents.json
We can also EDIT the code of ingest. and amke any change that we want.
Now, we will chunk our documents. IN the upper part LOAD->transform->chunking
ADd a block to chunking -> custom code.

Add this code:

```
import re
from typing import Any, Dict, List


@transformer
def chunk_documents(data: List[Dict[str, Any]], *args, **kwargs):
    documents = []
    
    for idx, item in enumerate(data):
        course = item['course']
        
        for info in item['documents']:
            section = info['section']
            question = info['question']
            answer = info['text']
            
            # Generate a unique document ID
            document_id = ':'.join([re.sub(r'\W', '_', part) 
                for part in [course, section, question]]).lower()
            
            # Format the document string
            chunk = '\n'.join([
                f'course:\n{course}\n',
                f'section:\n{section}\n',
                f'question:\n{question}\n',
                f'answer:\n{answer}\n',
            ])
            
            documents.append(dict(
                chunk=chunk,
                document=info,
                document_id=document_id,
            ))

    print(f'Documents:', len(documents))
            
    return [documents]
```


Run the code and we see that we combined the course, section, question and answer into a one liner.

Now, we will tokenize the chunks. Go to Tokenization. We have many strategies for tokenizing text. CHoose, Lemmatization (Spacy). Add custome code:

```
from typing import Dict, List

import spacy


@transformer
def lemmatize_text(documents: List[Dict], *args, **kwargs) -> List[Dict]:
    count = len(documents)
    print('Documents', count)

    nlp = spacy.load('en_core_web_sm')

    data = []

    for idx, document in enumerate(documents):
        document_id = document['document_id']
        if idx % 100 == 0:
            print(f'{idx + 1}/{count}')

        # Process the text chunk using spaCy
        chunk = document['chunk']
        doc = nlp(chunk)
        tokens = [token.lemma_ for token in doc]

        data.append(
            dict(
                chunk=chunk,
                document_id=document_id,
                tokens=tokens,
            )
        )

    print('\nData', len(data))

    return [data]
```

This code prints out the progress. We can see that chunks were tokenized and we can also see the words.

After tokenizing each chunk, we can create embeddings out of them.
We will chose Spacy embeddings.

Embedding can take some time so we will add our own code that also prints out the progress.
When finished, we can see the chunk, the document ID and the embedding. Vector with 96 dimensions. 

The final stage is exporting, when data preparation. We will export the chunks and the embeddings into a vector database. 
Chose Export -> Vector Database. Add block Elasticsearch. Connection string: http://elasticsearch:9200 and then add the code:

```
import json
from typing import Dict, List, Tuple, Union

import numpy as np
from elasticsearch import Elasticsearch


@data_exporter
def elasticsearch(documents: List[Dict[str, Union[Dict, List[int], str]]], *args, **kwargs):
    connection_string = kwargs.get('connection_string', 'http://localhost:9200')
    index_name = kwargs.get('index_name', 'documents')
    number_of_shards = kwargs.get('number_of_shards', 1)
    number_of_replicas = kwargs.get('number_of_replicas', 0)
    dimensions = kwargs.get('dimensions')

    if dimensions is None and len(documents) > 0:
        document = documents[0]
        dimensions = len(document.get('embedding') or [])

    es_client = Elasticsearch(connection_string)

    print(f'Connecting to Elasticsearch at {connection_string}')

    index_settings = {
        "settings": {
            "number_of_shards": number_of_shards,
            "number_of_replicas": number_of_replicas,
        },
        "mappings": {
            "properties": {
                "chunk": {"type": "text"},
                "document_id": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": dimensions}
            }
        }
    }

    # Recreate the index by deleting if it exists and then creating with new settings
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
        print(f'Index {index_name} deleted')

    es_client.indices.create(index=index_name, body=index_settings)
    print('Index created with properties:')
    print(json.dumps(index_settings, indent=2))
    print('Embedding dimensions:', dimensions)

    count = len(documents)
    print(f'Indexing {count} documents to Elasticsearch index {index_name}')
    for idx, document in enumerate(documents):
        if idx % 100 == 0:
		        print(f'{idx + 1}/{count}')

        if isinstance(document['embedding'], np.ndarray):
            document['embedding'] = document['embedding'].tolist()

        es_client.index(index=index_name, document=document)

    return [[d['embedding'] for d in documents[:10]]]

```

this is exporting so we do not get something back. But the last line of code shows one line of what we exported.

Now that we exported our chunks and embeddings to a vector database, we have completed the data preparation for a RAG pipeline. We can optionally try out and query the databse for the chunks. Is is not required but we can try.

Go to Inference -> retrieval -> Iterative retrieval.
Add elasticsearch  template. Pase the conecttion string: http://elasticsearch:9200

add this code:


```
from typing import Dict, List, Union

import numpy as np
from elasticsearch import Elasticsearch, exceptions


SAMPLE__EMBEDDINGS = [
    [-0.1465761959552765, -0.4822517931461334, 0.07130702584981918, -0.25872930884361267, -0.1563894897699356, 0.16641047596931458, 0.24484659731388092, 0.2410498708486557, 0.008032954297959805, 0.17045290768146515, -0.009397129528224468, 0.09619587659835815, -0.22729521989822388, 0.10254761576652527, 0.016890447586774826, -0.13290464878082275, 0.11240798979997635, -0.11204371601343155, -0.057132963091135025, -0.011206787079572678, -0.007982085458934307, 0.279083788394928, 0.20115645229816437, -0.1427406221628189, -0.19398854672908783, -0.035979654639959335, 0.20723149180412292, 0.29891034960746765, 0.21407313644886017, 0.09746530652046204, 0.1671638935804367, 0.08161208778619766, 0.3090828061103821, -0.20648667216300964, 0.48498260974884033, -0.12691514194011688, 0.518856406211853, -0.26291757822036743, -0.0949832871556282, 0.09556109458208084, -0.20844918489456177, 0.2685297429561615, 0.053442806005477905, 0.05103180184960365, 0.1029752567410469, 0.04935301095247269, -0.11679927259683609, -0.012528933584690094, -0.08489680290222168, 0.013589601963758469, -0.32059246301651, 0.10357264429330826, -0.09533575177192688, 0.02984568662941456, 0.2793693542480469, -0.2653750777244568, -0.24152781069278717, -0.3563413619995117, 0.09674381464719772, -0.26155123114585876, -0.1397126317024231, -0.009133181534707546, 0.05972130224108696, -0.10438819974660873, 0.21889159083366394, 0.0694752112030983, -0.1312003880739212, -0.31072548031806946, -0.002836169209331274, 0.2468366175889969, 0.09420009702444077, 0.1284026801586151, -0.03227006644010544, -0.012532072141766548, 0.6650756597518921, -0.14863784611225128, 0.005239118821918964, -0.3317912817001343, 0.16372767090797424, -0.20166568458080292, 0.029721004888415337, -0.18536655604839325, -0.3608534038066864, -0.18234892189502716, 0.019248824566602707, 0.25257956981658936, 0.09671413153409958, 0.15569280087947845, -0.38228726387023926, 0.37017977237701416, 0.03356296569108963, -0.21182948350906372, 0.48848846554756165, 0.18350018560886383, -0.23519110679626465, -0.17464864253997803], [-0.18246106803417206, -0.36036479473114014, 0.3282334506511688, -0.230922132730484, 0.09600532799959183, 0.6859422326087952, 0.0581890344619751, 0.4913463294506073, 0.1536773443222046, -0.2965141832828522, 0.08466599136590958, 0.319297194480896, -0.15651769936084747, -0.043428342789411545, 0.014402368105947971, 0.16681505739688873, 0.22521673142910004, -0.2715776264667511, -0.11033261567354202, -0.04398636147379875, 0.3480629622936249, 0.11897992342710495, 0.8724615573883057, 0.10258488357067108, -0.5719427466392517, -0.03029855526983738, 0.23351268470287323, 0.20660561323165894, 0.575685441493988, -0.12116186320781708, 0.18459142744541168, -0.12865227460861206, 0.3948173522949219, -0.34464019536972046, 0.6699116230010986, -0.45167359709739685, 1.1505522727966309, -0.4498964548110962, -0.3248189687728882, -0.29674994945526123, -0.3570491075515747, 0.5436431765556335, 0.49576905369758606, -0.11180296540260315, -0.02045607566833496, -0.22768598794937134, -0.37912657856941223, -0.30414703488349915, -0.48289090394973755, -0.04158346354961395, -0.3547952473163605, 0.0687602087855339, 0.041512664407491684, 0.33524179458618164, 0.21826978027820587, -0.443082332611084, -0.5049593448638916, -0.5298929810523987, -0.02618088759481907, -0.2748631536960602, -0.1986193209886551, 0.35475826263427734, 0.22456413507461548, -0.29532068967819214, 0.25150877237319946, 0.243370920419693, -0.29938358068466187, -0.2128247618675232, -0.15292000770568848, -0.14813245832920074, -0.06183856353163719, -0.1251668632030487, 0.14256533980369568, -0.22781267762184143, 0.8101184964179993, 0.19796361029148102, 0.09104947745800018, -0.4860817790031433, 0.3078012764453888, -0.27373194694519043, 0.11800770461559296, -0.45869407057762146, 0.09508189558982849, -0.23971715569496155, -0.27427223324775696, 0.5139415264129639, 0.1871502846479416, 0.06647063046693802, -0.4054469168186188, 0.4751380681991577, 0.17067894339561462, 0.12443914264440536, 0.3577817678451538, 0.10574143379926682, -0.3181760311126709, -0.23804502189159393]
]


@data_loader
def search(*args, **kwargs) -> List[Dict]:
    """
    query_embedding: Union[List[int], np.ndarray]
    """
    
    connection_string = kwargs.get('connection_string', 'http://localhost:9200')
    index_name = kwargs.get('index_name', 'documents')
    source = kwargs.get('source', "cosineSimilarity(params.query_vector, 'embedding') + 1.0")
    top_k = kwargs.get('top_k', 5)
    chunk_column = kwargs.get('chunk_column', 'content')

    query_embedding = None
    if len(args):
        query_embedding = args[0]
    if not query_embedding:
        query_embedding = SAMPLE__EMBEDDINGS[0]

    if isinstance(query_embedding, np.ndarray):
        query_embedding = query_embedding.tolist()

    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": source,
                "params": {"query_vector": query_embedding},
            }
        }
    }

    print("Sending script query:", script_query)

    es_client = Elasticsearch(connection_string)
    
    try:
        response = es_client.search(
            index=index_name,
            body={
                "size": top_k,
                "query": script_query,
                "_source": [chunk_column],
            },
        )

        print("Raw response from Elasticsearch:", response)

        return [hit['_source'][chunk_column] for hit in response['hits']['hits']]
    
    except exceptions.BadRequestError as e:
        print(f"BadRequestError: {e.info}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []
```
This code contains a hardcoded sample embedding. 

Finally, lets create a schedule that will trigger the pipeline daily basis.
Go back to pipelines and click on the pipeline.
New trigger -> schedule 

Make it daily, with 3600 timeout that fails, skip run and create initiali pipeline checked. Start datetime, some day in the past. Save.
Now enable the trigger. 
It will start running and fail. That is because ew have the retrieval test. we need to remove it.

