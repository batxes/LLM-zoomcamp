if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test

import hashlib

def generate_document_id(doc):
    combined = f"{doc['section']}-{doc['question']}-{doc['text'][:10]}"
    hash_object = hashlib.md5(combined.encode())
    hash_hex = hash_object.hexdigest()
    document_id = hash_hex[:8]
    return document_id

@transformer
def transform(data):

    documents = []

    for course_dict in data:
        print (course_dict)
        course_dict['document_id'] = generate_document_id(course_dict)
        documents.append(course_dict)
        print (course_dict)

    print(len(documents))

    return documents



@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'