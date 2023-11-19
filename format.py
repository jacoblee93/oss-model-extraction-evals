import mailbox
from langchain.schema import Document
from langchain.document_transformers import Html2TextTransformer
 
# Replace 'input.mbox' with your MBOX file name
mbox_file = '../Mail/Spam.mbox'
 
mbox = mailbox.mbox(mbox_file)
html2text = Html2TextTransformer()
 
for i, message in enumerate(mbox):
    print(i)
    with open(f'./dataset/email_{i}.eml', 'wb') as f:
        raw_doc = Document(page_content=message.as_string())
        transformed_docs = html2text.transform_documents([raw_doc])
        f.write(transformed_docs[0].page_content.replace("jacoblee93", "jacob").replace("= ", "").replace("=E2=80=94", "").encode("utf-8"))