import os
from django.core.files.storage import default_storage
from django.http import JsonResponse
from openai import OpenAI
from dotenv import load_dotenv
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from pinecone import Pinecone, ServerlessSpec
import time
from sentence_transformers import SentenceTransformer
import torch

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

# Initialize SentenceTransformer model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

@api_view(['POST'])
def echo(request):
    data = request.data.copy()
    data['message'] = 'hakndvskjai'
    return Response(data)

@api_view(['POST'])
def search(request):
    index_name = "quickstart"
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=model.get_sentence_embedding_dimension(),
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    return Response({"message": "Search index created or already exists."})

@api_view(['POST'])
def realQuestion(request):
    index_name = 'semantic-search'

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            index_name,
            dimension=model.get_sentence_embedding_dimension(),
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-east-1')
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

    index = pc.Index(index_name)

    query = request.data.get('prompt', '')
    xq = model.encode(query).tolist()
    xc = index.query(vector=xq, top_k=10, include_metadata=True)

    context = ""
    for result in xc['matches']:
        context += f"{round(result['score'], 2)}: {result['metadata']['text']}\n"

    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a geoeconomic analyst that uses evidence in documents to answer questions. Using the given documents answer the question given."},
            {"role": "user", "content": context + query}
        ]
    )

    res = completion.choices[0].message.content
    return Response({"response": res, "sources": ""})

@api_view(['POST'])
def upload_file(request):
    SOURCES_DIR = './mattprojecttry2/summariesAndDocuments/sources'
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided.'}, status=400)

    file = request.FILES['file']
    file_name = file.name
    os.makedirs(SOURCES_DIR, exist_ok=True)
    file_path = os.path.join(SOURCES_DIR, file_name)

    with default_storage.open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)

    print(f"File added to sources: {file_name}")
    return JsonResponse({'message': f'File {file_name} uploaded successfully.'})

def gpt35turbo(content, prompt):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": prompt}
        ]
    )
    return completion.choices[0].message.content

@api_view(['POST'])
def gpt35turboQuestion(request):
    data = request.data
    return Response({"response": gpt35turbo(data['context'], data['prompt'])})

@api_view(['POST'])
def realQuestion2(request):
    pdfs = pickPDF(request.data.get('prompt', ''))
    path = "./mattprojecttry2/summariesAndDocuments/text"
    context = ""
    pdfs = [x for x in pdfs if x.strip()]

    for file in pdfs:
        with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
            context += f.read()

    prompt = request.data.get('prompt', '')
    context = context[:int(4.3841932615960023299 * 15000)]  # Limit context size

    response = gpt35turbo(
        "You are a financial advisor. Answer these questions using the documents given.",
        f"Here are the summaries: {context}\nHere is the prompt: {prompt}"
    )

    return Response({"response": response, "sources": pdfs})

@csrf_exempt
def pickPDF(prompt):
    summaries = ""
    path = "./mattprojecttry2/summariesAndDocuments/summaries"

    for file in os.listdir(path):
        with open(os.path.join(path, file), 'r') as f:
            summaries += f"{file}: {f.read()}\n"

    result = gpt35turbo(
        "You are a resource librarian. Pick the most relevant documents to the question. Only respond with an array of the file names given. Example: [\"GeoFragcosts.txt\", \"InvestmentandFDI.txt\"]",
        f"Here are the summaries: {summaries}\nHere is the prompt: {prompt}"
    )

    result = result.strip("[]").replace("\"", "").split(",")
    return [r.strip() for r in result if r.strip()]