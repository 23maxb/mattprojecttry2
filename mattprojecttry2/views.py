import os

from django.core.files.storage import default_storage
from django.http import JsonResponse
from openai import OpenAI
from dotenv import load_dotenv
from rest_framework.response import Response
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
import os


@api_view(['POST'])
def echo(request):
    data = request.data
    data['message'] = 'hakndvskjai'
    return Response(data)


@api_view(['POST'])
def upload_file(request):
    SOURCES_DIR = './mattprojecttry2/summariesAndDocuments/sources'
    # Check if the request has a file in the 'file' parameter
    if 'file' not in request.FILES:
        return JsonResponse({'error': 'No file provided.'}, status=400)

    file = request.FILES['file']
    file_name = file.name

    # Ensure the 'sources' directory exists
    os.makedirs(SOURCES_DIR, exist_ok=True)

    # Save the file
    file_path = os.path.join(SOURCES_DIR, file_name)
    with default_storage.open(file_path, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
    print("File added to sources: " + file_name)
    return JsonResponse({'message': f'File {file_name} uploaded successfully.'})


@csrf_exempt
def gpt35turbo(content, prompt):
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": content},
            {"role": "user", "content": prompt}
        ]
    )
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content


@api_view(['POST'])
def gpt35turboQuestion(data):
    return gpt35turbo(data['context'], data['prompt'])


@api_view(['POST'])
def realQuestion(request):
    pdfs = pickPDF(request.data.get('prompt', ''))
    path = "./mattprojecttry2/summariesAndDocuments/text"
    context = ""
    # remove spaces from pdfs list
    pdfs = [x for x in pdfs if x != '']
    for (i, file) in enumerate(pdfs):
        with open(path + "/" + file, 'r', encoding='utf-8') as f:
            if i == 0:
                context += f.read()
    prompt = request.data.get('prompt', '')
    if len(context) > 4.3841932615960023299 * 15000:
        context = context[:int(4.3841932615960023299 * 15000)]
    return Response({"response": gpt35turbo(
        "You are a financial advisor answer these questions using the documents given.",
        "Here are the summaries: " + context + "Here is the prompt: " + prompt), "\nSources": pdfs})


@csrf_exempt
def pickPDF(prompt):
    load_dotenv()
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    summaries = ""
    path = "./mattprojecttry2/summariesAndDocuments/summaries"
    print(os.listdir("./"))
    dir_list = os.listdir(path)
    for (i, file) in enumerate(dir_list):
        with open(path + "/" + file, 'r') as f:
            summaries += dir_list[i] + ": " + f.read()

    result = gpt35turbo(
        "You are a resource librarian. I will give you a few summaries of documents and you must pick the most relevant documents to the questions. Do not answer the prompt directly. Instead, pick the most relevant documents and give their names ranked by order of relevance. Summaries will be seperated by the names of the files. Only respond with an array of the file names given. Here is an example: [\"GeoFragcosts.txt\", \"InvestmentandFDI.txt\"] Here is the prompt: " + prompt,
        "Here are the summaries: " + summaries)
    result = result[result.index('[') + 1:result.index(']')].replace("\n", "").replace("\"", "").split(",")
    for i in range(len(result)):
        result[i] = result[i].strip()
    return result
