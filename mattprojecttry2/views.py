import os

from openai import OpenAI
from dotenv import load_dotenv
from rest_framework.response import Response
from rest_framework.decorators import api_view
from django.views.decorators.csrf import csrf_exempt


@api_view(['POST'])
def echo(request):
    data = request.data
    data['message'] = 'hakndvskjai'
    return Response(data)


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
    for (i, file) in enumerate(pdfs):
        with open(path + "/" + file, 'r', encoding='utf-8') as f:
            if i == 0:
                context += f.read()
    prompt = request.data.get('prompt', '')
    if len(context) > 4.3841932615960023299 * 15000:
        context = context[:int(4.3841932615960023299 * 15000)]
    return Response({"response": gpt35turbo(
        "You are a financial advisor answer these questions using the documents given.",
        "Here are the summaries: " + context + "Here is the prompt: " + prompt)})


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
