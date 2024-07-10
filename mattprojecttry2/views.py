from rest_framework.decorators import api_view
from rest_framework.response import Response


@api_view(['POST'])
def echo(request):
    data = request.data
    data['message'] = 'hakndvskjai'
    return Response(data)
