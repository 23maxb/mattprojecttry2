from django.urls import path
from .views import echo, realQuestion, gpt35turboQuestion, upload_file

urlpatterns = [
    path('echo/', echo, name='echo'),
    path('gpt35turboQuestion/', gpt35turboQuestion, name='gpt35turboQuestion'),
    path('realQuestion/', realQuestion, name='realQuestion'),
    path('upload_file/', upload_file, name='upload_file')
]
