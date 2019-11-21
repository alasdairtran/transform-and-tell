import json

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from tell.client import CaptioningClient

from .extractor import ExtractError, extract_article

client = CaptioningClient(ip='localhost', port=5558, port_out=5559)


@csrf_exempt
def post_caption(request):
    query = json.loads(request.body)

    if not query['url'].strip():
        return JsonResponse({'error': 'The URL cannot be empty.'})
    if 'nytimes.com' not in query['url']:
        return JsonResponse({'error': 'The URL must come from nytimes.com'})

    try:
        article = extract_article(query['url'], int(query['pos']))
    except ExtractError as e:
        return JsonResponse({'error': str(e)})
    except Exception:
        return JsonResponse({'error': 'Cannot parse the article. Pick another URL.'})

    output = client.parse([article])[0]

    data = {
        'title': article['title'],
        'image_url': article['image_url'],
        'generated_caption': output['caption'],
        'true_caption': article['true_caption'],
        'start': output['start'],
        'before': output['before'],
        'after': output['after'],
    }
    return JsonResponse(data)
